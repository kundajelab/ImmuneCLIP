import torch
import numpy as np
import torch.nn as nn
import re

from .utils import get_sequence_embeddings, insert_spaces, get_attention_mask, apply_masking_seq
from .swe_pooling import SWE_Pooling

class EpitopeEncoderESM(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, model_config, hidden_dim=1024, device='cpu'):
        super().__init__()

        self.ln_config = ln_cfg
        self.model_config = model_config
        self.projection_dim = projection_dim

        if self.model_config.receptor_model_name == 'esm3':
            from .lora import setup_peft_esm3
            from .configs import peft_config_esm3

            # load the LoRA adapted ESM-3 Model here:
            self.esm_lora, self.esm_tokenizer = setup_peft_esm3(peft_config_esm3, ln_cfg.no_lora)
        else:
            from .lora import setup_peft_esm2
            from .configs import peft_config_esm2

            # load the LoRA adapted ESM-2 Model here:
            self.esm_lora, self.esm_tokenizer = setup_peft_esm2(peft_config_esm2, ln_cfg.no_lora, ln_cfg.regular_ft)

        # For ESM2, we need linker to represent multimers
        self.linker_size = 25
        self.gly_linker = 'G'*self.linker_size

        if self.projection_dim:
            if hidden_dim:
                print("Using multi-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(hidden_dim, projection_dim),
                                )
                # Initialize the projection head weights
                nn.init.kaiming_uniform_(self.proj_head[0].weight)
                nn.init.kaiming_uniform_(self.proj_head[-1].weight)
            else:
                print("Using single-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, projection_dim),
                                    nn.LayerNorm(projection_dim),
                                )
                # Initialize the projection head weights
                nn.init.kaiming_uniform_(self.proj_head[0].weight)
        else:
            print("NOT using projection head")
        
        if self.ln_config.swe_pooling:
            self.swe_pooling = SWE_Pooling(d_in=input_dim, num_ref_points=512, num_slices=projection_dim)

            self.proj_head = nn.Sequential(
                                nn.Linear(projection_dim, projection_dim // 2),
                                nn.LayerNorm(projection_dim // 2),
                            )   

        self.device = device

            
    def forward(self, x, mask):
        seqs_tokens = self.process_seqs(x, mask, mask_prob=self.ln_config.mask_prob)
        
        if self.model_config.receptor_model_name == 'esm3':
            outputs = self.esm_lora(sequence_tokens=seqs_tokens['input_ids']).embeddings
        else:
            outputs = self.esm_lora(**seqs_tokens)

        if self.ln_config.swe_pooling:
            assert self.ln_config.include_mhc == False, "SWE pooling not supported for MHC sequences yet" # TODO: implement for MHC sequences
            # for SWE pooling:
            attn_mask = get_attention_mask(seqs_tokens, is_sep=False, is_cls=False)
            # attn_mask = get_attention_mask(seqs_tokens)
            if isinstance(outputs, dict):
                outputs = outputs['last_hidden_state']
            elif self.model_config.receptor_model_name == 'esm3':
                pass
            else:
                outputs = outputs.last_hidden_state
            seq_embeds = self.swe_pooling(outputs, attn_mask)
        else:
            # for regular mean pooling
            epitope_mask = None
            if self.ln_config.include_mhc:
                epitope_seqs, mhca_seqs, mhcb_seqs = x
                epitope_mask = torch.zeros_like(seqs_tokens['attention_mask'])
                for i, (seq, mhcA, mhcB) in enumerate(zip(epitope_seqs, mhca_seqs, mhcb_seqs)):
                    # assumes no special tokens
                    epitope_mask[i, len(mhcA) + self.linker_size : len(mhcA) + self.linker_size + len(seq)] = 1

            if self.model_config.receptor_model_name == 'esm3':
                outputs = {'last_hidden_state': outputs}
            seq_embeds = get_sequence_embeddings(seqs_tokens, outputs, is_sep=False, is_cls=False, epitope_mask=epitope_mask)

        if self.projection_dim:
            return self.proj_head(seq_embeds)
        else:
            return seq_embeds
    
    def process_seqs(self, inputs, mask, mask_prob=0.15):
        '''
        input: list of epitope sequences or epitope-mhc array (3,N) where N is the number of samples

        if self.include_mhc = True, expecting input to be list containing tuples of epitope and MHC sequences in form
        (epitope_seq, mhc.a_seq, mhc.b_seq)

        if self.include_mhc = False, expecting input to be list of strings of epitope sequences
        '''
        if self.ln_config.include_mhc:
            epitope_seqs, mhca_seqs, mhcb_seqs = inputs

            if self.ln_config.mhc_groove_only:
                # keep only A1+A2 domains (roundly AA 0-180) for class 1 MHCs, and A1+B1 domains (rougly AA 0-90) for class 2 MHCs
                for i, (mhcA, mhcB) in enumerate(zip(mhca_seqs, mhcb_seqs)):
                    if mhcB == "B2M":
                        mhca_seqs[i] = mhcA[:180]
                    else:
                        mhca_seqs[i] = mhcA[:90]
                        mhcb_seqs[i] = mhcB[:90]

            # Create the pMHC sequence in the order [mhcA ..G.. epitope ..G.. mhcB]
            seqs = [
                (
                    f"{mhcA}{self.gly_linker}{seq}{self.gly_linker}{mhcB}"
                    if mhcB != "B2M" else
                    f"{mhcA}{self.gly_linker}{seq}"
                )
                for seq, mhcA, mhcB in zip(epitope_seqs, mhca_seqs, mhcb_seqs)
            ]

            # marking where the Glycine linker starts
            # linker between mhcA and epitope
            attn_starts = [(i, len(mhcA)) for i, mhcA in enumerate(mhca_seqs)]
            # linker between epitope and mhcB
            attn_starts.extend([
                (i, len(mhcA) + self.linker_size + len(seq))
                for i, (seq, mhcA, mhcB) in enumerate(zip(epitope_seqs, mhca_seqs, mhcb_seqs)) if mhcB != "B2M"
            ])
        else:
            seqs = inputs

        # removing special tokens since epitopes are protein fragments (peptides)
        seqs_tokens = self.esm_tokenizer(seqs, return_tensors="pt", add_special_tokens=False, padding=True)

        if mask:
            if self.ln_config.include_mhc:
                mask_regions = [np.zeros(len(seq), dtype=bool) for seq in seqs]
                for i, (seq, mhcA, mhcB) in enumerate(zip(epitope_seqs, mhca_seqs, mhcb_seqs)):
                    # always include epitope sequence for random masking
                    epitope_offset = len(mhcA) + self.linker_size
                    mask_regions[i][epitope_offset : epitope_offset + len(seq)] = True
                    # if class I MHC, only apply random masks to A1+A2 domains (rougly AAs 0-180)
                    if mhcB == "B2M":
                        mask_regions[i][0 : min(180, len(mhcA))] = True

                    # if class II MHC, only apply random masks to A1+B1 domains (rougly AAs 0-90 for each)
                    else:
                        mask_regions[i][0 : min(90, len(mhcA))] = True
                        beta_offset = len(mhcA) + self.linker_size + len(seq) + self.linker_size
                        mask_regions[i][beta_offset : min(beta_offset+90, beta_offset+len(mhcB))] = True
            else:
                mask_regions = True # seqs is just the epitope, so all tokens can be masked

            # masking the sequences for training
            seqs, attn_mask_indices = apply_masking_seq(seqs, mask_token='<mask>', mask_regions=mask_regions, p=mask_prob)
            indices_tensor = torch.tensor(attn_mask_indices, dtype=torch.long)
            if len(indices_tensor) > 0:
                seqs_tokens['attention_mask'][indices_tensor[:, 0], indices_tensor[:, 1]] = 0.

        # if necessary, masking the linker region
        if self.ln_config.include_mhc:
            for i, start in attn_starts:
                seqs_tokens['attention_mask'][i, start:start+self.linker_size] = 0.

        return seqs_tokens.to(self.device)

class EpitopeEncoderOneHot(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, model_config, device='cpu'):
        super().__init__()

        self.ln_config = ln_cfg
        self.projection_dim = projection_dim

        if self.projection_dim:
            print("Using single-layer projection head")
            self.proj_head = nn.Sequential(
                                nn.Linear(input_dim, projection_dim),
                                nn.LayerNorm(projection_dim),
                            )
        else:
            assert False, "Projection head must be used with one-hot encoding!"
        
        # Define the amino acid to index mapping
        self.amino_acid_to_index = {
                'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
                'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
                'X': 20  # Unknown amino acid
            }

        self.device = device

    def forward(self, x, mask):
        seqs = x
        seqs_onehot = self.create_padded_one_hot_tensor(seqs, len(self.amino_acid_to_index))

        proj_output = self.proj_head(seqs_onehot)

        # average the projected embeddings by seq length:
        seq_lens = torch.sum(seqs_onehot, dim=(1, 2))
        # Create a mask with shape (batch_size, max_seq_length)
        seq_mask = torch.arange(proj_output.size(1)).unsqueeze(0).to(self.device) < seq_lens.unsqueeze(-1)
        seq_mask = seq_mask.unsqueeze(2)  # Shape (batch_size, max_seq_length, 1)
        # Sum the embeddings across the sequence length dimension using the mask
        masked_embeddings = proj_output * seq_mask
        sum_embeddings = masked_embeddings.sum(dim=1)

        # Divide by the true sequence lengths to get the average
        avg_embeddings = sum_embeddings / seq_lens.unsqueeze(1)#.to(embeddings.device)
        
        return avg_embeddings


    # @staticmethod
    def encode_amino_acid_sequence(self, sequence):
        """ Convert an amino acid sequence to a list of indices. """
        return [self.amino_acid_to_index[aa] for aa in sequence]

    # @staticmethod
    def one_hot_encode_sequence(self, sequence, vocab_size):
        """ One-hot encode a single sequence. """
        encoding = np.zeros((len(sequence), vocab_size), dtype=int)
        for idx, char in enumerate(sequence):
            encoding[idx, char] = 1
        return encoding

    # @staticmethod
    def pad_sequences(self, encoded_sequences, max_length):
        """ Pad the encoded sequences to the maximum length. """
        padded_sequences = []
        for seq in encoded_sequences:
            padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=0)
            padded_sequences.append(padded_seq)
        return np.array(padded_sequences)

    # @staticmethod
    def create_padded_one_hot_tensor(self, sequences, vocab_size):
        """ Convert a batch of sequences to a padded one-hot encoding tensor. """
        # Encode and one-hot encode each sequence
        encoded_sequences = [self.one_hot_encode_sequence(self.encode_amino_acid_sequence(seq), vocab_size) for seq in sequences]
        
        # Determine the maximum sequence length
        max_length = max(len(seq) for seq in sequences)
        
        # Pad the sequences
        padded_sequences = self.pad_sequences(encoded_sequences, max_length)
        
        # Convert to a PyTorch tensor
        padded_tensor = torch.tensor(padded_sequences, dtype=torch.float32)
        
        return padded_tensor.to(self.device)
    

class TCREncoderTCRBert(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, hidden_dim=1024, device='cpu'):
        super().__init__()
        from .lora import setup_peft_tcrbert
        from .configs import peft_config_tcrbert

        self.tcrbert_tra_lora, self.tcrbert_tra_tokenizer = setup_peft_tcrbert(peft_config_tcrbert, no_lora=ln_cfg.no_lora, regular_ft=ln_cfg.regular_ft)   
        self.tcrbert_trb_lora, self.tcrbert_trb_tokenizer = setup_peft_tcrbert(peft_config_tcrbert, no_lora=ln_cfg.no_lora, regular_ft=ln_cfg.regular_ft)

        self.ln_config = ln_cfg

        if hidden_dim:
            print("Using multi-layer projection head")
            self.proj_head = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.LeakyReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(hidden_dim, projection_dim),
                            )
            # Initialize the projection head weights
            nn.init.kaiming_uniform_(self.proj_head[0].weight)
            nn.init.kaiming_uniform_(self.proj_head[-1].weight)
        else:
            print("Using single-layer projection head")
            self.proj_head = nn.Sequential(
                                nn.Linear(input_dim, projection_dim),
                                nn.LayerNorm(projection_dim),
                            )
            # Initialize the projection head weights
            nn.init.kaiming_uniform_(self.proj_head[0].weight)

        if self.ln_config.swe_pooling:
            self.swe_pooling_a = SWE_Pooling(d_in=input_dim // 2, num_ref_points=256, num_slices=projection_dim // 2)
            self.swe_pooling_b = SWE_Pooling(d_in=input_dim // 2, num_ref_points=256, num_slices=projection_dim // 2)

            self.proj_head = nn.Sequential(
                                nn.Linear(projection_dim, projection_dim // 2),
                                nn.LayerNorm(projection_dim // 2),
                            )

        self.device = device

    def forward(self, x, mask):
        tra_tokens, trb_tokens = self.process_seqs(x, mask, mask_prob=self.ln_config.mask_prob)

        # feed to TCRBERT
        rescoding_tra = self.tcrbert_tra_lora(**tra_tokens)
        rescoding_trb = self.tcrbert_trb_lora(**trb_tokens)

        if self.ln_config.swe_pooling:
            # for SWE pooling:
            attn_mask_a = get_attention_mask(tra_tokens)
            if isinstance(rescoding_tra, dict):
                rescoding_tra = rescoding_tra['last_hidden_state']
            else:
                rescoding_tra = rescoding_tra.last_hidden_state
            tra_outputs = self.swe_pooling_a(rescoding_tra, attn_mask_a)

            attn_mask_b = get_attention_mask(trb_tokens)
            if isinstance(rescoding_trb, dict):
                rescoding_trb = rescoding_trb['last_hidden_state']
            else:
                rescoding_trb = rescoding_trb.last_hidden_state
            trb_outputs = self.swe_pooling_b(rescoding_trb, attn_mask_b)
        else:
            # for regular mean pooling
            tra_outputs = get_sequence_embeddings(tra_tokens, rescoding_tra)
            trb_outputs = get_sequence_embeddings(trb_tokens, rescoding_trb)

        tcr_embeds = torch.cat((tra_outputs, trb_outputs), dim=-1)

        return self.proj_head(tcr_embeds)


    def process_seqs(self, seqs, mask, mask_prob=0.15):
        tra_seqs_, trb_seqs_ = seqs

        # insert spaces between residues for correct formatting:
        tra_seqs = [insert_spaces(seq) for seq in tra_seqs_]
        trb_seqs = [insert_spaces(seq) for seq in trb_seqs_]

        tra_tokens = self.tcrbert_tra_tokenizer(tra_seqs, return_tensors="pt", padding=True)
        trb_tokens = self.tcrbert_trb_tokenizer(trb_seqs, return_tensors="pt", padding=True)

        if mask:
            # masking the sequences for training
            tra_seqs_, attn_mask_indices = apply_masking_seq(tra_seqs_, p=mask_prob)
            indices_tensor = torch.tensor(attn_mask_indices, dtype=torch.long)
            if len(indices_tensor) > 0:
                tra_tokens['attention_mask'][indices_tensor[:, 0], indices_tensor[:, 1] + 1] = 0. # +1 to account for the CLS token

            trb_seqs_, attn_mask_indices = apply_masking_seq(trb_seqs_, p=mask_prob)
            indices_tensor = torch.tensor(attn_mask_indices, dtype=torch.long)
            if len(indices_tensor) > 0:
                trb_tokens['attention_mask'][indices_tensor[:, 0], indices_tensor[:, 1] + 1] = 0. # +1 to account for the CLS token

        # print("TCR Seqs:", tra_seqs_)
        # print("seqs_tokens:", tra_tokens['attention_mask'])

        return tra_tokens.to(self.device), trb_tokens.to(self.device)


class TCREncoderTCRLang(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, hidden_dim=1024, device='cpu'):
        super().__init__()
        from .lora import setup_peft_ablang2
        from .configs import peft_config_ablang2

        self.ablang2_lora, self.ablang2_tokenizer = setup_peft_ablang2(peft_config_ablang2, receptor_type='TCR', device=device, no_lora=ln_cfg.no_lora)
        self.padding_idx = 21
        self.mask_token = 23
        self.sep_token_id = 25

        self.ln_config = ln_cfg

        if hidden_dim:
            print("Using multi-layer projection head")
            self.proj_head = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.LeakyReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(hidden_dim, projection_dim),
                            )
        else:
            print("Using single-layer projection head")
            self.proj_head = nn.Sequential(
                                nn.Linear(input_dim, projection_dim),
                                nn.LayerNorm(projection_dim),
                            )
        
        if self.ln_config.swe_pooling:
            self.swe_pooling = SWE_Pooling(d_in=input_dim, num_ref_points=512, num_slices=projection_dim)

            self.proj_head = nn.Sequential(
                                nn.Linear(projection_dim, projection_dim // 2),
                                nn.LayerNorm(projection_dim // 2),
                            )

        self.device = device

    def forward(self, x, mask):
        seq_tokens = self.process_seqs(x, mask, mask_prob=self.ln_config.mask_prob)

        # print("seq tokens:", seq_tokens)

        # feed to TCRLang
        rescoding = self.ablang2_lora(seq_tokens)

        # process TCRLang outputs
        seq_inputs = {'attention_mask': ~((seq_tokens == self.padding_idx) | (seq_tokens == self.mask_token))}
        model_output = {'last_hidden_state': rescoding.last_hidden_states}

        if self.ln_config.swe_pooling:
            # for SWE pooling:
            attn_mask = seq_inputs['attention_mask']
            model_embed = model_output['last_hidden_state']

            seq_outputs = self.swe_pooling(model_embed, attn_mask)

        else:
            # for regular mean pooling
            seq_outputs = get_sequence_embeddings(seq_inputs, model_output, is_sep=False, is_cls=False)

        return self.proj_head(seq_outputs)
    
    def process_seqs(self, seqs, mask, mask_prob=0.15):
        H_seqs, L_seqs = seqs

        # format the seq strings accordingly to TCRLang (B chain comes first, so we swap H and L orders):
        ab_seqs = [f"{L_seqs[i]}|{H_seqs[i]}" for i in range(len(H_seqs))]

        seqs_tokens = self.ablang2_tokenizer(ab_seqs, pad=True, w_extra_tkns=False, device=self.device)

        if mask:
            # masking the sequences for training
            ab_seqs, attn_mask_indices = apply_masking_seq(ab_seqs, mask_token='*', p=mask_prob)
            indices_tensor = torch.tensor(attn_mask_indices, dtype=torch.long)
            if len(indices_tensor) > 0:
                seqs_tokens[indices_tensor[:, 0], indices_tensor[:, 1]] = self.mask_token

            # leave the SEP tokens ('|') unmasked!!
            for i, l_seq in enumerate(L_seqs):
                seqs_tokens[i, len(l_seq)] = self.sep_token_id

        return seqs_tokens

class TCREncoderESM(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, model_config=None, hidden_dim=1024, device='cpu'):
        super().__init__()
        from .lora import setup_peft_esm2
        from .configs import peft_config_esm2

        self.ln_config = ln_cfg
        self.model_config = model_config
        self.projection_dim = projection_dim

        if self.model_config.receptor_model_name == 'esm3':
            from .lora import setup_peft_esm3
            from .configs import peft_config_esm3

            # load the LoRA adapted ESM-3 Model here:
            self.esm_lora, self.esm_tokenizer = setup_peft_esm3(peft_config_esm3, ln_cfg.no_lora)
        else:
            from .lora import setup_peft_esm2
            from .configs import peft_config_esm2

            # load the LoRA adapted ESM-2 Model here:
            self.esm_lora, self.esm_tokenizer = setup_peft_esm2(peft_config_esm2, ln_cfg.no_lora, ln_cfg.regular_ft)

        if self.projection_dim:
            if hidden_dim:
                print("Using multi-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(hidden_dim, projection_dim),
                                )
            else:
                print("Using single-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, projection_dim),
                                    nn.LayerNorm(projection_dim),
                                )
        else:
            print("NOT using projection head")

        # for ESM-2, we need linker to represent multimers
        self.linker_size = 25
        self.gly_linker = 'G'*self.linker_size
        self.gly_idx = 6 # according to: https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/main/vocab.txt

        self.device = device
            
    def forward(self, x, mask):
        seqs_tokens = self.process_seqs(x, mask, mask_prob=self.ln_config.mask_prob)

        if self.model_config.receptor_model_name == 'esm3':
            outputs = self.esm_lora(sequence_tokens=seqs_tokens['input_ids']).embeddings
            outputs = {'last_hidden_state': outputs}
        else:
            outputs = self.esm_lora(**seqs_tokens)

        seq_embeds = get_sequence_embeddings(seqs_tokens, outputs, is_sep=False, is_cls=False)

        if self.projection_dim:
            return self.proj_head(seq_embeds)
        else:
            return seq_embeds
    
    def process_seqs(self, seqs, mask, mask_prob=0.15):
        '''
        seqs: list of epitope sequences
        '''
        tra_seqs, trb_seqs = seqs
        seqs = [f"{tra_seqs[i]}{self.gly_linker}{trb_seqs[i]}" for i in range(len(tra_seqs))]
        mask_regions = [[True]*len(seqa)+[False]*self.linker_size+[True]*len(seqb) for seqa, seqb in zip(tra_seqs, trb_seqs)]

        # marking where the Glycine linker starts
        attn_starts = [len(alpha_chain) for alpha_chain in tra_seqs]

        # removing special tokens since epitopes are protein fragments (peptides)
        seqs_tokens = self.esm_tokenizer(seqs, return_tensors="pt", add_special_tokens=False, padding=True)

        if mask:
            # masking the sequences for training
            seqs, attn_mask_indices = apply_masking_seq(seqs, mask_token='<mask>', mask_regions=mask_regions, p=mask_prob)
            indices_tensor = torch.tensor(attn_mask_indices, dtype=torch.long)
            if len(indices_tensor) > 0:
                seqs_tokens['attention_mask'][indices_tensor[:, 0], indices_tensor[:, 1]] = 0.

            # remove mask tokens on linker region:
            for i in range(len(attn_starts)):
                seqs_tokens['input_ids'][i, attn_starts[i]:attn_starts[i]+self.linker_size] = self.gly_idx


        # attention masking the linker region
        for i in range(len(attn_starts)):
            seqs_tokens['attention_mask'][i, attn_starts[i]:attn_starts[i]+self.linker_size] = 0.
        
        # print("TCR Seqs:", seqs)
        # print("seqs_tokens:", seqs_tokens['attention_mask'])

        return seqs_tokens.to(self.device)


class TCREncoderESMBetaOnly(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, model_config=None, hidden_dim=1024, device='cpu'):
        super().__init__()
        from .lora import setup_peft_esm2
        from .configs import peft_config_esm2

        # # load the LoRA adapted ESM Model here:
        # self.esm_lora, self.esm_tokenizer = setup_peft_esm2(peft_config_esm2, ln_cfg.no_lora)

        self.ln_config = ln_cfg
        self.model_config = model_config
        self.projection_dim = projection_dim

        if self.model_config.receptor_model_name == 'esm3':
            from .lora import setup_peft_esm3
            from .configs import peft_config_esm3

            # load the LoRA adapted ESM-3 Model here:
            self.esm_lora, self.esm_tokenizer = setup_peft_esm3(peft_config_esm3, ln_cfg.no_lora)
        else:
            from .lora import setup_peft_esm2
            from .configs import peft_config_esm2

            # load the LoRA adapted ESM-2 Model here:
            self.esm_lora, self.esm_tokenizer = setup_peft_esm2(peft_config_esm2, ln_cfg.no_lora)

        if self.projection_dim:
            if hidden_dim:
                print("Using multi-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(hidden_dim, projection_dim),
                                )
            else:
                print("Using single-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, projection_dim),
                                    nn.LayerNorm(projection_dim),
                                )
        else:
            print("NOT using projection head")

        self.device = device
            
    def forward(self, x, mask):
        seqs_tokens = self.process_seqs(x, mask, mask_prob=self.ln_config.mask_prob)

        if self.model_config.receptor_model_name == 'esm3':
            outputs = self.esm_lora(sequence_tokens=seqs_tokens['input_ids']).embeddings
            outputs = {'last_hidden_state': outputs}
        else:
            outputs = self.esm_lora(**seqs_tokens)

        seq_embeds = get_sequence_embeddings(seqs_tokens, outputs, is_sep=False, is_cls=False)

        if self.projection_dim:
            return self.proj_head(seq_embeds)
        else:
            return seq_embeds
    
    def process_seqs(self, seqs, mask, mask_prob=0.15):
        '''
        seqs: list of epitope sequences
        '''
        tra_seqs, trb_seqs = seqs
        seqs = trb_seqs

        # removing special tokens since epitopes are protein fragments (peptides)
        seqs_tokens = self.esm_tokenizer(seqs, return_tensors="pt", add_special_tokens=False, padding=True)

        if mask:
            # masking the sequences for training
            seqs, attn_mask_indices = apply_masking_seq(seqs, mask_token='<mask>', p=mask_prob)
            indices_tensor = torch.tensor(attn_mask_indices, dtype=torch.long)
            if len(indices_tensor) > 0:
                seqs_tokens['attention_mask'][indices_tensor[:, 0], indices_tensor[:, 1]] = 0.

        return seqs_tokens.to(self.device)


class TCREncoderInHouse(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, model_config=None, hidden_dim=1024, device='cpu'):
        super().__init__()
        from .lora import setup_peft_inhouse
        from .configs import peft_config_inhouse
        import os

        model_ckpt_path = os.getenv('INHOUSE_MODEL_CKPT_PATH')

        self.inhouse_lora, self.inhouse_tokenizer = setup_peft_inhouse(peft_config_inhouse, ln_cfg.no_lora, model_ckpt_path=model_ckpt_path)

        self.ln_config = ln_cfg
        self.model_config = model_config
        self.projection_dim = projection_dim

        if self.projection_dim:
            if hidden_dim:
                print("Using multi-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(hidden_dim, projection_dim),
                                )
            else:
                print("Using single-layer projection head")
                self.proj_head = nn.Sequential(
                                    nn.Linear(input_dim, projection_dim),
                                    nn.LayerNorm(projection_dim),
                                )
        else:
            print("NOT using projection head")

        self.device = device
    
    def forward(self, x, mask):
        seq_tokens = self.process_seqs(x, mask, mask_prob=self.ln_config.mask_prob)

        # feed to InHouse Model
        # print("seq tokens input_ids:", seq_tokens['input_ids'])
        # print("seq tokens attention_mask:", seq_tokens['attention_mask'])
        seq_outputs, _ = self.inhouse_lora(seq_tokens["input_ids"], seq_tokens["attention_mask"])

        # print("seq outputs:", seq_outputs)

        return self.proj_head(seq_outputs)
    
    def process_seqs(self, seqs, mask, mask_prob=0.15):
        tra_seqs, trb_seqs = seqs

        if mask:
            tra_seqs_, tra_masks = apply_masking_seq(tra_seqs, mask_token='<mask>', p=mask_prob)
            trb_seqs_, trb_masks = apply_masking_seq(trb_seqs, mask_token='<mask>', p=mask_prob)

            # adjust the tra_masks and trb_masks to the correct indices:
            tra_masks = [(n, 1+i) for (n, i) in tra_masks]
            trb_masks = [(n, 1+len(tra_seqs[n])+2+i) for (n, i) in trb_masks]

            indices_tensor = torch.tensor(tra_masks + trb_masks, dtype=torch.long)

            tra_seqs, trb_seqs = tra_seqs_, trb_seqs_

        # format the seq strings accordingly to InHouse:
        ab_seqs = [self.apply_special_token_formatting(tra_seqs[i], trb_seqs[i]) for i in range(len(tra_seqs))]

        seqs_tokens = self.inhouse_tokenizer(ab_seqs, return_tensors="pt", add_special_tokens=False, padding=True)

        # adjust the attention mask
        if mask and len(indices_tensor) > 0:
            seqs_tokens['attention_mask'][indices_tensor[:, 0], indices_tensor[:, 1]] = 0.

        return seqs_tokens.to(self.device)
    
    def apply_special_token_formatting(self, alpha, beta):
        '''
            Apply RoBERTa style formatting to input:
            <cls>seq1<sep><sep>seq2<sep>
        '''
        return f"{self.inhouse_tokenizer.cls_token}{alpha}{self.inhouse_tokenizer.eos_token}{self.inhouse_tokenizer.eos_token}{beta}{self.inhouse_tokenizer.eos_token}"

class TCREncoderOneHot(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, model_config, device='cpu'):
        super().__init__()

        self.ln_config = ln_cfg
        self.projection_dim = projection_dim

        if self.projection_dim:
            print("Using single-layer projection head")
            self.proj_head = nn.Sequential(
                                nn.Linear(input_dim, projection_dim),
                                nn.LayerNorm(projection_dim),
                            )
        else:
            assert False, "Projection head must be used with one-hot encoding!"
        
        # Define the amino acid to index mapping
        self.amino_acid_to_index = {
                'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
                'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
                'X': 20  # Unknown amino acid
            }

        self.device = device

    def forward(self, x, mask):
        seqs = [seqa + seqb for seqa, seqb in zip(x[0], x[1])]
        seqs_onehot = self.create_padded_one_hot_tensor(seqs, len(self.amino_acid_to_index))

        proj_output = self.proj_head(seqs_onehot)

        # average the projected embeddings by seq length:
        seq_lens = torch.sum(seqs_onehot, dim=(1, 2))
        # Create a mask with shape (batch_size, max_seq_length)
        seq_mask = torch.arange(proj_output.size(1)).unsqueeze(0).to(self.device) < seq_lens.unsqueeze(-1)
        seq_mask = seq_mask.unsqueeze(2)  # Shape (batch_size, max_seq_length, 1)
        # Sum the embeddings across the sequence length dimension using the mask
        masked_embeddings = proj_output * seq_mask
        sum_embeddings = masked_embeddings.sum(dim=1)

        # Divide by the true sequence lengths to get the average
        avg_embeddings = sum_embeddings / seq_lens.unsqueeze(1)#.to(embeddings.device)
        
        return avg_embeddings


    # @staticmethod
    def encode_amino_acid_sequence(self, sequence):
        """ Convert an amino acid sequence to a list of indices. """
        return [self.amino_acid_to_index[aa] for aa in sequence]

    # @staticmethod
    def one_hot_encode_sequence(self, sequence, vocab_size):
        """ One-hot encode a single sequence. """
        encoding = np.zeros((len(sequence), vocab_size), dtype=int)
        for idx, char in enumerate(sequence):
            encoding[idx, char] = 1
        return encoding

    # @staticmethod
    def pad_sequences(self, encoded_sequences, max_length):
        """ Pad the encoded sequences to the maximum length. """
        padded_sequences = []
        for seq in encoded_sequences:
            padded_seq = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=0)
            padded_sequences.append(padded_seq)
        return np.array(padded_sequences)

    # @staticmethod
    def create_padded_one_hot_tensor(self, sequences, vocab_size):
        """ Convert a batch of sequences to a padded one-hot encoding tensor. """
        # Encode and one-hot encode each sequence
        encoded_sequences = [self.one_hot_encode_sequence(self.encode_amino_acid_sequence(seq), vocab_size) for seq in sequences]
        
        # Determine the maximum sequence length
        max_length = max(len(seq) for seq in sequences)
        
        # Pad the sequences
        padded_sequences = self.pad_sequences(encoded_sequences, max_length)
        
        # Convert to a PyTorch tensor
        padded_tensor = torch.tensor(padded_sequences, dtype=torch.float32)
        
        return padded_tensor.to(self.device)