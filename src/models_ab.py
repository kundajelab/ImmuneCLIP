import torch
import numpy as np
import torch.nn as nn
import re

from .utils import get_sequence_embeddings, insert_spaces, get_attention_mask, apply_masking_seq
from .swe_pooling import SWE_Pooling

class AntibodyEncoderAbLang(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, device='cpu'):
        super().__init__()
        from .lora import setup_peft_ablang
        from .configs import peft_config_ablang

        # load the LoRA adapted AbLang HL Models here:
        self.ablang_H_lora, self.ablang_H_tokenizer = setup_peft_ablang(peft_config_ablang, chain='H')
        self.ablang_L_lora, self.ablang_L_tokenizer = setup_peft_ablang(peft_config_ablang, chain='L')

        self.proj_head = nn.Sequential(
                            nn.Linear(input_dim, projection_dim),
                            nn.LayerNorm(projection_dim),
                         )
        
        self.device = device
    
    def forward(self, x):
        H_seqs, L_seqs = x
        H_seqs_tokens = self.process_seqs(H_seqs, chain='H')
        L_seqs_tokens = self.process_seqs(L_seqs, chain='L')

        try:
            H_outputs = self.ablang_H_lora(**H_seqs_tokens)
        except:
            print("Error in feeding H sequences")

            print("H seq:", H_seqs)
            print("H seq tokens max:", torch.max(H_seqs_tokens['input_ids']))
            print("self.ablang_H_lora: ", self.ablang_H_lora)

            raise ValueError
        
        try:
            L_outputs = self.ablang_L_lora(**L_seqs_tokens)
        except:
            print("Error in feeding L sequences")

            print("L seq:", L_seqs)
            print("L seq tokens max:", torch.max(L_seqs_tokens['input_ids']))
            print("self.ablang_L_lora: ", self.ablang_L_lora)


        H_outputs = get_sequence_embeddings(H_seqs_tokens, H_outputs)
        L_outputs = get_sequence_embeddings(L_seqs_tokens, L_outputs)

        Ab_seq_embeds = torch.cat((H_outputs, L_outputs), dim=-1)

        return self.proj_head(Ab_seq_embeds)
    
    def process_seqs(self, seqs, chain):
        '''
        seqs: tuple of sequences
        '''

        # format the seq strings accordingly to AbLang:
        seqs = [insert_spaces(seq) for seq in seqs]
        # seqs = [' '.join(seq) for seq in seqs]

        if chain == 'H':
            seqs_tokens = self.ablang_H_tokenizer(seqs, return_tensors="pt", padding=True)
        else:
            seqs_tokens = self.ablang_L_tokenizer(seqs, return_tensors="pt", padding=True)

        return seqs_tokens.to(self.device)


class AntibodyEncoderAbLang2(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, device='cpu'):
        super().__init__()
        from .lora import setup_peft_ablang2
        from .configs import peft_config_ablang2

        self.ablang2_lora, self.ablang2_tokenizer = setup_peft_ablang2(peft_config_ablang2, receptor_type='BCR', device=device, no_lora=ln_cfg.no_lora)
        self.padding_idx = 21

        self.proj_head = nn.Sequential(
                            nn.Linear(input_dim, projection_dim),
                            nn.LayerNorm(projection_dim),
                         )

        self.device = device
    
    def forward(self, x):
        seq_tokens = self.process_seqs(x)

        # print("seq tokens:", seq_tokens)

        # feed to AbLang2
        rescoding = self.ablang2_lora(seq_tokens)

        # process AbLang2 outputs
        seq_inputs = {'attention_mask': ~(seq_tokens == self.padding_idx)}
        model_output = {'last_hidden_state': rescoding.last_hidden_states}

        seq_outputs = get_sequence_embeddings(seq_inputs, model_output, is_sep=False, is_cls=False)

        return self.proj_head(seq_outputs)
    
    def process_seqs(self, seqs):
        H_seqs, L_seqs = seqs

        # format the seq strings accordingly to AbLang2:
        ab_seqs = [f"{H_seqs[i]}|{L_seqs[i]}" for i in range(len(H_seqs))]

        seqs_tokens = self.ablang2_tokenizer(ab_seqs, pad=True, w_extra_tkns=False, device=self.device)

        return seqs_tokens


class AntibodyEncoderAntiberta2(nn.Module):
    def __init__(self, input_dim, projection_dim, ln_cfg, device='cpu'):
        super().__init__()
        from .lora import setup_peft_aberta2
        from .configs import peft_config_aberta2

        self.aberta2_lora, self.aberta2_tokenizer = setup_peft_aberta2(peft_config_aberta2)

        self.proj_head = nn.Sequential(
                            nn.Linear(input_dim, projection_dim),
                            nn.LayerNorm(projection_dim),
                         )

        self.device = device
    
    def forward(self, x):
        seq_tokens = self.process_seqs(x)

        try:
            # feed to AntiBERTa
            rescoding = self.aberta2_lora(**seq_tokens)
        except:
            print("seqs:", x)
            print("seq tokens:", seq_tokens)
            raise ValueError

        seq_embeds = get_sequence_embeddings(seq_tokens, rescoding)

        return self.proj_head(seq_embeds)
    
    def process_seqs(self, seqs):
        H_seqs, L_seqs = seqs

        # format the seq strings accordingly to Antiberta2:
        ab_seqs = [f"{insert_spaces(H_seqs[i])} [SEP] {insert_spaces(L_seqs[i])}" for i in range(len(H_seqs))]

        seqs_tokens = self.aberta2_tokenizer(ab_seqs, return_tensors="pt", padding=True)

        return seqs_tokens.to(self.device)
