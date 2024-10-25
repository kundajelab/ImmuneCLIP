import torch
import numpy as np
import pandas as pd
import re
import json
import functools
import os

# adapted from the HuggingFace repo for AbLang
def get_sequence_embeddings(encoded_input, model_output, is_sep=True, is_cls=True, epitope_mask=None):
    if isinstance(model_output, dict):
        output_last_h_state = model_output['last_hidden_state']
    else:
        output_last_h_state = model_output.last_hidden_state
    
    mask = encoded_input['attention_mask'].float()
    if is_sep:
        d = {k: v for k, v in torch.nonzero(mask).cpu().numpy()} # dict of sep tokens
        # make sep token invisible
        for i in d:
            mask[i, d[i]] = 0
    if is_cls:
        mask[:, 0] = 0.0 # make cls token invisible
    if epitope_mask is not None:
        mask = mask * epitope_mask # make non-epitope regions invisible
    mask = mask.unsqueeze(-1).expand(output_last_h_state.size())
    sum_embeddings = torch.sum(output_last_h_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_attention_mask(encoded_input, is_sep=True, is_cls=True):
    mask = encoded_input['attention_mask'].float()
    if is_sep:
        d = {k: v for k, v in torch.nonzero(mask).cpu().numpy()} # dict of sep tokens
        # make sep token invisible
        for i in d:
            mask[i, d[i]] = 0
    if is_cls:
        mask[:, 0] = 0.0 # make cls token invisible
    return mask

# load CATCR dataset (contains epitope/CDR3-B data from VDJdb, IEDB, McPAS-TCR):
def load_catcr_data(tsv_path):
    df_catcr = pd.read_csv(tsv_path, delimiter=',')

    # rename the TCR column to light_chain:
    df_catcr = df_catcr.rename(columns={'CDR3_B':'light_chain', 'EPITOPE':'epitope'})
    
    # no alpha chain, so create an empty heavy chain column:
    df_catcr['heavy_chain'] = ""

    # reset index
    df_catcr = df_catcr.reset_index(drop=True)

    return df_catcr

# load CATCR dataset (contains epitope/CDR3-B data from VDJdb, IEDB, McPAS-TCR):
def load_catcr_data_presplit(tsv_path):
    df_catcr_train = pd.read_csv(os.path.join(tsv_path, "train.csv"), delimiter=',')
    df_catcr_test = pd.read_csv(os.path.join(tsv_path, "test.csv"), delimiter=',')

    # rename the TCR column to light_chain:
    df_catcr_train = df_catcr_train.rename(columns={'CDR3_B':'light_chain',
                                                    'EPITOPE':'epitope'})
    df_catcr_test = df_catcr_test.rename(columns={'CDR3_B':'light_chain',
                                                  'EPITOPE':'epitope'})
    
    # no alpha chain, so create an empty heavy chain column:
    df_catcr_train['heavy_chain'] = ""
    df_catcr_test['heavy_chain'] = ""

    # reset index
    df_catcr_train = df_catcr_train.reset_index(drop=True)
    df_catcr_test = df_catcr_test.reset_index(drop=True)

    return df_catcr_train, df_catcr_test

# load MixTCRPred dataset (contains data from VDJdb, IEDB, McPAS, and 10x Genomics)
def load_mixtcrpred_data(tsv_path):
    df_mix = pd.read_csv(tsv_path, delimiter=',')

    # drop rows where Epitope name is not a peptide sequence:
    df_mix = df_mix[df_mix['epitope'].str.isupper()]
    df_mix = df_mix[df_mix['epitope'].apply(is_alpha_only)]

    # drop rows whose TCR sequences are missing:
    df_mix = df_mix.dropna(subset=['cdr3_TRA', 'cdr3_TRB'])

    # rename the TCR columns to heavy_chain and light_chain:
    df_mix = df_mix.rename(columns={'cdr3_TRA':'heavy_chain', 'cdr3_TRB':'light_chain'})

    # drop duplicates of epitope-TRA-TRB triplets:
    df_mix = df_mix.drop_duplicates(subset=['epitope', 'heavy_chain', 'light_chain']) # drops ~2700 entries

    # only restrict it to human data:
    df_mix = df_mix.loc[df_mix["species"] == "HomoSapiens"]

    # reset index
    df_mix = df_mix.reset_index(drop=True)

    return df_mix

def load_mixtcrpred_data_pmhc(tsv_path, mhc_map_path):
    df_mhc_map = pd.read_csv(mhc_map_path, delimiter='\t', index_col=0) # format where index is two-part name like "A*01:01" and the column label is "max_sequence" containing the sequence with signal peptide removed
    # raise NotImplementedError("mhc_map needs to be updated to include all the mouse MHCs if we are including mouse data")

    df_mix = load_mixtcrpred_data(tsv_path)

    def clean_mhc_allele(allele):
        ''' mixtcrpred will sometimes have MHC names formatted like "HLA-DRB1:01".
            In these cases, the colon must be replaced with an asterisk'''
        if allele != "B2M" and "-" not in allele: # from splitting HLA-__A/__B entries, some HLAs have lost their prefix. Add it back. H2-* alleles are fine
            allele = "HLA-" + allele 

        pattern = re.compile(r'(HLA-[A-Za-z0-9]+):([:0-9]+)')
        return pattern.sub(r'\1*\2', allele)
    
    # Create new columns (will change their values below)
    df_mix["mhc.a"] = df_mix["MHC"]
    df_mix["mhc.b"] = df_mix["MHC"]

    # make sure mouse names are consistent with map_mhc_allele's canonical alleles
    for i, row in df_mix.iterrows():
        if row.MHC_class == "MHCI":
            df_mix.at[i, "mhc.a"] = row.MHC
            df_mix.at[i, "mhc.b"] = "B2M"
        else: # MHCII
            # Cases that need to be handled
            # Counter({'H2-IAb': 3674,
            #  'HLA-DPB1*04:01': 388,
            #  'HLA-DRB1:01': 71,
            #  'HLA-DQA1:02/DQB1*06:02': 46,
            #  'HLA-DRB1*04:05': 31,
            #  'H2-IEk': 25,
            #  'H2-Kb': 24,
            #  'HLA-DRA:01/DRB1:01': 21,
            #  'HLA-DRB1*07:01': 20,
            #  'HLA-DRB1*15:01': 15,
            #  'HLA-DQA1*05:01/DQB1*02:01': 14,
            #  'HLA-DRB1*04:01': 14,
            #  'HLA-DQA': 13,
            #  'H-2q': 12,
            #  'HLA-DRB1*11:01': 10,
            #  'HLA-DQ2': 10,
            #  'HLA-DRA:01': 10})
            if "/" in row.MHC:
                df_mix.at[i, "mhc.a"] = row.MHC.split("/")[0]
                df_mix.at[i, "mhc.b"] = row.MHC.split("/")[1]
            else:
                # inconsistencies in mixtcrpred: there are 24 examples of 'H2-Kb' and 12 examples of 'H-2q'
                # which are labeled as MHC II even though they are MHC I alleles. Switch them to MHC I
                if row.MHC == "H2-Kb":
                    df_mix.at[i, "mhc.a"] = "H2-Kb"
                    df_mix.at[i, "mhc.b"] = "B2M"
                elif row.MHC == "H-2q":
                    # also switch formatting to H2- nomenclature
                    df_mix.at[i, "mhc.a"] = "H2-Q"
                    df_mix.at[i, "mhc.b"] = "B2M"
                else:
                    if row.MHC == "H2-IAb": # switch this nomenclature to H2-A
                        df_mix.at[i, "mhc.a"] = "H2-AA"
                        df_mix.at[i, "mhc.b"] = "H2-AB"
                    elif row.MHC == "H2-IEk": # add A and B chains for this allele
                        df_mix.at[i, "mhc.a"] = "H2-IEkA"
                        df_mix.at[i, "mhc.b"] = "H2-IEkB"
                    elif row.MHC == "HLA-DQ2":
                        df_mix.at[i, "mhc.a"] = "HLA-DQA1"
                        df_mix.at[i, "mhc.b"] = "HLA-DQB1"
                    elif row.MHC == "HLA-DQA":
                        df_mix.at[i, "mhc.a"] = "HLA-DQA"
                        df_mix.at[i, "mhc.b"] = "HLA-DQB"
                    elif row.MHC == "HLA-DRA:01":
                        df_mix.at[i, "mhc.a"] = "HLA-DRA:01"
                        df_mix.at[i, "mhc.b"] = "HLA-DRB"
                    else:
                        # remainder are all 'HLA-D*B[:*]...' alleles
                        # extract the allele name
                        pattern = re.compile(r'HLA-([A-Za-z]+)[0-9]*[:*].*')
                        allele_name = pattern.search(row.MHC).group(1) # e.g. extracts DRB from HLA-DRB1:01 or HLA-DRB1*11:01
                        mhca_name = allele_name.replace("B", "A")
                        df_mix.at[i, "mhc.a"] = f"HLA-{mhca_name}"
                        df_mix.at[i, "mhc.b"] = row.MHC


    df_mix["mhc.a"] = df_mix["mhc.a"].apply(clean_mhc_allele)
    df_mix["mhc.b"] = df_mix["mhc.b"].apply(clean_mhc_allele)

    df_mix["mhc.a_seq"] = df_mix["mhc.a"].apply(functools.partial(map_mhc_allele, df_mhc_map=df_mhc_map))
    df_mix["mhc.b_seq"] = df_mix["mhc.b"].apply(functools.partial(map_mhc_allele, df_mhc_map=df_mhc_map))



    return df_mix


# load the IEDB dataset:
def load_iedb_data(tsv_path, replace_X=False, remove_X=False, use_anarci=False):
    df_iedb = pd.read_csv(tsv_path, delimiter='\t')

    # drop rows where Epitope name is not a peptide sequence:
    df_iedb = df_iedb[df_iedb['Epitope - Name'].str.isupper()]
    df_iedb = df_iedb[df_iedb['Epitope - Name'].apply(is_alpha_only)]

    # drop rows whose Ab HL sequences missing:
    df_iedb = df_iedb.dropna(subset=['Chain 1 - Protein Sequence', 'Chain 2 - Protein Sequence'])

    # drop rows whose CDRs are missing:
    cdr_columns = ['Chain 1 - CDR3 Calculated', 'Chain 1 - CDR2 Calculated', 'Chain 1 - CDR1 Calculated',
                   'Chain 2 - CDR3 Calculated', 'Chain 2 - CDR2 Calculated', 'Chain 2 - CDR1 Calculated']
    df_iedb = df_iedb.dropna(subset=cdr_columns)

    if use_anarci:
        from anarci import run_anarci
        # run ANARCI to get the Fv region of the sequences:
        print("running anarci on sequences...")
        for col_id in ['Chain 1 - Protein Sequence', 'Chain 2 - Protein Sequence']:
            seqs = df_iedb[col_id].str.upper()
            seqs_ = [(str(i), s) for i, s in enumerate(seqs)]
            anarci_results = run_anarci(seqs_)
            start_end_pairs = [(anarci_results[2][i][0]['query_start'], anarci_results[2][i][0]['query_end']) for i in range(len(seqs_))]
            seqs = [seq[a:b] for seq, (a,b) in zip(seqs, start_end_pairs)]
            df_iedb[col_id] = seqs
        
        df_iedb = df_iedb.reset_index(drop=True)
        # FOR FUTURE USERS: IF PERFORMING BCR CALCULATIONS, SAVE df_iedb TO A CSV FILE
        # e.g. df_iedb.to_csv("path/to/iedb_data_with_anarci.csv", index=False)
        print("done running anarci!")

    # change column names:
    df_iedb = df_iedb.rename(columns={'Epitope - Name': 'epitope',
                                      'Chain 1 - Protein Sequence': 'heavy_chain',
                                      'Chain 2 - Protein Sequence': 'light_chain',
                                      'Chain 1 - CDR3 Calculated': 'heavy_chain_cdr3',
                                      'Chain 1 - CDR2 Calculated': 'heavy_chain_cdr2',
                                      'Chain 1 - CDR1 Calculated': 'heavy_chain_cdr1',
                                      'Chain 2 - CDR3 Calculated': 'light_chain_cdr3',
                                      'Chain 2 - CDR2 Calculated': 'light_chain_cdr2',
                                      'Chain 2 - CDR1 Calculated': 'light_chain_cdr1'})
    
    if replace_X:
        # replace X's with [MASK] in the sequences for AbLang:
        df_iedb['heavy_chain'] = df_iedb['heavy_chain'].str.replace('X', '[MASK]')
        df_iedb['light_chain'] = df_iedb['light_chain'].str.replace('X', '[MASK]')
    
    if remove_X:
        # remove rows with 'X' in the sequences:
        df_iedb = df_iedb[~df_iedb['heavy_chain'].str.contains('X')]
        df_iedb = df_iedb[~df_iedb['light_chain'].str.contains('X')]

    # remove all examples where the heavy_chain and light_chain values are the same (likely means invalid pair):
    df_iedb = df_iedb[df_iedb['heavy_chain'] != df_iedb['light_chain']]

    # reset index
    df_iedb = df_iedb.reset_index(drop=True)
    
    return df_iedb

def load_iedb_data_cdr3(tsv_path, replace_hashtag=False):
    df_iedb = pd.read_csv(tsv_path, delimiter='\t')

    # drop rows where Epitope name is not a peptide sequence:
    df_iedb = df_iedb[df_iedb['Epitope - Name'].str.isupper()]
    df_iedb = df_iedb[df_iedb['Epitope - Name'].apply(is_alpha_only)]

    # drop rows whose Ab HL sequences missing:
    df_iedb = df_iedb.dropna(subset=['Chain 1 - CDR3 Curated', 'Chain 2 - CDR3 Curated'])

    # change column names:
    df_iedb = df_iedb.rename(columns={'Epitope - Name': 'epitope',
                                      'Chain 1 - CDR3 Curated': 'heavy_chain',
                                      'Chain 2 - CDR3 Curated': 'light_chain',})
    
    # remove all examples where the heavy_chain and light_chain values are the same (likely means invalid pair):
    df_iedb = df_iedb[df_iedb['heavy_chain'] != df_iedb['light_chain']]

    # drop duplicates of epitope-TRA-TRB triplets:
    df_iedb = df_iedb.drop_duplicates(subset=['epitope', 'heavy_chain', 'light_chain'])

    if replace_hashtag:
        # replace #'s with X in the sequences for TCRLang:
        df_iedb['heavy_chain'] = df_iedb['heavy_chain'].str.replace('#', 'X')
        df_iedb['light_chain'] = df_iedb['light_chain'].str.replace('#', 'X')
    
    # make the AA's upper case in the alpha and beta chains:
    df_iedb['heavy_chain'] = df_iedb['heavy_chain'].str.upper()
    df_iedb['light_chain'] = df_iedb['light_chain'].str.upper()

    # strip the peptides with whitespace:
    df_iedb['heavy_chain'] = df_iedb['heavy_chain'].str.strip()
    df_iedb['light_chain'] = df_iedb['light_chain'].str.strip()

    # reset index
    df_iedb = df_iedb.reset_index(drop=True)

    return df_iedb


def load_vdjdb_data_cdr3(tsv_path):
    print("path name:", tsv_path)
    df_vdj = pd.read_csv(tsv_path, delimiter='\t')

    # Get subset of columsn we are interested in
    df_vdj = df_vdj[["cdr3.alpha", "cdr3.beta", "species", "mhc.a", "mhc.b", "mhc.class", "antigen.epitope", "cdr3fix.alpha", "cdr3fix.beta"]]

    # subset to only paired data (both alpha and beta chain CDR3s are known)
    df_vdj = df_vdj.dropna(subset=["cdr3.alpha", "cdr3.beta"])

    # Extract the fixed CDR3 sequences and use those as the ground truth CDR3 sequence
    # https://github.com/antigenomics/vdjdb-db?tab=readme-ov-file#cdr3-sequence-fixing
    # There is always a fixed value for every existing CDR3, but sometimes the "fixed" value is the same as the empirical one
    df_vdj["heavy_chain"] = df_vdj["cdr3fix.alpha"].apply(json.loads).apply(vdjdb_extract_fixed_cdr3)
    df_vdj["light_chain"] = df_vdj["cdr3fix.beta"].apply(json.loads).apply(vdjdb_extract_fixed_cdr3)
    df_vdj = df_vdj.rename(columns={'antigen.epitope':'epitope'})

    # remove all examples where the heavy_chain and light_chain values are the same (likely means invalid pair):
    df_vdj = df_vdj[df_vdj["heavy_chain"] != df_vdj["light_chain"]] # only removes 1 entry

    # drop duplicates of epitope-TRA-TRB triplets:
    df_vdj = df_vdj.drop_duplicates(subset=['epitope', 'heavy_chain', 'light_chain']) # drops ~2700 entries

    # make the AA's upper case in the alpha and beta chains (shouldn't be necessary, but just to be safe)
    df_vdj['heavy_chain'] = df_vdj['heavy_chain'].str.upper()
    df_vdj['light_chain'] = df_vdj['light_chain'].str.upper()

    df_vdj = df_vdj.loc[df_vdj["species"] == "HomoSapiens"] # Before this point, species counts are {'HomoSapiens': 29556, 'MusMusculus': 2264}

    # reset index
    df_vdj = df_vdj.reset_index(drop=True)

    return df_vdj

def load_vdjdb_data_pmhc(tsv_path, mhc_map_path):
    df_mhc_map = pd.read_csv(mhc_map_path, delimiter='\t', index_col=0) # format where index is two-part name like "A*01:01" and the column label is "max_sequence" containing the sequence with signal peptide removed

    df_vdj = load_vdjdb_data_cdr3(tsv_path)

    # Map the MHC allele names to their sequences
    df_vdj["mhc.a_seq"] = df_vdj["mhc.a"].apply(functools.partial(map_mhc_allele, df_mhc_map=df_mhc_map))
    df_vdj["mhc.b_seq"] = df_vdj["mhc.b"].apply(functools.partial(map_mhc_allele, df_mhc_map=df_mhc_map))

    return df_vdj
    
    

# Helper function to extract fixed CDR3 from the VDJdb "cdr3fix.[alpha/beta]" column
def vdjdb_extract_fixed_cdr3(obj):
    return obj["cdr3"]

# Helper function to map a given MHC allele name to its sequence
def map_mhc_allele(allele_name, df_mhc_map):
    '''
    allele_name should be in the original VDJ format, such as "HLA-A*03:01"

    For allele_names that only specify type and no subtype (e.g. "HLA-B*08"), will map to subtype 01 (e.g. "HLA-B*08:01")

    For allele names not found in the MHC map, will map to canonical allele as specified here: https://www.ebi.ac.uk/ipd/imgt/hla/alignment/help/references/
    '''

    canonical_alleles = { # only explicitly listed subset found in vdjdb human data
        "HLA-A": "HLA-A*01:01",
        "HLA-B": "HLA-B*07:02",
        "HLA-C": "HLA-C*01:02",
        "HLA-E": "HLA-E*01:01",
        "HLA-DRA": "HLA-DRA*01:01",
        "HLA-DRB": "HLA-DRB1*01:01", # not in imgt/hla, but found in mixtcrpred so mapping to DRB1
        "HLA-DRB1": "HLA-DRB1*01:01",
        "HLA-DRB3": "HLA-DRB3*01:01",
        "HLA-DRB5": "HLA-DRB5*01:01",
        "HLA-DQA": "HLA-DQA1*01:01", # not in imgt/hla, but found in vdjdb so mapping to DQA1
        "HLA-DQA1": "HLA-DQA1*01:01",
        "HLA-DQB": "HLA-DQB1*05:01", # not in imgt/hla but found in mixtcrpred so mapping to DQB1
        "HLA-DQB1": "HLA-DQB1*05:01",
        "HLA-DPA": "HLA-DPA1*01:03", # not in imgt/hla but found in vdjdb so mappin gto DPA1
        "HLA-DPA1": "HLA-DPA1*01:03",
        "HLA-DPB": "HLA-DPB1*01:01", # not in imgt/hla but found in vdjdb so mappin gto DPB1
        "HLA-DPB1": "HLA-DPB1*01:01",
        # "DQ2": "HLA-DQA2*01:01", # found in mixtcrpred, unsure which DQ allele it is referring to, but only DQA has a canonical allele for 2. DQB only has canonical allele for 1
    }

    if allele_name == "B2M": # Handles beta chain placeholder for Class I MHCs that do not have a beta chain
        return "B2M"
    
    hla_id = allele_name
    if hla_id.startswith("HLA-"):
        components = hla_id.split(":")
        num_parts = len(components)
        if num_parts > 2:
            hla_id = ":".join(components[:2])
        elif num_parts == 1: # either is of form HLA-A*01 or HLA-A
            pattern = re.compile(r'(HLA-[A-Za-z0-9]+)\*([0-9]+)')
            if pattern.match(hla_id):
                hla_id = ":".join([components[0], "01"])
            else:
                pass # leave it as is
        
        if hla_id not in df_mhc_map.index:
            hla_gene = hla_id.split("*")[0]
            if hla_gene in canonical_alleles:
                hla_id = canonical_alleles[hla_gene]
            else:
                raise Exception(f"Could not find MHC allele {allele_name} in the MHC map and no canonical allele for {hla_gene} specified.")
    else: # handles mouse H2 alleles
        hla_id = allele_name
        

    allele_sequence = df_mhc_map.loc[hla_id, "max_sequence"]
    return allele_sequence



def load_pird_data_cdr3(csv_path):
    '''
    Load the PIRD dataset from a CSV file in latin-1 encoding (default from database download) and return a pandas dataframe.
    '''
    df_pird = pd.read_csv(csv_path, encoding='latin-1')
    df_pird = df_pird.replace('-', np.nan)

    # get subset of columns we are interested in:
    df_pird = df_pird[['Antigen.sequence', 'HLA', 'CDR3.alpha.aa', 'CDR3.beta.aa']]

    # subset to only paired data (both alpha and beta chain CDR3s are known)
    df_pird = df_pird.dropna(subset=["Antigen.sequence", "CDR3.alpha.aa", "CDR3.beta.aa"])

    # drop rows where Epitope name is not a peptide sequence:
    df_pird = df_pird[df_pird['Antigen.sequence'].str.isupper()]
    df_pird = df_pird[df_pird['Antigen.sequence'].apply(is_alpha_only)]

    # change column names:
    df_pird = df_pird.rename(columns={'Antigen.sequence': 'epitope',
                                      'CDR3.alpha.aa': 'heavy_chain',
                                      'CDR3.beta.aa': 'light_chain'})
    
    # remove all examples where the heavy_chain and light_chain values are the same (likely means invalid pair):
    df_pird = df_pird[df_pird["heavy_chain"] != df_pird["light_chain"]] # only removes 1 entry

    # drop duplicates of epitope-TRA-TRB triplets:
    df_pird = df_pird.drop_duplicates(subset=['epitope', 'heavy_chain', 'light_chain']) # drops 82 entries

    # make the AA's upper case in the alpha and beta chains (shouldn't be necessary, but just to be safe)
    df_pird['heavy_chain'] = df_pird['heavy_chain'].str.upper()
    df_pird['light_chain'] = df_pird['light_chain'].str.upper()

    # reset index
    df_pird = df_pird.reset_index(drop=True)

    return df_pird

# Function to check if the string contains only alphabetic characters
def is_alpha_only(s):
    return s.isalpha()

def insert_spaces(sequence):
    # Regular expression to match single amino acids or special tokens like '[UNK]'
    pattern = re.compile(r'\[.*?\]|.')
    
    # Find all matches and join them with a space
    spaced_sequence = ' '.join(pattern.findall(sequence))
    
    return spaced_sequence


def construct_label_matrices(epitope_seqs, receptor_seqs, include_mhc):
    if include_mhc:
        epitope_seqs = epitope_seqs[0] # extract epitope seqs from the array [epitopes_seqs, mhca_seqs, mhcb_seqs]
    bs = len(epitope_seqs)

    # Create a 2D tensor filled with zeros
    label_matrix = torch.zeros((bs, bs), dtype=torch.float32)
    # Construct the label matrix
    for i, correct_ep in enumerate(epitope_seqs):
        count = epitope_seqs.count(correct_ep)
        for j, ep in enumerate(epitope_seqs):
            if ep == correct_ep:
                label_matrix[i, j] = 1.0 / count

    return label_matrix

def construct_label_matrices_ones(epitope_seqs, receptor_seqs, include_mhc):
    if include_mhc:
        epitope_seqs = epitope_seqs[0] # extract epitope seqs from the array [epitopes_seqs, mhca_seqs, mhcb_seqs]
    bs = len(epitope_seqs)

    # Create a 2D tensor filled with zeros
    label_matrix = torch.zeros((bs, bs), dtype=torch.float32)
    # Construct the label matrix
    for i, correct_ep in enumerate(epitope_seqs):
        for j, ep in enumerate(epitope_seqs):
            if ep == correct_ep:
                label_matrix[i, j] = 1.0

    return label_matrix

def apply_masking_seq(sequences, mask_token='.', mask_regions=True, p=0.15):
    '''
    mask_regions: True or List[np.array(dtype=bool)]
    - if True, all amino acids will be considered
    - if List, only amino acids with True values in the list will be considered (i.e. mask for regions to mask)

    For each sequence string in the sequences list, apply masking by changing the 
    amino acid with the mask_token with a certain probability.
    '''

    if mask_regions is True: # convert True value into all-True arrays every sequence
        mask_regions = [np.ones(len(seq), dtype=bool) for seq in sequences]

    masked_sequences = []
    mask_indices = []
    for n, seq in enumerate(sequences):
        masked_seq = ''
        # seq_mask_indices = []
        mask_count = 0
        for i, aa in enumerate(seq):
            if mask_regions[n][i] and torch.rand(1) < p and mask_count < sum(mask_regions[n]) - 1:
                masked_seq += mask_token
                mask_indices.append([n, i])
                mask_count += 1
            else:
                masked_seq += aa
        masked_sequences.append(masked_seq)
    
    return masked_sequences, mask_indices