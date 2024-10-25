import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split, Sampler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
from collections import defaultdict, deque
import random
import os

from .utils import (load_iedb_data, load_iedb_data_cdr3, load_vdjdb_data_cdr3, 
                    load_vdjdb_data_pmhc, load_pird_data_cdr3, load_mixtcrpred_data, 
                    load_mixtcrpred_data_pmhc, load_catcr_data)

class EpitopeReceptorDataset(Dataset):
    '''Returns receptor paired with epitope-only data'''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("idx", idx)
        epitope_seq = self.data.iloc[idx]['epitope']
        heavy_chain_seq = self.data.iloc[idx]['heavy_chain']
        light_chain_seq = self.data.iloc[idx]['light_chain']
        return epitope_seq, (heavy_chain_seq, light_chain_seq)
    
class pMhcReceptorDataset(Dataset):
    '''Returns receptor paired with epitope+MHC data'''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        epitope_seq = self.data.iloc[idx]['epitope']
        mhc_a = self.data.iloc[idx]['mhc.a_seq']
        mhc_b = self.data.iloc[idx]['mhc.b_seq']
        heavy_chain_seq = self.data.iloc[idx]['heavy_chain']
        light_chain_seq = self.data.iloc[idx]['light_chain']
        return (epitope_seq, mhc_a, mhc_b), (heavy_chain_seq, light_chain_seq)

class EpitopeReceptorDataModule(pl.LightningDataModule):
    def __init__(self, tsv_file, mhc_file=None, batch_size=32, include_mhc=False, ln_cfg = None,
                 model_config = None, split_ratio=(0.7, 0.15, 0.15), random_seed=7):
        super().__init__()
        self.tsv_file = tsv_file
        self.batch_size = batch_size
        self.model_config = model_config
        self.ln_cfg = ln_cfg
        self.include_mhc = include_mhc
        self.mhc_file = mhc_file
        if self.include_mhc:
            assert self.mhc_file is not None, "Must provide a file with MHC data"
        self.split_ratio = split_ratio

        self.random_seed = random_seed

    def prepare_data_must(self):
        # Read the TSV file
        if 'IEDB' in self.tsv_file:
            if self.model_config.receptor_model_name == 'ablang':
                self.data = load_iedb_data(self.tsv_file, replace_X=True)
            elif self.model_config.receptor_model_name == 'tcrlang':
                self.data = load_iedb_data_cdr3(self.tsv_file, replace_hashtag=True)
            elif self.model_config.receptor_model_name == 'tcrbert':
                self.data = load_iedb_data_cdr3(self.tsv_file)
            else:
                self.data = load_iedb_data(self.tsv_file)
        elif 'vdjdb' in self.tsv_file:
            if self.include_mhc:
                self.data = load_vdjdb_data_pmhc(self.tsv_file, self.mhc_file)
            else:
                self.data = load_vdjdb_data_cdr3(self.tsv_file)
        elif 'mixtcrpred' in self.tsv_file:
            if self.include_mhc:
                self.data = load_mixtcrpred_data_pmhc(self.tsv_file, self.mhc_file)
            else:
                self.data = load_mixtcrpred_data(self.tsv_file)
        elif 'pird' in self.tsv_file:
            self.data = load_pird_data_cdr3(self.tsv_file)
        elif 'catcr' in self.tsv_file:
            self.data = load_catcr_data(self.tsv_file)

            # self.train_data, self.test_data = load_catcr_data(self.tsv_file)
            # return
        else:
            raise ValueError(f"Can't process this tsv file: {self.tsv_file}")

        # Ensure the data has the correct columns
        assert 'epitope' in self.data.columns
        assert 'heavy_chain' in self.data.columns
        assert 'light_chain' in self.data.columns

    def split_data_random(self):
        if self.ln_cfg.unique_epitopes:
            # ------------------------------------------------------
            # Splitting data via unique epitopes:

            np.random.seed(self.random_seed)

            # Get unique values in epitope column
            unique_epitopes = self.data['epitope'].unique()

            # Shuffle the unique values:
            np.random.shuffle(unique_epitopes)

            # Split the unique values into train, dev, and test sets
            train_size = int(self.split_ratio[0] * len(unique_epitopes))
            dev_size = int(self.split_ratio[1] * len(unique_epitopes))
            test_size = len(unique_epitopes) - train_size - dev_size

            train_values = unique_epitopes[:train_size]#[:100]
            dev_values = unique_epitopes[train_size:train_size + dev_size]
            test_values = unique_epitopes[train_size + dev_size:]

            # Create train, dev, and test dataframes
            # making sure that each set has a unique set of epitopes
            self.train_data = self.data[self.data['epitope'].isin(train_values)]
            self.dev_data = self.data[self.data['epitope'].isin(dev_values)]
            self.test_data = self.data[self.data['epitope'].isin(test_values)]

        elif self.ln_cfg.fewshot_ratio:
            self.train_data, self.dev_data = split_df_by_ratio(self.data, 0.85, random_seed=self.random_seed)
            if self.ln_cfg.fewshot_ratio < 1:
                self.train_data, _ = split_df_by_ratio(self.train_data, self.ln_cfg.fewshot_ratio, random_seed=self.random_seed)
            self.test_data = self.dev_data.copy()

        else:
            # ------------------------------------------------------
            # Split the data into train, dev, and test sets
            total_size = len(self.data)
            train_size = int(self.split_ratio[0] * total_size)
            dev_size = int(self.split_ratio[1] * total_size)
            test_size = total_size - train_size - dev_size

            self.train_data, self.temp = train_test_split(self.data, test_size=0.3, random_state=self.random_seed)
            self.dev_data, self.test_data = train_test_split(self.temp, test_size=0.5, random_state=self.random_seed)
        
        # # oversample here:
        # if self.ln_cfg.oversample:
        #     self.train_data = upsample_epitopes(self.train_data, 'epitope')

        # Reset the index of the dataframes
        self.train_data = self.train_data.reset_index(drop=True)
        self.dev_data = self.dev_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)


    def setup(self, stage=None):
        self.prepare_data_must()
        
        if "catcr" in self.tsv_file:
            # # copy the test data into dev:
            # self.dev_data = self.test_data.copy()
            self.split_data_random()
        else:
            self.split_data_random()
        
        if self.ln_cfg.save_embed_path:
            self.save_datasplit(self.ln_cfg.save_embed_path)

    def train_dataloader(self):
        if self.include_mhc:
            train_dataset = pMhcReceptorDataset(self.train_data)
        else:
            train_dataset = EpitopeReceptorDataset(self.train_data)

        if self.ln_cfg.oversample:
            train_sampler = OversampleSampler(self.train_data, self.batch_size)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=train_sampler, 
                              num_workers=4, persistent_workers=True)
        else:
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        if self.include_mhc:
            dev_dataset = pMhcReceptorDataset(self.dev_data)
        else:
            dev_dataset = EpitopeReceptorDataset(self.dev_data)
        # return DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False,
        #                   num_workers=4, pin_memory=True, persistent_workers=True)
        return DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=4, persistent_workers=True)


    def test_dataloader(self):
        if self.include_mhc:
            test_dataset = pMhcReceptorDataset(self.test_data)
        else:
            test_dataset = EpitopeReceptorDataset(self.test_data)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    

    def save_datasplit(self, savepath):
        '''
        Save the pandas dataframes of train/dev/test splits to a specified path
        '''

        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        train_path = os.path.join(savepath, 'train.tsv')
        dev_path = os.path.join(savepath, 'dev.tsv')
        test_path = os.path.join(savepath, 'test.tsv')

        self.train_data.to_csv(train_path, sep='\t', index=False)
        self.dev_data.to_csv(dev_path, sep='\t', index=False)
        self.test_data.to_csv(test_path, sep='\t', index=False)


# class OversampleSampler(Sampler):
#     def __init__(self, df, batch_size):
#         self.df = df
#         self.batch_size = batch_size
#         self.indices = self.generate_indices()

#     def generate_indices(self):
#         epitope_counts = self.df['epitope'].value_counts()
#         max_count = epitope_counts.max()

#         # Group indices by epitope and shuffle them
#         epitope_index_dict = {
#             epitope: np.random.permutation(self.df[self.df['epitope'] == epitope].index.tolist() * int( max_count // count )).tolist()
#             for epitope, count in epitope_counts.items()
#         }

#         # Generate batches with as distinct epitopes as possible
#         batched_indices = []
#         while any(epitope_index_dict.values()):
#             batch = []
#             available_epitopes = [epitope for epitope, indices in epitope_index_dict.items() if indices]
#             np.random.shuffle(available_epitopes)
#             selected_epitopes = available_epitopes[:self.batch_size]

#             for epitope in selected_epitopes:
#                 if epitope_index_dict[epitope]:
#                     batch.append(epitope_index_dict[epitope].pop())

#             # Fill the remaining batch size with other available indices if needed
#             if len(batch) < self.batch_size:
#                 remaining_epitopes = [epitope for epitope, indices in epitope_index_dict.items() if indices]
#                 np.random.shuffle(remaining_epitopes)
#                 for epitope in remaining_epitopes:
#                     if len(batch) >= self.batch_size:
#                         break
#                     if epitope_index_dict[epitope]:
#                         batch.append(epitope_index_dict[epitope].pop())

#             np.random.shuffle(batch)
#             batched_indices.append(batch)
        
#         # shuffle the order of minibatches as well
#         np.random.shuffle(batched_indices)
#         batched_indices = sum(batched_indices, [])

#         return np.array(batched_indices)

#     def __iter__(self):
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)


class OversampleSampler(Sampler):
    def __init__(self, df, batch_size):
        self.df = df
        self.indices = self.generate_indices()
        
        # print("DF size: ", self.df.shape)
        # print("Oversampled indices:", self.indices[:1000])

    def generate_indices(self):
        epitope_counts = self.df['epitope'].value_counts()
        max_count = epitope_counts.max()

        oversample_indices = []
        for epitope, count in epitope_counts.items():
            epitope_indices = self.df[self.df['epitope'] == epitope].index.tolist()
            oversample_ratio = max_count // count # int( np.sqrt(max_count // count) )
            oversample_indices.extend(epitope_indices * oversample_ratio)

        # return oversample_indices
        
        # Shuffle the oversampled indices to ensure randomness
        np.random.shuffle(oversample_indices)
        return np.array(oversample_indices)

    def __iter__(self):
        # # Shuffle the oversampled indices to ensure randomness
        # np.random.shuffle(self.indices)
        # return iter(np.array(self.indices))
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class UniqueValueSampler(Sampler):
    def __init__(self, dataframe, batch_size, seed=42):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.unique_values = list(dataframe['epitope'].unique())
        self.original_indices_by_value = defaultdict(list)
        self.indices_by_value = defaultdict(list)
        
        for idx, value in enumerate(dataframe['epitope']):
            self.original_indices_by_value[value].append(idx)
        
        self.seed = seed
        self.reset_indices()

    def reset_indices(self):
        # Reset the indices for each unique value from the original indices
        self.indices_by_value = {value: indices[:] for value, indices in self.original_indices_by_value.items()}

    def __iter__(self):
        random.seed(self.seed)
        shuffled_unique_values = self.unique_values[:]
        random.shuffle(shuffled_unique_values)
        
        batches = []
        current_batch = []
        used_values = set()

        for value in shuffled_unique_values:
            if value not in used_values and self.indices_by_value[value]:
                index = self.indices_by_value[value].pop(0)
                current_batch.append(index)
                used_values.add(value)
            
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
                used_values.clear()
        
        # Add the last batch if it contains any items
        if current_batch:
            batches.append(current_batch)

        # Ensure we cover all indices, even if they don't form a full batch
        remaining_indices = [idx for value in shuffled_unique_values for idx in self.indices_by_value[value]]
        for i in range(0, len(remaining_indices), self.batch_size):
            batches.append(remaining_indices[i:i+self.batch_size])

        # Flatten the list of batches to a list of indices
        flattened_batches = [idx for batch in batches for idx in batch]

        self.reset_indices()  # Reset indices for the next epoch
        return iter(flattened_batches)
    
    def __len__(self):
        return len(self.dataframe)


def compute_weights(epitope_seqs):
    '''
    given a list of redundant epitope sequences, count the number of times each unique epitope appears
    and compute the inverse square-rooted count weights for each epitope
    and save them into a dictionary
    '''
    epitope_weights = {}
    for seq in epitope_seqs:
        if seq in epitope_weights:
            epitope_weights[seq] += 1
        else:
            epitope_weights[seq] = 1
    
    # compute the inverse square-rooted count weights
    for seq in epitope_weights:
        epitope_weights[seq] = np.sqrt(1 / epitope_weights[seq])
    
    return epitope_weights


def upsample_epitopes(df: pd.DataFrame, epitope_column: str) -> pd.DataFrame:
    """
    Upsample the DataFrame so that each unique epitope is repeated by the ratio
    of max count to its current count.

    Parameters:
    - df: pandas DataFrame containing the data.
    - epitope_column: name of the column containing the epitope identifiers.

    Returns:
    - upsampled_df: pandas DataFrame with epitopes upsampled by the calculated ratio.
    """
    # Step 1: Count the number of entries for each epitope
    epitope_counts = df[epitope_column].value_counts()

    # Step 2: Find the maximum count
    max_count = epitope_counts.max()

    # Step 3: Function to upsample each group by the ratio
    def upsample(group):
        # Calculate the number of repetitions needed for each epitope group
        num_repeats = max_count // len(group)
        # Repeat each group by the calculated number of repeats
        return group.loc[group.index.repeat(num_repeats)]

    # Step 4: Apply the upsample function to each epitope group
    upsampled_df = df.groupby(epitope_column, group_keys=False).apply(upsample)

    return upsampled_df

def split_df_by_ratio(df, r, random_seed=14):
    # Create two empty lists to hold the dataframes for the two sets
    df_1_list = []
    df_2_list = []

    # Group the dataframe by the epitope
    grouped = df.groupby('epitope')

    # Iterate through each group
    for epitope, group in grouped:
        # Shuffle the group
        shuffled_group = group.sample(frac=1, random_state=random_seed)

        # Determine the split index
        split_idx = int(len(shuffled_group) * r)

        # Split the group into two parts based on the ratio
        df_1_list.append(shuffled_group.iloc[:split_idx])
        df_2_list.append(shuffled_group.iloc[split_idx:])

    # Concatenate all the individual dataframes to create the final dataframes
    df_1 = pd.concat(df_1_list).reset_index(drop=True)
    df_2 = pd.concat(df_2_list).reset_index(drop=True)

    return df_1, df_2