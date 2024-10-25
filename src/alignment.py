import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .models import *
from .utils import construct_label_matrices, construct_label_matrices_ones
from .data_module import compute_weights
from .lr_scheduler import CosineAnnealingWarmUpRestarts

class CLIPModel(pl.LightningModule):
    def __init__(self, lightning_config, model_config, device='cuda', **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.ln_cfg = lightning_config
        self.model_config = model_config

        self.epitope_input_dim = model_config.epitope_input_dim
        self.receptor_input_dim = model_config.receptor_input_dim
        self.projection_dim = model_config.projection_dim
        self.hidden_dim = model_config.hidden_dim

        # loss functions:
        self.bceloss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        self.mse_weight = lightning_config.mse_weight
        self.epitope_weights = None

        # logging:
        self.log_iterations = None
        self.training_step_metrics = {}
        self.val_step_metrics = {}
        self.test_step_metrics = {}

        # for evaluation later:
        self.epitope_embeddings = []
        self.receptor_embeddings = []
        self.epitope_sequences = []
        self.receptor_sequences = []


    def forward(self, epitope_seqs, receptor_seqs, mask=False):
        epitope_proj = self.epitope_encoder(epitope_seqs, mask=mask)
        receptor_proj = self.receptor_encoder(receptor_seqs, mask=mask)
        return epitope_proj, receptor_proj

    
    def clip_loss_multiclass(self, epitope_features, receptor_features, label_matrix, temperature=1.0):
        """
        Compute the multi-class CLIP loss for epitope and receptor features based on label_indices

        Args:
            epitope_features: Tensor of shape (batch_size, feature_dim) representing the epitope embeddings.
            receptor_features: Tensor of shape (batch_size, feature_dim) representing the receptor embeddings.
            label_indices: list of length (batch_size) where each element is a list of indices 
            of the correct labels for each epitope.
            temperature: A scaling factor to control the sharpness of the similarity distribution.
        
        Returns:
            loss: A scalar tensor representing the multi-class CLIP loss.
        """

        # Normalize the features to unit length (each dim bs x proj_dim)
        epitope_features = F.normalize(epitope_features, dim=-1)
        receptor_features = F.normalize(receptor_features, dim=-1)

        # MSE Loss between the normalized features:
        diff_norm = torch.norm(epitope_features - receptor_features, dim=-1)
        mse_loss = F.mse_loss(diff_norm, torch.zeros(len(diff_norm)).to(self.device), reduction='mean')

        # Compute the logits (similarities) as the dot product of epitope and receptor features
        logits_per_epitope = epitope_features @ receptor_features.t()
        logits_per_receptor = receptor_features @ epitope_features.t()

        # Scale by temperature
        logits_per_epitope /= temperature
        logits_per_receptor /= temperature

        # Compute the cross-entropy loss for both epitope-to-receptor and receptor-to-epitope
        # epitopes_loss = self.celoss(logits_per_epitope, label_matrix)
        # receptor_loss = self.celoss(logits_per_receptor, label_matrix)

        # Compute the binary cross-entropy loss for both epitope-to-receptor and receptor-to-epitope
        epitopes_loss = self.bceloss_logits(logits_per_epitope, label_matrix)
        receptor_loss = self.bceloss_logits(logits_per_receptor, label_matrix)

        # multiply the loss with inverse square-rooted count weights:

        clip_loss =  (epitopes_loss + receptor_loss) / 2.0 # shape: (batch_size)
        return clip_loss, mse_loss


    def training_step(self, batch, batch_idx):
        """
        Training step for the CLIPBody Model
        """
        epitope_seqs, receptor_seqs = batch

        epitope_proj, receptor_proj = self(epitope_seqs, receptor_seqs, mask=self.ln_cfg.mask_seqs)

        # label_matrix = construct_label_matrices(epitope_seqs, receptor_seqs).to(self.device)
        label_matrix = construct_label_matrices_ones(epitope_seqs, receptor_seqs, self.ln_cfg.include_mhc).to(self.device)

        # print("epitope seqs:", epitope_seqs)

        # construct weight matrices for the epitope sequences:
        if self.ln_cfg.weigh_epitope_count:
            weights = torch.tensor([self.epitope_weights[seq] for seq in epitope_seqs]).to(self.device)
            clip_loss, mse_loss = self.clip_loss_multiclass(epitope_proj, receptor_proj, label_matrix, temperature=0.07)
            clip_loss = clip_loss * weights
            clip_loss = clip_loss.sum()
        else:
            clip_loss, mse_loss = self.clip_loss_multiclass(epitope_proj, receptor_proj, label_matrix, temperature=0.07)
            clip_loss = clip_loss.mean()
        
        loss = clip_loss * (1 - self.mse_weight) + mse_loss * self.mse_weight
        training_metrics = {
            'loss': loss,
        }
        self.training_step_metrics.setdefault('loss', []).append(loss.detach().item())
        if self.ln_cfg.mse_weight > 0:
            self.training_step_metrics.setdefault('clip_loss', []).append(clip_loss.detach().item())
            self.training_step_metrics.setdefault('mse_loss', []).append(mse_loss.detach().item())

        return training_metrics
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the CLIPBody Model
        """
        
        epitope_seqs, receptor_seqs = batch
        try:
            epitope_proj, receptor_proj = self(epitope_seqs, receptor_seqs)
        except:
            print("Error in feeding sequences")
            print("epitope_seqs", epitope_seqs)
            print("receptor_seqs", receptor_seqs)
            raise ValueError
            
        # label_matrix = construct_label_matrices(epitope_seqs, receptor_seqs).to(self.device)
        label_matrix = construct_label_matrices_ones(epitope_seqs, receptor_seqs, self.ln_cfg.include_mhc).to(self.device)

        # construct weight matrices for the epitope sequences:
        if self.ln_cfg.weigh_epitope_count and not self.ln_cfg.unique_epitopes:
            weights = torch.tensor([self.epitope_weights[seq] for seq in epitope_seqs]).to(self.device)
            clip_loss, mse_loss = self.clip_loss_multiclass(epitope_proj, receptor_proj, label_matrix, temperature=0.07)
            clip_loss = clip_loss * weights
            clip_loss = clip_loss.sum()
        else:
            clip_loss, mse_loss = self.clip_loss_multiclass(epitope_proj, receptor_proj, label_matrix, temperature=0.07)
            clip_loss = clip_loss.mean()
                
        loss = clip_loss * (1 - self.mse_weight) + mse_loss * self.mse_weight
        val_metrics = {
            'loss': loss,
        }
        self.val_step_metrics.setdefault('loss', []).append(loss.detach().item())
        if self.ln_cfg.mse_weight > 0:
            self.val_step_metrics.setdefault('clip_loss', []).append(clip_loss.detach().item())
            self.val_step_metrics.setdefault('mse_loss', []).append(mse_loss.detach().item())

        return val_metrics
    
    def test_step(self, batch, batch_idx):
        """
        Test step for the CLIPBody Model
        """
        epitope_seqs, receptor_seqs = batch
        epitope_proj, receptor_proj = self(epitope_seqs, receptor_seqs)

        # save the embeddings batches for evaluation later
        if self.ln_cfg.include_mhc:
            epitope_seqs_to_save = epitope_seqs[0] # extract epitope seqs from the array [epitopes_seqs, mhca_seqs, mhcb_seqs]
        else:
            epitope_seqs_to_save = epitope_seqs
        self.epitope_sequences.append(epitope_seqs_to_save)
        self.receptor_sequences.append(receptor_seqs)
        self.epitope_embeddings.append(epitope_proj)
        self.receptor_embeddings.append(receptor_proj)

        # label_matrix = construct_label_matrices(epitope_seqs, receptor_seqs).to(self.device)
        label_matrix = construct_label_matrices_ones(epitope_seqs, receptor_seqs, self.ln_cfg.include_mhc).to(self.device)

        clip_loss, mse_loss = self.clip_loss_multiclass(epitope_proj, receptor_proj, label_matrix, temperature=0.07)

        clip_loss = clip_loss.mean()
        
        loss = clip_loss * (1 - self.mse_weight) + mse_loss * self.mse_weight
        test_metrics = {
            'loss': loss,
        }
        self.test_step_metrics.setdefault('loss', []).append(loss.detach().item())
        if self.ln_cfg.mse_weight > 0:
            self.test_step_metrics.setdefault('clip_loss', []).append(clip_loss.detach().item())
            self.test_step_metrics.setdefault('mse_loss', []).append(mse_loss.detach().item())

        return test_metrics
    
    def on_fit_start(self):
        # compute the weights for each epitope sequence
        if self.ln_cfg.weigh_epitope_count:
            print("Weighing the Epitopes by inverse sqrt of their counts!")
            self.epitope_weights = compute_weights(self.trainer.datamodule.train_dataloader().dataset.data['epitope'].tolist())

    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        for metric, values in self.training_step_metrics.items():
            avg_metric = self.aggregate_metric(values)
            self.log(f'train_{metric}', avg_metric, prog_bar=False, sync_dist=True)
            print(f'Epoch train end: {metric}/train', avg_metric)
        self.training_step_metrics.clear()

        for metric, values in self.val_step_metrics.items():
            avg_metric = self.aggregate_metric(values)
            self.log(f'val_{metric}', avg_metric, prog_bar=False, sync_dist=True)
            print(f'Epoch validation end: {metric}/val', avg_metric)
        self.val_step_metrics.clear()
    
    def on_test_epoch_end(self):
        for metric, values in self.test_step_metrics.items():
            avg_metric = self.aggregate_metric(values)
            # self.log(f'test_{metric}', avg_metric, prog_bar=False, sync_dist=True)
            print(f'Epoch test end: {metric}/test', avg_metric)
        self.test_step_metrics.clear()

        # save the embeddings as numpy arrays:
        if self.ln_cfg.save_embed_path:
            if not os.path.isdir(self.ln_cfg.save_embed_path):
                os.makedirs(self.ln_cfg.save_embed_path)

            epitope_sequences = np.concatenate(self.epitope_sequences, axis=0)
            receptor_sequences = np.concatenate(self.receptor_sequences, axis=1)
            epitope_embeddings = torch.cat(self.epitope_embeddings, dim=0).detach().cpu().numpy()
            receptor_embeddings = torch.cat(self.receptor_embeddings, dim=0).detach().cpu().numpy()

            # actually save the embeds
            print("Saving sequences and embeddings to disk...")
            np.save(self.ln_cfg.save_embed_path + '/epitope_seqs.npy', epitope_sequences)
            np.save(self.ln_cfg.save_embed_path + '/receptor_seqs.npy', receptor_sequences)
            np.save(self.ln_cfg.save_embed_path + '/epitope_embeds.npy', epitope_embeddings)
            np.save(self.ln_cfg.save_embed_path + '/receptor_embeds.npy', receptor_embeddings)


    @staticmethod
    def aggregate_metric(step_outputs):
        return np.mean(step_outputs)

    def configure_optimizers(self):
        if self.ln_cfg.regular_ft:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.ln_cfg.lr, weight_decay=self.ln_cfg.weight_decay)
            return {
                "optimizer": optimizer,
            }

        if self.ln_cfg.lr_scheduler == 'plateau':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.ln_cfg.lr, weight_decay=self.ln_cfg.weight_decay)
            scheduler_lr = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2, min_lr=1e-6)
        
        elif self.ln_cfg.lr_scheduler == 'cos_anneal':
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=self.ln_cfg.weight_decay)
            scheduler_lr = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=self.ln_cfg.lr, T_up=2, gamma=0.7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_lr,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    
    def configure_model(self):

        self.epitope_encoder = EpitopeEncoderESM(self.epitope_input_dim, self.projection_dim, hidden_dim=self.hidden_dim,
                                                 ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)

        if self.model_config.receptor_model_name == 'ablang':
            self.receptor_encoder = AntibodyEncoderAbLang(self.receptor_input_dim, self.projection_dim, ln_cfg=self.ln_cfg, device=self.device)
        elif self.model_config.receptor_model_name == 'ablang2':
            self.receptor_encoder = AntibodyEncoderAbLang2(self.receptor_input_dim, self.projection_dim, ln_cfg=self.ln_cfg, device=self.device)
        elif self.model_config.receptor_model_name == 'antiberta2':
            self.receptor_encoder = AntibodyEncoderAntiberta2(self.receptor_input_dim, self.projection_dim, ln_cfg=self.ln_cfg, device=self.device)
        elif self.model_config.receptor_model_name == 'tcrbert':
            self.receptor_encoder = TCREncoderTCRBert(self.receptor_input_dim, self.projection_dim, 
                                                      hidden_dim=self.hidden_dim, ln_cfg=self.ln_cfg, device=self.device)
        elif self.model_config.receptor_model_name == 'tcrlang':
            self.receptor_encoder = TCREncoderTCRLang(self.receptor_input_dim, self.projection_dim, 
                                                      hidden_dim=self.hidden_dim, ln_cfg=self.ln_cfg, device=self.device)
        elif self.model_config.receptor_model_name in ['esm2', 'esm3']:
            if "catcr" in self.ln_cfg.dataset_path:
                self.receptor_encoder = TCREncoderESMBetaOnly(self.receptor_input_dim, self.projection_dim, hidden_dim=self.hidden_dim,
                                                              ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)
            else:
                #TODO: REPLACE THIS LINE!
                # self.receptor_encoder = TCREncoderESMBetaOnly(self.receptor_input_dim, self.projection_dim, hidden_dim=self.hidden_dim,
                #                                               ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)
                self.receptor_encoder = TCREncoderESM(self.receptor_input_dim, self.projection_dim, hidden_dim=self.hidden_dim,
                                                      ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)
        elif self.model_config.receptor_model_name == 'inhouse':
            self.receptor_encoder = TCREncoderInHouse(self.receptor_input_dim, self.projection_dim, hidden_dim=self.hidden_dim,
                                                      ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)
        elif self.model_config.receptor_model_name == 'onehot':
            self.epitope_encoder = EpitopeEncoderOneHot(self.epitope_input_dim, self.projection_dim,
                                                        ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)
            self.receptor_encoder = TCREncoderOneHot(self.receptor_input_dim, self.projection_dim,
                                                      ln_cfg=self.ln_cfg, model_config=self.model_config, device=self.device)

        else:
            raise NotImplementedError("Such Ab Model not implemented yet. Please choose from existing models.")


    # for inference later
    def put_submodules_to_device(self, device):
        self.epitope_encoder.device = device
        self.receptor_encoder.device = device