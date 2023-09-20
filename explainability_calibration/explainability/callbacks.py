from collections import defaultdict
import re
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
import torch
from sklearn.metrics import f1_score
from .utils import align_annotations_with_model

class ARTokenF1ExplainabilityCallback(Callback):
    
    def __init__(self, tokenizer, add_residuals=True, idx_layer=2):
        self.tokenizer = tokenizer
        self.add_residuals = add_residuals
        self.idx_layer = idx_layer
        self.batch_results = {"attentions": [], "WO": [], "WY": [], "BO": [], "BY": [], "LO": [], "LY": []}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        attention_tokens = self._rollout_attentions(outputs["attentions"], add_residual=self.add_residuals, idx_layer=self.idx_layer)
        attention_tokens = attention_tokens.cpu().numpy()
        
        for sentence, input_ids, attention, wo, wy, bo, by, lo, ly in zip(
            batch["sentence"], batch["input_ids"], attention_tokens, batch["WO"], batch["WY"], batch["BO"], batch["BY"], batch["LO"], batch["LY"]
        ):
            if isinstance(sentence,list):
                sentence = self.tokenizer.sep_token.join(sentence)
            attention = self._simplify_attention(attention, input_ids)
            self.batch_results["attentions"].append(attention)
            
            groups = {"WO": wo, "WY": wy, "BO": bo, "BY": by, "LO": lo, "LY": ly}
            for name, group in groups.items():
                new_annotations = align_annotations_with_model(self.tokenizer, sentence, group)
                self.batch_results[name].append(new_annotations)

    def _simplify_attention(self, attention, input_ids):
        return [att for att, iid in zip(attention,input_ids) if iid not in self.tokenizer.all_special_ids]
        
    def on_validation_epoch_end(self, trainer, pl_module):
        groups = [g for g in self.batch_results.keys() if g != "attentions"]
        rlr = {group: np.mean([sum(r) / len(r) for r in self.batch_results[group]]) for group in groups}
        num_samples = len(self.batch_results["attentions"])
        for i in range(num_samples):
            attention = self.batch_results["attentions"][i]
            for group in groups:
                top_kd = self._compute_top_kd(attention, rlr[group])
                token_f1 = self._compute_token_f1(top_kd, self.batch_results[group][i])
                pl_module.log(f"loss/token_f1_attention_rollout_layer_{self.idx_layer:02d}_group_{group}", token_f1)
        self.batch_results = {"attentions": [], "WO": [], "WY": [], "BO": [], "BY": [], "LO": [], "LY": []}

    @staticmethod
    def _compute_top_kd(attentions, rlr):
        top_kd = int(np.round(rlr * len(attentions)))
        top_kd_tokens = np.argsort(attentions)[:-top_kd-1:-1]
        top_kd_binary = np.zeros(len(attentions), dtype=int)
        top_kd_binary[top_kd_tokens] = 1
        return top_kd_binary

    @staticmethod
    def _compute_token_f1(top_kd, binary_rationale):
        return f1_score(binary_rationale,top_kd)

    @staticmethod
    def _rollout_attentions(attention_batches, add_residual=True, idx_layer=2):
        attentions = torch.stack(attention_batches,dim=1)
        attentions = attentions.sum(dim=2) / attentions.shape[2] # (batch, num_heads, seq_len, seq_len)
        
        if add_residual:
            residual_att = torch.eye(attentions.shape[2],device=attentions.device)[None, None, ...]
            aug_att_mat = attentions + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
        else:
            aug_att_mat = attentions

        joint_attentions = torch.zeros(aug_att_mat.shape,device=aug_att_mat.device)

        layers = joint_attentions.shape[1]
        joint_attentions[:,0] = aug_att_mat[:,0]
        for i in range(1, layers):
            joint_attentions[:,i,:,:] = torch.bmm(aug_att_mat[:,i,:,:], joint_attentions[:,i-1,:,:])
        
        attention_tokens = joint_attentions[:,idx_layer].sum(dim=1)

        return attention_tokens


    

    

