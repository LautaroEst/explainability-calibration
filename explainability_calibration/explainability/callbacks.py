from collections import defaultdict
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torch
from .utils import merge_symbols, merge_subwords, merge_hyphens

class ARTokenF1ExplainabilityCallback(Callback):
    
    def __init__(self, tokenizer, add_residuals=True, idx_layer=2):
        
        self.tokenizer = tokenizer
        self.add_residuals = add_residuals
        self.idx_layer = idx_layer
        self.batch_results = defaultdict(lambda: [])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        attention_tokens = self._rollout_attentions(outputs["attentions"], add_residual=self.add_residuals, idx_layer=self.idx_layer)
        attention_tokens = attention_tokens.cpu().numpy()
        # Merge tokens if necessary
        for i, (input_ids, label, sentence_id, wo, wy, bo, by, lo, ly) in enumerate(
            zip(batch["input_ids"], batch["label"], batch["sentence_id"], batch["WO"], batch["WY"], batch["BO"], batch["BY"], batch["LO"], batch["LY"])
        ):
            input_tokens = self.tokenizer.decode(
                input_ids, skip_special_tokens=True
            )
            input_tokens = input_tokens.split()
            input_tokens_merged, merged_attention = merge_symbols(
                input_tokens, attention_tokens[i, 1 : len(input_tokens) + 1]
            )
            input_tokens_merged, merged_attention = merge_subwords(
                input_tokens_merged, merged_attention
            )
            input_tokens_merged, merged_attention = merge_hyphens(
                input_tokens_merged, merged_attention
            )
            self.batch_results["tokens_input_ids"].append(input_ids.tolist())
            self.batch_results["token_input"].append(input_tokens)
            self.batch_results["token_merged"].append(input_tokens_merged)
            self.batch_results["attention"].append(merged_attention)
            self.batch_results["label"].append(label)
            self.batch_results["sentence_id"].append(sentence_id)
            self.batch_results["WO"].append(wo)
            self.batch_results["WY"].append(wy)
            self.batch_results["BO"].append(bo)
            self.batch_results["BY"].append(by)
            self.batch_results["LO"].append(lo)
            self.batch_results["LY"].append(ly)
            

    def on_validation_epoch_end(self, trainer, pl_module):
        import pdb; pdb.set_trace()
        topk_dg = self._compute_topk_dg(self.batch_results["attention"], self.batch_results["sentence_id"])
        for group in ["WO", "WY", "BO", "BY", "LO", "LY"]:
            token_f1 = self._compute_token_f1(topk_dg, self.batch_results[group])
            pl_module.log(f"loss/token_f1_attention_rollout_layer_{self.idx_layer:02d}",token_f1)
        self.batch_results = defaultdict(lambda: [])


    @staticmethod
    def _compute_topk_dg(attentions, sentences_ids):
        pass

    @staticmethod
    def _compute_token_f1(topk_dg, gold_labels):
        pass

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


