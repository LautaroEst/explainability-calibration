from .evaluation import accuracy
import lightning.pytorch as pl
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup


class SequenceClassificationModel(pl.LightningModule):

    def __init__(
        self, 
        base_model, 
        num_labels,
        learning_rate,
        weight_decay,
        warmup_proportion,
        num_train_optimization_steps
    ):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            local_files_only=True
        )
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion
        self.num_train_optimization_steps = num_train_optimization_steps

    def forward(self, batch):
        outputs = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            output_attentions=True
        )
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def configure_optimizers(self):

        # Optimizer:
        param_optimizer = self.base_model.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": self.weight_decay,
        },{
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "lr": self.learning_rate,
        }
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        # Scheduler:
        warmup_steps = int(self.warmup_proportion * self.num_train_optimization_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        output = self(train_batch)
        if self.num_labels == 1:
            labels = train_batch["label"].unsqueeze(dim=-1).type(output["logits"].dtype)
            loss = F.binary_cross_entropy_with_logits(output["logits"],labels)
        else:
            loss = F.cross_entropy(output["logits"],train_batch["label"])
        self.log("train_loss", loss, batch_size=len(train_batch))

        acc = accuracy(output["logits"],train_batch["label"])
        self.log("train_acc", acc, on_epoch=True, batch_size=len(train_batch))
        return loss
    
    def validation_step(self, validation_batch, batch_idx):
        output = self(validation_batch)
        if self.num_labels == 1:
            labels = validation_batch["label"].unsqueeze(dim=-1).type(output["logits"].dtype)
            loss = F.binary_cross_entropy_with_logits(output["logits"],labels)
        else:
            loss = F.cross_entropy(output["logits"],validation_batch["label"])
        self.log("val_loss", loss, batch_size=len(validation_batch))
        
        acc = accuracy(output["logits"],validation_batch["label"])
        self.log("val_acc", acc, on_epoch=True, batch_size=len(validation_batch))
        return loss
    
    def backward(self, loss):
        loss.backward()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

    