
from pytorch_lightning import LightningModule
from torch import nn
import torch
from src.models.model import MaCHUP
from torchmetrics import Accuracy, Precision, Recall, AUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from pytorch_lightning.cli import OptimizerCallable
from src.evaluation.metrics import *
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torch.nn.functional import sigmoid



class MLPHead(nn.Module):
    
    def __init__(self, d_model, n_classes,dropout = 0):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        
        self.linear = nn.Linear(d_model, n_classes)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear_2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear(x)
        return x
    

class LinearHead(nn.Module):
        
        def __init__(self, d_model, n_classes, dropout = 0):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
            
            self.linear = nn.Linear(d_model, n_classes)
            
            
        def forward(self, x):
            x = self.linear(x)
            return x

class MaCHUPFinetune(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encodec: nn.Module,
        optimizer: OptimizerCallable = None,
        sequence_len= 1024,
        debug=False,
        adapt_sequence_len=True,
        checkpoint_path: str = None,
        task = "GTZAN",
        n_classes=10,
        mlp_head = False,
        freeze_encoder = False,
        use_global_representation = "avg",
        use_embeddings = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.machup = MaCHUP(
            encoder = encoder,
            decoder= decoder,
            encodec= encodec,
            debug=debug,
            adapt_sequence_len=adapt_sequence_len,
            use_embeddings=use_embeddings)
            
        self.optimizer = optimizer
        self.freeze_encoder = freeze_encoder
        self.use_global_representation = use_global_representation
        
        if checkpoint_path:
            self.load_machup_weights(checkpoint_path)
        self.task = task
        
        
        if self.freeze_encoder: 
            self.machup.freeze()
            for param in self.machup.parameters():
                assert param.requires_grad == False
            
            self.machup.eval()
            print("Encoder successfully frozen")
            
        
        if self.task == "GTZAN":
            self.loss_fn = nn.CrossEntropyLoss()
        if self.task == "MTGTop50Tags":
            self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.n_classes = n_classes
        
        if mlp_head:
            self.head = MLPHead(d_model = self.machup.d_model, n_classes = self.n_classes)
        else:
            self.head = LinearHead(d_model = self.machup.d_model, n_classes = self.n_classes)
        
            
        
            
        self.auroc = MultilabelAUROC(num_labels=self.n_classes)
        self.ap = MultilabelAveragePrecision(num_labels=self.n_classes)
        
            
    def load_machup_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.machup.load_state_dict(checkpoint['state_dict'], strict=False)
        # print a message saying that weights were correctly loaded
        print(f"Weights loaded from checkpoint {checkpoint_path}")
        
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer

    def forward(self, x):
        encoded = self.machup.finetune_forward(x)['encoded']
        # hook_out = self.hook.output
        if self.use_global_representation == "avg":
            class_encoded = torch.mean(encoded[:,1:,:], dim=1)
        else:
            class_encoded = encoded[:,0,:]
        head_out = self.head(class_encoded)
        return encoded,head_out

    # defining training step, one per task
    def training_step(self, batch, batch_idx):
        if self.task == "GTZAN":
            return self.GTZAN_train_step(batch, batch_idx)
        if self.task == "MTGTop50Tags":
            return self.mtg_top50_train_step(batch, batch_idx)
        
    
    def validation_step(self, batch, batch_idx):
        if self.task == "GTZAN":
            return self.GTZAN_validation_step(batch, batch_idx)
        if self.task == "MTGTop50Tags":
            return self.mtg_top50_validation_step(batch, batch_idx)
    
        
    
        
    def GTZAN_train_step(self, batch, batch_idx):
        
        x = batch
        y = batch['label']
        
        
        encoded, head_out = self(x)
        loss = self.loss_fn(head_out, y)
        
        
        # get all the metrics here
        acc = accuracy_score_multiclass(y, head_out)
        prec = precision_score_multiclass(y, head_out)
        rec = recall_score_multiclass(y, head_out)
        # auroc = roc_auc_score_multiclass(y, head_out)
        
        # log all the metrics here
        self.log('train_crossentropy', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=  True)
        self.log('train_precision', prec, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_recall', rec, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log('train_auroc', auroc, on_step=True, on_epoch=True, prog_bar=False)
        
        
        return loss
    
    def GTZAN_validation_step(self,batch,batch_idx):
        x = batch['wav']
        y = batch['label']
        
        encoded, head_out = self(x)
        
        loss = self.loss_fn(head_out, y)
        
        head_out = head_out.cpu()
        y = y.cpu()
        
        # get all the metrics here
        acc = accuracy_score_multiclass(y, head_out)
        prec = precision_score_multiclass(y, head_out)
        rec = recall_score_multiclass(y, head_out)
        # auroc = roc_auc_score_multiclass(y, head_out)
        
        
        # log all the metrics here
        self.log('val_crossentropy', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_precision', prec, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_recall', rec, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log('val_auroc', auroc, on_step=True, on_epoch=True, prog_bar=False)
        
        
        return loss
    

    def mtg_top50_train_step(self,batch, batch_idx):
        
        x = batch['wav']
        y = batch['label']
        
        
        encoded, head_out = self(x)
        loss = self.loss_fn(head_out, y)
        
        
        # get all the metrics here
        auroc = self.auroc(preds = sigmoid(head_out), target = y.int())
        ap = self.ap(preds = sigmoid(head_out), target =  y.int())
        
        
        # log all the metrics here
        self.log('train_crossentropy', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_auroc', auroc, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_ap', ap, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss
        
        
    def mtg_top50_val_step(self,batch,batch_idx):
        
        x = batch['wav']
        y = batch['label']
        
        
        encoded, head_out = self(x)
        loss = self.loss_fn(head_out, y)
        
        
        auroc = self.auroc(preds = head_out, target = y.int())
        ap = self.ap(preds = head_out, target = y.int())
        
        
        # log all the metrics here
        self.log('val_crossentropy', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_auroc', auroc, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_ap', ap, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        return loss