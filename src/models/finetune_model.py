
from pytorch_lightning import LightningModule
from torch import nn
import torch
from src.models.model import MaCHUP
from torchmetrics import Accuracy, Precision, Recall, AUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from pytorch_lightning.cli import OptimizerCallable
from src.evaluation.metrics import *


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output[0]

    def close(self):
        self.hook.remove()

class Head(nn.Module):
    
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        
        self.linear = nn.Linear(d_model, n_classes)
        self.linear_2 = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x = self.linear_2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear(x)
        return x

class MaCHUPFinetune(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        encodec: nn.Module,
        optimizer: OptimizerCallable = None,
        pattern="delay",
        n_codebooks=4,
        sequence_len=1024,
        pattern_special_token=1024,
        mask_special_token=1025,
        pad_special_token=1026,
        mask_before=False,
        debug=False,
        masked_loss_ratio=0.9,
        masked_objective=True,
        window_size=50,
        adapt_sequence_len=True,
        contrastive_to_masked_ratio=0.5,
        global_vs_local_contrastive_loss_ratio=0.1,
        global_class_vs_average_contrastive_loss_ratio=1,
        contrastive_temperature = 0.5,
        reduce_lr_monitor = 'overall_loss',
        only_global_contrastive  = False,
        checkpoint_path: str = None,
        task = "GTZAN",
        target_layer = "transformer_encoder",
        freeze_encoder = True,
        use_global_representation = "class",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.machup = MaCHUP(
            encoder = encoder,
            decoder= decoder,
            encodec= encodec,
            debug=debug,
            adapt_sequence_len=adapt_sequence_len)
            
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
            print("Encoder successfully frozen")
            
        
        if self.task == "GTZAN":
            self.loss_fn = nn.CrossEntropyLoss()
            
        
        self.hook = Hook(self.machup._modules.get(target_layer))
        self.head = Head(self.machup.d_model, 10)
        self.target_layer = target_layer
            
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
            class_encoded = torch.mean(encoded, dim=1)
        else:
            class_encoded = encoded[:,0,:]
        head_out = self.head(class_encoded)
        return encoded,head_out

    # defining training step, one per task
    def training_step(self, batch, batch_idx):
        if self.task == "GTZAN":
            return self.GTZAN_train_step(batch, batch_idx)
        
    
    def validation_step(self, batch, batch_idx):
        if self.task == "GTZAN":
            return self.GTZAN_validation_step(batch, batch_idx)
        
        
        
    
        
    def GTZAN_train_step(self, batch, batch_idx):
        
        x = batch
        y = batch['label']
        
        
        encoded, head_out = self.forward(x)
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
        
        #log confusion matrix as an image with wandb
        # softmax the head_out for probabilities
        head_out = torch.nn.functional.softmax(head_out, dim=1)
        
        return loss
    
    def GTZAN_validation_step(self,batch,batch_idx):
        x = batch['wav']
        y = batch['label']
        
        encoded, head_out = self.forward(x)
        
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
        
        #log confusion matrix as an image with wandb
        # softmax the head_out for probabilities
        head_out = torch.nn.functional.softmax(head_out, dim=1)
        
        return loss
    

    