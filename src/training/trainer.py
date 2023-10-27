import pytorch_lightning as pl

from pytorch_lightning import Trainer

from wandb import WandbLogger

from src.models.model import MuMRVQ



class TrainerWrap():

    def __init__(self, accelerator='auto', no_log=False, resume=False, resume_id=None, debug = False, checkpoint_path = None, devices = 1, max_epochs = 100, accumulate_grad_batches = None) -> None:
        
        self.resume_id = resume_id
        self.devices = devices
        self.checkpoint_path = checkpoint_path
        self.resume_run = resume
        self.max_epochs = max_epochs
        self.accumulate_grad_batches = accumulate_grad_batches

        

        if not no_log:
            logger = WandbLogger(project="MuMRVQ", log_model=True)
        else:
            logger = None

        if logger is not None:
            self.trainer = Trainer(
                accelerator=accelerator,
                max_epochs=self.config.max_epochs,
                devices=self.devices,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                callbacks=None,
                logger=logger,
                gradient_clip_val=0.5,
                log_every_n_steps=2
            )
        else:
            self.trainer = Trainer(
                accelerator=accelerator,
                max_epochs=self.config.max_epochs,
                devices=self.devices,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                callbacks=None,
                gradient_clip_val=0.5,
                log_every_n_steps=2
            )

        ### callbacks, all that

        def get_trainer(self):
            return self.trainer

        
