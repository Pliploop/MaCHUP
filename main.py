
from src.models.model import MuMRVQ
from pytorch_lightning.cli import LightningCLI
from src.dataloading.datasets import CustomAudioDataModule
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import yaml

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            if trainer.logger is not None:
                print('========== wandb logging dir ================')
                print(trainer.logger.log_dir)
                print('============ wandb logger save dir =========')
                print(trainer.logger.save_dir)
                
                config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                with open(self.config_filename, "r") as config_file:
                    config = yaml.load(config_file, Loader=yaml.FullLoader)
                    trainer.logger.experiment.config.update(config)

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.encoder.init_args.n_codebooks", "model.decoder.init_args.n_codebooks")
        parser.link_arguments("model.encoder.init_args.emebdding_size", "model.decoder.init_args.eùbedding_size")
        parser.link_arguments("model.encoder.init_args.card", "model.decoder.init_args.card")
        parser.link_arguments("model.encoder.init_args.embedding_behaviour", "model.decoder.init_args.embedding_behaviour")
        parser.link_arguments("model.encoder.init_args.sequence_len", "model.decoder.init_args.sequence_len")
        parser.link_arguments("data.target_sample_rate","model.encodec.init_args.sample_rate")
        parser.add_argument("--log", default=False)

if __name__ == "__main__":
    
    cli = MyLightningCLI(model_class=MuMRVQ, datamodule_class=CustomAudioDataModule, seed_everything_default = 123, run = False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite":True})
    
    
    cli.instantiate_classes()

    
    if cli.config.log:
        logger = WandbLogger(project="MuMRVQ", log_model=True)
    else:
        logger = None
        
    cli.trainer.logger = logger
    
    cli.trainer.fit(model=cli.model,datamodule=cli.datamodule)

    ## wandb logger here + set callbacks for checkpointing and config saving before running fit with the wandb logger experiment name.

    