
from src.models.finetune_model import MaCHUPFinetune
from pytorch_lightning.cli import LightningCLI
from src.dataloading.finetuning_datasets import GTZANFinetuneDataset
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataloading.finetuning_datasets import FineTuneDataModule
import yaml
import os
import shutil


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.logger is not None:
            experiment_name = trainer.logger.experiment.name
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            with open(self.config_filename, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                trainer.logger.experiment.config.update(config)
                
            with open(os.path.join(os.path.join(self.config['ckpt_path'], experiment_name+f'_finetune_{config["model"]["task"]}'), "config.yaml"), 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

            
            trainer.logger.experiment.name = experiment_name+f'_finetune_{config["model"]["task"]}'


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "model.encoder.init_args.n_codebooks", "model.decoder.init_args.n_codebooks")
        parser.link_arguments("model.encoder.init_args.emebdding_size",
                              "model.decoder.init_args.eùbedding_size")
        parser.link_arguments("model.encoder.init_args.card",
                              "model.decoder.init_args.card")
        parser.link_arguments("model.encoder.init_args.embedding_behaviour",
                              "model.decoder.init_args.embedding_behaviour")
        parser.link_arguments("model.sequence_len",
                              "model.encoder.init_args.sequence_len")
        parser.link_arguments("model.sequence_len",
                              "model.decoder.init_args.sequence_len")
        parser.link_arguments("model.task",
                              "data.task")
        parser.link_arguments("data.target_sample_rate",
                              "model.encodec.init_args.sample_rate")
        parser.add_argument("--log", default=False)
        parser.add_argument("--log_model", default=False)
        parser.add_argument("--ckpt_path", default="MuMRVQ_checkpoints")


if __name__ == "__main__":
    

    cli = MyLightningCLI(model_class=MaCHUPFinetune, datamodule_class=FineTuneDataModule, seed_everything_default=123,
                         run=False, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True},)

    cli.instantiate_classes()

    if cli.config.log:
        logger = WandbLogger(project="MuMRVQ")
        experiment_name = logger.experiment.name+f"_finetune_{cli.config['model']['task']}"
        ckpt_path = cli.config.ckpt_path
    else:
        logger = None

    cli.trainer.logger = logger

    try:
        if not os.path.exists(os.path.join(ckpt_path, experiment_name)):
            os.makedirs(os.path.join(ckpt_path, experiment_name))
    except:
        pass
    
    print(cli.config)

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
