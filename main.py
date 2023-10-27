
from src.models.model import MuMRVQ
from pytorch_lightning.cli import LightningCLI
from src.dataloading.datasets import CustomAudioDataModule
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            if trainer.logger is not None:
                trainer.logger.log_hyperparams({"config": config})
                trainer.logger.experiment.config.update(config)

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.encoder.init_args.n_codebooks", "model.decoder.init_args.n_codebooks")
        parser.link_arguments("model.encoder.init_args.emebdding_size", "model.decoder.init_args.eùbedding_size")
        parser.link_arguments("model.encoder.init_args.card", "model.decoder.init_args.card")
        parser.link_arguments("model.encoder.init_args.embedding_behaviour", "model.decoder.init_args.embedding_behaviour")
        parser.link_arguments("model.encoder.init_args.sequence_len", "model.decoder.init_args.sequence_len")
        parser.link_arguments("data.target_sample_rate","model.encodec.init_args.sample_rate")

if __name__ == "__main__":
    
    cli = MyLightningCLI(model_class=MuMRVQ, datamodule_class=CustomAudioDataModule, seed_everything_default = 123, run = False, save_config_callback=LoggerSaveConfigCallback)
    
    cli.instantiate_classes()

    model = cli.model
    datamodule = cli.datamodule
    trainer = cli.trainer

    

    print(model)
    print(datamodule)
    print(trainer)