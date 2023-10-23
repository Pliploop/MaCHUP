from src.models.Encodec import Encodec
from src.models.Transformer.Encoder import TransformerEncoder
from src.models.model import MuMRVQ
from pytorch_lightning.cli import LightningCLI


from src.models.Encodec import Encodec
from src.models.Transformer.Encoder import TransformerEncoder



if __name__ == "__main__":
    
    cli = LightningCLI(MuMRVQ, seed_everything_default = 123)
    