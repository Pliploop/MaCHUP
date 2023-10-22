from torch import nn

from src.models.Encodec import Encodec


class MuMRVQ (nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encodec = Encodec()
    
    def forward(self,x):
        return self.encodec(x)