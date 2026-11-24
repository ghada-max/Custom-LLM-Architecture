""" config file is the singe source of truth for every model dimension """

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768 #embedding dim width
    max_seq_len: int =1024
    dropout: float=0.1
    qkv_bias: bool=False
    n_heads: int=12
    d_ff: int=3072
    n_layers: int = 12 # Number of Transformer layers
    @property
    def d_k(self):
        return self.d_model//self.n_heads