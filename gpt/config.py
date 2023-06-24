from dataclasses import dataclass
from typing import Tuple, Optional , NamedTuple
import torch


@dataclass
class GPTConfig:
    def __init__(self):
        self.vocab_size = 3020
        self.model_dim = 512
        self.num_heads = 8
        self.num_layers = 6
        self.ff_dim = 2048
        self.dropout = 0.1
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.001
        self.num_workers = 4
        self.output_dim = 35
        self.seed = 1
        self.log_interval = 10
        self.save_model = True
        self.save_model_path = "model.pt"
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

    def __str__(self):
        return str(self.__dict__)
    

