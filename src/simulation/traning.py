import pfl
from pfl.algorithm.federated_averaging import FederatedAveraging


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model import LEXGNN
from src.model.pfl_wrapper import WrapperModel

from src.utils.graph_dataset_loader import load_data


