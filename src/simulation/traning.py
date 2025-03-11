from numpy import load
import pfl

from pfl.aggregate import simulate
from pfl.aggregate.simulate import SimulatedBackend	
from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import CentralEvaluationCallback
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model import LEXGNN
from src.model.pfl_wrapper import WrapperModel
from src.model.mode_trainer import train

from data.federated_dataset import load_federated_dataset

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 