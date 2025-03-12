import os
import random
import numpy as np

from pfl.aggregate import simulate
from pfl.aggregate.simulate import SimulatedBackend
from numpy import load


from pfl.aggregate.simulate import SimulatedBackend	
from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import CentralEvaluationCallback
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams
from pfl.data.dataset import Dataset

from pfl.model.pytorch import PyTorchModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model import LEXGNN
from src.model.pfl_wrapper import WrapperModel
from src.simulation.aggregation import MaxMagAggregator

from data.federated_dataset import load_federated_dataset

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.mps.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 

def fed_train():
    set_random_seed(22)
    
    cohort_size = 10
    central_num_iterations = 5


    train_dataset, val_dataset, test_dataset, n_input, train_loader, valid_loader, test_loader, labels = load_federated_dataset()

    model = LEXGNN(
        in_dim=n_input,
        n_class=2,
        hidden_dim=64,
        n_layer=2,
        num_heads=4,
        dropout=0.0
    )

    simulate_backend = SimulatedBackend(
        training_data=train_dataset,
        val_data=val_dataset,
        aggregator=MaxMagAggregator()
    )


    model_train_params = NNTrainHyperParams(
        local_learning_rate=0.005,
        local_num_epochs=2,
        local_batch_size=3,
    )

    model_eval_params = NNEvalHyperParams(
        local_batch_size=2
    )

    central_data = Dataset(
        raw_data=[n_input, labels]
    )
    callbacks = [
        CentralEvaluationCallback(
            dataset=central_data,
            model_eval_params=model_eval_params,
            frequency=4,
        )
    ]
    torch_model = PyTorchModel(
        model=model,
        local_optimizer_create=torch.optim.SGD,
        central_optimizer=torch.optim.SGD(model_train_params, 1.0)
    )
    algorithm_params = NNAlgorithmParams(
        central_num_iterations=central_num_iterations,
        evaluation_frequency=4,
        train_cohort_size=cohort_size,
        val_cohort_size=0,
    )

    model = FederatedAveraging().run(
        algorithm_params=algorithm_params,
        backend=simulate_backend,
        model=torch_model,
        model_train_params=model_train_params,
        model_eval_params=model_eval_params,
        callbacks=callbacks,
    )


