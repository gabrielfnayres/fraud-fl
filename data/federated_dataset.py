import torch

from sklearn.datasets import make_classification
import numpy as np

from pfl.data import get_data_sampler, ArtificialFederatedDataset

from src.utils.graph_dataset_loader import load_data

class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, user_id_to_data):
        self._user_id_to_data = user_id_to_data

    def __getitem__(self, i):
        return [torch.as_tensor(x) for x in self._user_id_to_data[i]]

    def __len__(self):
        return len(self._user_id_to_data)

    
def load_federated_dataset():

    features, train_loader, valid_loader, test_loader, labels = load_data(
        data_name='amazon',
        seed=2,
        train_ratio=0.4,
        test_ratio=0.6,
        n_layer=2,
        batch_size=1024
    )


    num_classes = len(torch.unique(labels))
    dirichlet_alpha = [0.1] * num_classes

    data_sampler = get_data_sampler(
        sample_type='dirichlet',
        labels=labels,
        alpha=dirichlet_alpha

    )

    train_sample_len = lambda: len(train_loader)

    federated_train_dataset = ArtificialFederatedDataset.from_slices(
        data=[features, labels],
        data_sampler=data_sampler,
        sample_dataset_len=train_sample_len
    )

    valid_sample_len = lambda : len(valid_loader)   

    federated_val_dataset = ArtificialFederatedDataset.from_slices(
        data=[features, labels],
        data_sampler=data_sampler,
        sample_dataset_len=valid_sample_len
    )

    test_sample_len = lambda : len(test_loader)

    federated_test_dataset = ArtificialFederatedDataset.from_slices(
        data=[features, labels],
        data_sampler=data_sampler,
        sample_dataset_len=test_sample_len
    )

    return federated_train_dataset, federated_val_dataset, federated_test_dataset, features, train_loader, valid_loader, test_loader, labels