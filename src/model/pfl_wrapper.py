import torch 
import torch.nn

from pfl.model.pytorch import PyTorchModel
from pfl.internal.ops import get_ops

class WrapperModel(PyTorchModel):

    def __init__(self, model):
        super().__init__()
        self._ops = get_ops()
        self.torch_model = model

    def forward(self, x):
        pt_tensors =  torch.tensor(x, device=torch.device("mps"))
        with torch.no_grad():
            out = self.torch_model(pt_tensors)
        return self._ops.to_numpy(out)

    def get_parameters(self):
        return [p.detach().cpu().numpy() for p in self.torch_model.parameters()]

    def set_parameters(self, parameters):
        with torch.no_grad():
            for p, param in zip(self.torch_model.parameters(), parameters):
                p.data.copy_(torch.tensor(param, device=torch.device("mps")))
