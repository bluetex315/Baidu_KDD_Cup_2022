import torch
import torch.nn.functional as F
import torch.nn as nn

class BaselineFCModel(nn.Module):
    """
    Desc:
        A simple FC model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineFCModel, self).__init__()
        self.in_features = int(settings["in_var"] * settings["input_len"] / settings["step_size"])
        self.out_features = int(settings["output_len"] * settings["out_var"])
        print("model line21: flattened features", self.in_features)
        self.hid1 = nn.Linear(in_features=self.in_features, out_features=self.in_features*2)
        self.hid2 = nn.Linear(self.in_features*2, 256)
        self.projection = nn.Linear(in_features=256, out_features=self.out_features)

        # print("model line26", self.in_features, self.out_features)

    def forward(self, x):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        output = self.projection(x)

        return output # [Batch, *]

