from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn

class BaselineGRUModel(nn.Module):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGRUModel, self).__init__()

        self.input_size = int(settings["in_var"])
        self.hidden_size = int(settings["gru_hidden_size"])
        
        self.in_features = int(settings["input_len"] / settings["step_size"] * self.hidden_size)
        self.out_features = int(settings["output_len"] * settings["out_var"])

        self.num_layers = int(settings["gru_layers"])
        self.dropout = settings["dropout"]
        self.gru1 = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, batch_first=True, dropout=self.dropout, 
                                        )
        # self.bn = torch.nn.BatchNorm1d(num_features=settings["input_len"])
        # self.gru2 = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, 
        #                             num_layers=1, batch_first=True, dropout=self.dropout)
        self.projection = nn.Linear(in_features=self.in_features, out_features=self.out_features)

        # print("model line26", self.in_features, self.out_features)

    def forward(self, x):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        #print('model line 38',x.shape)
        #x = x.reshape(x.shape[0], -1)
        #print('model line 45',x.shape)
        # hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(0)
        #print(hidden.shape)
        #x = x.long()
        #x = self.emb(x)
        #print(x.shape)
        x, _ = self.gru1(x)
        # x = self.bn(x)
        #print('52',x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        #print('54',x.shape)
        output = self.projection(x)

        return output # [Batch, *]


