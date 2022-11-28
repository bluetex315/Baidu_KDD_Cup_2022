import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class BaselineLSTMModel(nn.Module):
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
        super(BaselineLSTMModel, self).__init__()

        self.input_size = int(settings["in_var"])
        self.hidden_size = int(settings["gru_hidden_size"])
        
        self.in_features = int(settings["input_len"] / settings["step_size"] * self.hidden_size)
        self.out_features = int(settings["output_len"] * settings["out_var"])

        self.num_layers = int(settings["gru_layers"])
        self.dropout = settings["dropout"]

        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, batch_first=True, dropout=self.dropout)

        self.bn1 = torch.nn.BatchNorm1d(num_features=144)
        self.bn2 = torch.nn.BatchNorm1d(num_features=144)
        self.projection = nn.Linear(in_features=self.in_features, out_features=self.out_features)

        self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.5, 0.5)
        nn.init.uniform_(self.u_omega, -0.5, 0.5)

    def attention(self, x):       #x:[batch, seq_len, hidden_dim]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)                     #[batch, seq_len, 1]

        context = torch.mul(att_score, x)                  #[batch, seq_len, hidden_dim]
        return context

    def forward(self, x):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x:
        Returns:
            A tensor
        """

        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(1)

        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        output = self.bn1(output)
        att_output = self.attention(output)
        att_output = self.bn2(att_output)
        att_output = torch.flatten(att_output, start_dim=1, end_dim=2)

        final_output = self.projection(att_output)

        return final_output # [Batch, *]
