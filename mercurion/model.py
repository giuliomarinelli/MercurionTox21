from torch import nn

class MercurionMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 128], output_dim=12, dropout=0.3):
        super(MercurionMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        return self.model(x)
