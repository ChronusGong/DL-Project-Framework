from torch import nn


class TransformerPredictor(nn.Module):
    model: nn.Sequential

    def __init__(self, input_size, hidden_size, output_size, n_head=4, n_encoder_layers=3, *args, **kwargs) -> None:
        super(TransformerPredictor, self).__init__(*args, **kwargs)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, dim_feedforward=hidden_size, nhead=n_head,
                                                   activation="gelu", dropout=0)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        fc = nn.Linear(input_size, output_size)

        self.model = nn.Sequential(
            encoder,
            fc,
        )

    def forward(self, x):
        return self.model(x)

    # def to(self, device):
    #     self.model.to(device)