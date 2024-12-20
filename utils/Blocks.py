import torch
import torch.nn as nn


class ShortcutMapping(nn.Module):
    def __init__(self, input_size: int, output_size: int, zero_size=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if zero_size is None:
            self.zero_size = input_size // 4 * 3
        else:
            self.zero_size = zero_size
        self.NonZero = nn.Linear(
            self.input_size - self.zero_size, self.output_size, bias=False
        )

    def forward(self, x):
        x_reshape = x.unsqueeze(0) if x.dim() == 1 else x
        Out = self.NonZero(x_reshape[:, self.zero_size :])
        return Out.squeeze(0) if x.dim() == 1 else Out

    def __repr__(self):
        return f"LinearShortcut(in_feature={self.input_size}, out_feature={self.output_size}, zero_feature={self.zero_size})"


class Triangular(nn.Module):
    def __init__(self, SlidingPair=True):
        super().__init__()
        self.SlidingPair = SlidingPair

    def forward(self, x):
        x_reshape = x.unsqueeze(0) if x.dim() == 1 else x
        num_dof = x_reshape.shape[1] // 4
        r, other = (
            x_reshape[:, :num_dof],
            x_reshape[:, num_dof:],
        )
        if self.SlidingPair:
            output = torch.concat([r, torch.sin(r), torch.cos(r), other], dim=1)
        else:
            output = torch.concat([torch.sin(r), torch.cos(r), other], dim=1)
        return output.squeeze(0) if x.dim() == 1 else output

    def __repr__(self):
        return f"Triangular(sliding_pair={self.SlidingPair})"


class HadamardProduct(nn.Module):
    def __init__(self, embedding_size, bias=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.weight = nn.Parameter(
            torch.ones(self.embedding_size),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.embedding_size),
                requires_grad=True,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return x * self.weight + self.bias if self.bias is not None else x * self.weight

    def __repr__(self):
        return f"HadamardProduct(in_feature={self.embedding_size}, out_feature={self.embedding_size}, bias={self.bias is not None})"


class ResLinear(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = output_size if hidden_size is None else hidden_size
        self.output_size = output_size
        self.Linear_1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.Linear_2 = nn.Linear(hidden_size, output_size, bias=bias)
        self.Shortcut = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.Linear_2(torch.relu(self.Linear_1(x))) + self.Shortcut(x)

    def __repr__(self):
        return f"ResLinear(in_feature={self.input_size}, out_feature={self.output_size}, bias={self.Linear_1.bias is not None})"
