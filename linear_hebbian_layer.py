import math

import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter

class LinearHebbianLayer(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        # >>> m = nn.Linear(20, 30)
        # >>> input = torch.randn(128, 20)
        # >>> output = m(input)
        # >>> print(output.size())
        torch.Size([128, 30])
    """
    # __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    # weight: Tensor

    def __init__(self, in_features: int, out_features: int, activation_fn, bias: bool = False ) -> None:
        super(LinearHebbianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.h = Parameter(torch.randn((in_features, out_features, 5)))
        self.activation_fn = activation_fn
        # self.reset_parameters()

    # def reset_parameters(self) -> None:
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None_LunarlanderAdam:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)


    def forward(self, pre):
        pre.sg((self.in_features,))
        post = self.activation_fn(pre @ self.weight).sg((self.out_features,))
        self.update(pre, post)
        return post

    def update(self, pre, post):
        pre.sg((self.in_features,))
        post.sg((self.out_features,))

        eta, A, B, C, D = [v.squeeze().sg((self.in_features, self.out_features)) for v in self.h.split(1, -1)]
        self.weight += eta * (
                A * (pre[:, None] @ post[None, :]).sg((self.in_features, self.out_features)) +
                (B * pre[:, None]).sg((self.in_features, self.out_features)) +
                (C * post[None, :]).sg((self.in_features, self.out_features)) +
                D
        )

    # def extra_repr(self) -> str:
    #     return 'in_features={}, out_features={}, bias={}'.format(
    #         self.in_features, self.out_features, self.bias is not None_LunarlanderAdam
    #     )
