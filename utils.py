"""
Define some utilities

Created by Kunhong Yu
Date: 2021/07/06
"""
import torch as t
from torch.nn import functional as F
from tse import TSE

####################################
#         Define utilities         #
####################################
def _conv_layer(in_channels : int,
                out_channels : int) -> t.nn.Sequential:
    """Define conv layer
    Args :
        --in_channels: input channels
        --out_channels: output channels
    return :
        --conv layer
    """
    conv_layer = t.nn.Sequential(
        t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = 3, stride = 1, padding = 1),
        t.nn.BatchNorm2d(out_channels),
        t.nn.ReLU(inplace = True)
    )

    return conv_layer


def vgg_block(in_channels : int,
              out_channels : int,
              repeat : int) -> list:
    """Define VGG block
    Args :
        --in_channels: input channels
        --out_channels: output channels
        --repeat
    return :
        --block
    """
    block = [
        _conv_layer(in_channels = in_channels if i == 0 else out_channels,
                    out_channels = out_channels)
        for i in range(repeat)
    ]

    return block

####################################
#            Define SE             #
####################################
class SEAttention(t.nn.Module):
    """Define SE operation"""

    def __init__(self, num_channels : int, attn_ratio : float):
        """
        Args :
             --num_channels: # of input channels
             --attn_ratio: hidden size ratio
        """
        super(SEAttention, self).__init__()

        self.num_channels = num_channels
        self.hidden_size = int(attn_ratio * self.num_channels)

        # 1. Trunk, we use T(x) = x
        # 2. SE attention
        self.SE = t.nn.Sequential(
            t.nn.Linear(self.num_channels, self.hidden_size),
            t.nn.BatchNorm1d(self.hidden_size),
            t.nn.ReLU(inplace = True),

            t.nn.Linear(self.hidden_size, self.num_channels),
            t.nn.BatchNorm1d(self.num_channels),
            t.nn.Sigmoid()
        )

    def forward(self, x):
        # 1. T(x)
        Tx = x
        # 2. SE attention
        x = F.adaptive_avg_pool2d(x, (1, 1)) # global average pooling
        x = x.squeeze()
        Ax = self.SE(x)

        # 3. output
        x = Tx * t.unsqueeze(t.unsqueeze(Ax, dim = -1), dim = -1) # broadcasting

        return x

def get_attention(channels : int, attn_ratio = 0.5, pool_kernel = 7, method = 'se'):
    """Get attention method
    Args :
        --channels: number of input channels
        --attn_ratio: attention ratio, default is 0.5
        --pool_kernel: 7 as default according to paper
        --method: 'se' or 'tse', default is 'se'
    return :
        --attn: attention method
    """
    if method == 'se':
        attn = SEAttention(num_channels = channels,
                           attn_ratio = attn_ratio)
    elif method == 'tse':
        attn = TSE(num_channels = channels, attn_ratio = attn_ratio, pool_kernel = pool_kernel)

    else:
        raise Exception('No other attentions!')

    return attn

####################################
#          Define VGG16            #
####################################
class VGG16(t.nn.Module):
    """Define VGG16-style model"""

    def __init__(self, attn_method = 'none', attn_ratio = 0.5):
        """
        Args :
            --attn_method: 'none'/'se'/'tse'
            --attn_ratio: hidden size ratio, default is 0.5
            --pre_attn: None or [att1, attn2, ...]
        """
        super(VGG16, self).__init__()

        self.attn_method = attn_method

        self.layer1 = t.nn.Sequential(*vgg_block(in_channels = 3,
                                                 out_channels = 64,
                                                 repeat = 2))

        if self.attn_method != 'none':
            self.attn1 = get_attention(channels = 64, attn_ratio = attn_ratio, pool_kernel = 2, method = self.attn_method)

        self.layer2 = t.nn.Sequential(*vgg_block(in_channels = 64,
                                                 out_channels = 128,
                                                 repeat = 2))

        if self.attn_method != 'none':
            self.attn2 = get_attention(channels = 128, attn_ratio = attn_ratio, pool_kernel = 2, method = self.attn_method)

        self.layer3 = t.nn.Sequential(*vgg_block(in_channels = 128,
                                                 out_channels = 256,
                                                 repeat = 3))

        if self.attn_method != 'none':
            self.attn3 = get_attention(channels = 256, attn_ratio = attn_ratio, pool_kernel = 2, method = self.attn_method)

        self.layer4 = t.nn.Sequential(*vgg_block(in_channels = 256,
                                                 out_channels = 512,
                                                 repeat = 3))

        if self.attn_method != 'none':
            self.attn4 = get_attention(channels = 512, attn_ratio = attn_ratio, pool_kernel = 2, method = self.attn_method)

        self.fc = t.nn.Sequential(  # unlike original VGG16, I reduce some fc
            # parameters to fit my 2070 device
            t.nn.Linear(512, 256),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(256, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x, pre_attn = None):
        """pre_attn is None or [attn1, attn2, ...]"""
        assert pre_attn is None or len(pre_attn) == 4

        x1 = self.layer1(x)
        if self.attn_method != 'none':
            if pre_attn is None:
                x1 = self.attn1(x1)
            else:
                x1 = pre_attn[0](x1)
        x1 = self.max_pool(x1)

        x2 = self.layer2(x1)
        if self.attn_method != 'none':
            if pre_attn is None:
                x2 = self.attn2(x2)
            else:
                x2 = pre_attn[1](x2)
        x2 = self.max_pool(x2)

        x3 = self.layer3(x2)
        if self.attn_method != 'none':
            if pre_attn is None:
                x3 = self.attn3(x3)
            else:
                x3 = pre_attn[2](x3)
        x3 = self.max_pool(x3)

        x4 = self.layer4(x3)
        if self.attn_method != 'none':
            if pre_attn is None:
                x4 = self.attn4(x4)
            else:
                x4 = pre_attn[3](x4)
        x4 = self.max_pool(x4)

        x = F.adaptive_avg_pool2d(x4, (1, 1))
        x = x.squeeze()
        x = self.fc(x)

        return x