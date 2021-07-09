"""
This file builds Tiled Squeeze-and-Excitation(TSE) from paper:
<STiled Squeeze-and-Excite: Channel Attention With Local Spatial Context> --> https://arxiv.org/abs/2107.02145

Created by Kunhong Yu
Date: 2021/07/06
"""
import torch as t

def weights_init(layer):
	"""
	weights initialization
	Args :
		--layer: one layer instance
	"""
	if isinstance(layer, t.nn.Linear) or isinstance(layer, t.nn.BatchNorm1d):
		t.nn.init.normal_(layer.weight, 0.0, 0.02) # we use 0.02 as initial value
		t.nn.init.constant_(layer.bias, 0.0)

class TSE(t.nn.Module):
	"""Define TSE operation"""
	"""According to the paper, simple TSE can be implemented by 
	several 1x1 conv followed by a average pooling with kernel size and stride,
	which is simple and effective to verify and to do parameter sharing
	In this implementation, column and row pooling kernel sizes are shared!
	"""

	def __init__(self, num_channels : int, attn_ratio : float, pool_kernel = 7):
		"""
		Args :
			--num_channels: # of input channels
			--attn_ratio: hidden size ratio
			--pool_kernel: pooling kernel size, default best is 7 according to paper
		"""
		super().__init__()

		self.num_channels = num_channels

		self.sigmoid = t.nn.Sigmoid()

		self.avg_pool = t.nn.AvgPool2d(kernel_size = pool_kernel, stride = pool_kernel, ceil_mode = True)

		self.tse = t.nn.Sequential(
			t.nn.Conv2d(self.num_channels, int(self.num_channels * attn_ratio), kernel_size = 1, stride = 1),
			t.nn.BatchNorm2d(int(self.num_channels * attn_ratio)),
			t.nn.ReLU(inplace = True),

			t.nn.Conv2d(int(self.num_channels * attn_ratio), self.num_channels, kernel_size = 1, stride = 1),
			t.nn.Sigmoid()
		)
		self.kernel_size = pool_kernel

	def forward(self, x):
		"""x has shape [m, C, H, W]"""
		_, C, H, W = x.size()
		# 1. TSE
		y = self.tse(self.avg_pool(x))

		# 2. Re-calibrated
		y = t.repeat_interleave(y, self.kernel_size, dim = -2)[:, :, :H, :]
		y = t.repeat_interleave(y, self.kernel_size, dim = -1)[:, :, :, :W]

		return x * y

# unit test
if __name__ == '__main__':
	tse = TSE(1024, 0.5, 7)
	print(tse)
