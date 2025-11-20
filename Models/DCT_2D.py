import torch
import torch.nn as nn
import math


class DCT2D(nn.Module):
    def __init__(self, height, width):
        super(DCT2D, self).__init__()
        self.dct_x = DCT2DSpatialTransformLayer_x(width)
        self.dct_y = DCT2DSpatialTransformLayer_y(height)

    def forward(self, x):
        # Apply DCT along x and y directions
        y = self.dct_x(x)
        y = self.dct_y(y)
        # Split into high and low frequencies
        high, low = self.split_high_low(y)
        return high, low

    def split_high_low(self, dct_coeffs):
        """
        Split DCT coefficients into high and low frequency components.
        Assume the lower half represents low frequencies, and the upper half high frequencies.
        """
        half_h = dct_coeffs.shape[2] // 2
        half_w = dct_coeffs.shape[3] // 2
        low = dct_coeffs[:, :, :half_h, :half_w]
        high = dct_coeffs[:, :, half_h:, half_w:]
        return high, low


class DCT2DSpatialTransformLayer_x(nn.Module):
    def __init__(self, width):
        super(DCT2DSpatialTransformLayer_x, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(width))

    def get_dct_filter(self, width):
        dct_filter = torch.zeros(width, width)
        for v in range(width):
            for j in range(width):
                DCT_base_x = math.cos(math.pi * (0.5 + j) * v / width) / math.sqrt(width)
                if v != 0:
                    DCT_base_x = DCT_base_x * math.sqrt(2)
                dct_filter[v, j] = DCT_base_x
        return dct_filter

    def forward(self, x):
        dct_components = []
        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, 1, x.shape[3]).expand_as(x)
            dct_components.append(dct_component.sum(3).unsqueeze(3))
        result = torch.concat(dct_components, dim=3)
        return result


class DCT2DSpatialTransformLayer_y(nn.Module):
    def __init__(self, height):
        super(DCT2DSpatialTransformLayer_y, self).__init__()
        self.register_buffer('weight', self.get_dct_filter(height))

    def get_dct_filter(self, height):
        dct_filter = torch.zeros(height, height)
        for k in range(height):
            for i in range(height):
                DCT_base_y = math.cos(math.pi * (0.5 + i) * k / height) / math.sqrt(height)
                if k != 0:
                    DCT_base_y = DCT_base_y * math.sqrt(2)
                dct_filter[k, i] = DCT_base_y
        return dct_filter

    def forward(self, x):
        dct_components = []
        for weight in self.weight.split(1, dim=0):
            dct_component = x * weight.view(1, 1, x.shape[2], 1).expand_as(x)
            dct_components.append(dct_component.sum(2).unsqueeze(2))
        result = torch.concat(dct_components, dim=2)
        return result


if __name__ == '__main__':
    x = torch.randn(1, 256, 48, 48).cuda()
    model = DCT2D(48, 48).cuda()
    high, low = model(x)
    print(high.shape, low.shape)
