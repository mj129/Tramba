import torch
import torch.nn.functional as F
import torch_dct as dct


def rgb2ycbcr(rgb_tensor):
    """
    【torch】
    convert rgb into ycbcr
    input: (B, 3, H, W), 3 means (r, g, b)
    output: (B, 3, H, W), 3 means (y, cb, cr), torch.float32
    注意：如果要显示出来，必须先转换为 torch.uint8
    """
    if len(rgb_tensor.shape) != 4 or rgb_tensor.shape[1] != 3:
        raise ValueError("input image is not a rgb tensor: %s" % str(rgb_tensor.shape))
    rgb_tensor = rgb_tensor.to(torch.float32)

    transform_matrix = torch.tensor([[0.257, 0.564, 0.098],
                                     [-0.148, -0.291, 0.439],
                                     [0.439, -0.368, -0.071]]).to(rgb_tensor.device)

    shift_matrix = torch.tensor([16, 128, 128]).reshape(-1, 1).to(rgb_tensor.device)

    ycbcr_tensor = torch.matmul(transform_matrix, rgb_tensor.flatten(2)) + shift_matrix
    ycbcr_tensor = ycbcr_tensor.reshape(rgb_tensor.shape)
    return ycbcr_tensor


def rgb2gray(rgb):
    """
    【torch】
    convert rgb into gray
    input: (B, 3, H, W), 3 means (r, g, b)
    output: (B, 1, H, W), torch.float32
    注意：output中都是小数，对于正常的灰度图像来说要把值转化为 torch.uint8 类型
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = torch.unsqueeze(gray, 1)
    return gray


def dct_2d(rgb_tensor, P=8):
    """
    Modified according to https://github.com/VisibleShadow/Implementation-of-Detecting-Camouflaged-Object-in-Frequency-Domain/blob/main/train.py
    """
    ycbcr_tensor = rgb2ycbcr(rgb_tensor)
    num_batchsize = ycbcr_tensor.shape[0]
    size = ycbcr_tensor.shape[2]
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, 3, size // P, P, size // P, P).permute(0, 2, 4, 1, 3, 5)
    ycbcr_tensor = dct.dct_2d(ycbcr_tensor, norm='ortho')
    ycbcr_tensor = ycbcr_tensor.reshape(num_batchsize, size // P, size // P, -1).permute(0, 3, 1, 2)
    return ycbcr_tensor
