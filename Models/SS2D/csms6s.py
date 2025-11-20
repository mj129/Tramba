from Models.SS2D.Hilbert import get_hilber_spacefill
from Models.SS2D.Dilation import generate_dilation_indices, get_dilation_scan, get_re_dilation_scan
from Models.SS2D.Window import generate_window_indices, get_window_scan, get_re_window_scan
from Models.SS2D.Spiral import spiral_scan
from Models.SS2D.SpiralLine import *
# from Models.selective_scan.SpiralLine_v2 import *
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


sizes = [12, 24, 48, 96]
dilation_rate = [4, 4, 4, 4]  # 2 4 6
dilation_index = {str(size): generate_dilation_indices(size, size, dilation_rate=rate)
                  for size, rate in zip(sizes, dilation_rate)}
print(f"Curren Dilation Rate: {dilation_rate}")

class CrossScan_Dilation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        index = dilation_index[str(H)]
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0:4] = get_dilation_scan(x, index)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        index = dilation_index[str(H)]
        y = get_re_dilation_scan(ys[:, 0:4], (B, C, H, W), index)
        return y.view(B, -1, H, W)


class CrossMerge_Dilation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        index = dilation_index[str(H)]
        ys = ys.view(B, K, D, -1)
        y = get_re_dilation_scan(ys[:, 0:4], (B, D, H, W), index)
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        index = dilation_index[str(H)]
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0:4] = get_dilation_scan(x.view(B, C, H, W), index)
        xs = xs.view(B, 4, C, H, W)
        return xs


#########################################
sizes = [12, 24, 48, 96]
window_sizes = [4, 8, 12, 16]  # 4 8 12 16
window_index = {str(size): generate_window_indices(size, size, window_size=window_size)
                for size, window_size in zip(sizes, window_sizes)}
print(f"Curren Window Size: {window_sizes}")

class CrossScan_Window(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        index = window_index[str(H)]
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0:4] = get_window_scan(x, index)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        index = window_index[str(H)]
        y = get_re_window_scan(ys[:, 0:4], (B, C, H, W), index)
        return y.view(B, -1, H, W)


class CrossMerge_Window(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        index = window_index[str(H)]
        ys = ys.view(B, K, D, -1)
        y = get_re_window_scan(ys[:, 0:4], (B, D, H, W), index)
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        index = window_index[str(H)]
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0:4] = get_window_scan(x.view(B, C, H, W), index)
        xs = xs.view(B, 4, C, H, W)
        return xs


# =========================
# Pre-generate indices for specific sizes
sizes = [12, 24, 48, 96, 7, 14, 28, 56]
spiral_line_index = {str(size): generate_indices(size, size) for size in sizes}


class CrossScan_Line(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        index = spiral_line_index[str(H)]
        xs = x.new_empty((B, 8, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs[:, 4:8] = get_line_scan(x, index)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        index = spiral_line_index[str(H)]
        L = H * W
        y_rc = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rc = y_rc[:, 0] + y_rc[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rc = y_rc.view(B, -1, H, W)
        y_h = get_line_re_scan(ys[:, 4:8], (B, C, H, W), index)
        y = y_h + y_rc
        return y.view(B, -1, H, W)


class CrossMerge_Line(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        index = spiral_line_index[str(H)]
        ys = ys.view(B, K, D, -1)

        y_rc = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y_rc = y_rc[:, 0] + y_rc[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rc = y_rc.view(B, -1, H, W)
        y_h = get_line_re_scan(ys[:, 4:8], (B, D, H, W), index)
        y = y_h + y_rc
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        index = spiral_line_index[str(H)]
        xs = x.new_empty((B, 8, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs[:, 4:8] = get_line_scan(x.view(B, C, H, W), index)
        xs = xs.view(B, 8, C, H, W)
        return xs


class CrossScan_Line_4direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        index = spiral_line_index[str(H)]
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0:4] = get_line_scan(x, index)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        index = spiral_line_index[str(H)]
        L = H * W
        y = get_line_re_scan(ys[:, 0:4], (B, C, H, W), index)
        return y.view(B, -1, H, W)


class CrossMerge_Line_4direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        index = spiral_line_index[str(H)]
        ys = ys.view(B, K, D, -1)

        y = get_line_re_scan(ys[:, 0:4], (B, D, H, W), index)
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        index = spiral_line_index[str(H)]
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0:4] = get_line_scan(x.view(B, C, H, W), index)
        xs = xs.view(B, 4, C, H, W)
        return xs


# =========================
spiral_index_12, spiral_index_12_re = spiral_scan(12, 12)
spiral_index_24, spiral_index_24_re = spiral_scan(24, 24)
spiral_index_48, spiral_index_48_re = spiral_scan(48, 48)
spiral_index_96, spiral_index_96_re = spiral_scan(96, 96)


def spiral_scan(input: torch.Tensor):
    B, C, H, W = input.shape
    assert H == W and H in [12, 24, 48, 96]
    spiral_indices = None
    if H == 24:
        spiral_indices = spiral_index_24
    elif H == 48:
        spiral_indices = spiral_index_48
    elif H == 96:
        spiral_indices = spiral_index_96
    elif H == 12:
        spiral_indices = spiral_index_12

    xs = input.new_empty((B, 2, C, H * W))
    xs[:, 0] = input.view(B, C, H * W).index_select(2, spiral_indices[0])
    xs[:, 1] = input.view(B, C, H * W).index_select(2, spiral_indices[1])
    return xs.contiguous()  # torch.Size([B, 2, C, H*W])


def spiral_reverse_scan(input: torch.Tensor, original_shape):
    B, C, H, W = original_shape
    _, K, _, HW = input.shape
    assert K == 4
    assert HW == H * W
    assert H == W and H in [12, 24, 48, 96]
    spiral_indices = None
    if H == 24:
        spiral_indices = spiral_index_24_re
    elif H == 48:
        spiral_indices = spiral_index_48_re
    elif H == 96:
        spiral_indices = spiral_index_96_re
    elif H == 12:
        spiral_indices = spiral_index_12_re

    output = torch.zeros((B, C, H, W), device=input.device, dtype=input.dtype)
    for i in range(2):
        x_indices = spiral_indices[i][:, 1].long()
        y_indices = spiral_indices[i][:, 0].long()

        output[:, :, x_indices, y_indices] += input[:, i, :, torch.arange(input.size(3))]
        output[:, :, x_indices, y_indices] += input[:, i + 2, :, (H * W - 1 - torch.arange(input.size(3)))]

    return output.contiguous()  # torch.Size([B, C, H, W])


class CrossScan_Spiral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs[:, 4:6] = spiral_scan(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y_rc = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rc = y_rc[:, 0] + y_rc[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rc = y_rc.view(B, -1, H, W)
        y_h = spiral_reverse_scan(ys[:, 4:8], (B, C, H, W))
        y = y_h + y_rc
        return y.view(B, -1, H, W)


class CrossMerge_Spiral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)

        y_rc = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y_rc = y_rc[:, 0] + y_rc[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rc = y_rc.view(B, -1, H, W)
        y_h = spiral_reverse_scan(ys[:, 4:8], (B, D, H, W))
        y = y_h + y_rc
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 8, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs[:, 4:6] = spiral_scan(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        xs = xs.view(B, 8, C, H, W)
        return xs


# =======================
hilbert_index_12, hilbert_index_12_re = get_hilber_spacefill(1, 1, 12, 12)
hilbert_index_12 = torch.tensor(hilbert_index_12, dtype=torch.int).cuda()
hilbert_index_12_re = torch.tensor(hilbert_index_12_re, dtype=torch.int).cuda()

hilbert_index_24, hilbert_index_24_re = get_hilber_spacefill(1, 1, 24, 24)
hilbert_index_24 = torch.tensor(hilbert_index_24, dtype=torch.int).cuda()
hilbert_index_24_re = torch.tensor(hilbert_index_24_re, dtype=torch.int).cuda()

hilbert_index_48, hilbert_index_48_re = get_hilber_spacefill(1, 1, 48, 48)
hilbert_index_48 = torch.tensor(hilbert_index_48, dtype=torch.int).cuda()
hilbert_index_48_re = torch.tensor(hilbert_index_48_re, dtype=torch.int).cuda()

hilbert_index_96, hilbert_index_96_re = get_hilber_spacefill(1, 1, 96, 96)
hilbert_index_96 = torch.tensor(hilbert_index_96, dtype=torch.int).cuda()
hilbert_index_96_re = torch.tensor(hilbert_index_96_re, dtype=torch.int).cuda()


def hilbert_scan(input: torch.Tensor):
    B, C, H, W = input.shape
    assert H == W and H in [12, 24, 48, 96]
    hilbert_indices = None
    if H == 24:
        hilbert_indices = hilbert_index_24
    elif H == 48:
        hilbert_indices = hilbert_index_48
    elif H == 96:
        hilbert_indices = hilbert_index_96
    elif H == 12:
        hilbert_indices = hilbert_index_12
    flip = torch.flip(input, [2])
    xs_1 = input.view(B, C, H * W).index_select(2, hilbert_indices).contiguous()
    xs_2 = flip.view(B, C, H * W).index_select(2, hilbert_indices).contiguous()
    xs_1_flip = torch.flip(xs_1, dims=[-1]).contiguous()
    xs_2_flip = torch.flip(xs_2, dims=[-1]).contiguous()
    return torch.stack([xs_1, xs_2, xs_1_flip, xs_2_flip], dim=1)  # torch.Size([B, 2, C, H*W])


def hilbert_reverse_scan(input: torch.Tensor, original_shape):
    B, C, H, W = original_shape
    _, _, _, HW = input.shape
    assert HW == H * W
    assert H == W and H in [12, 24, 48, 96]
    hilbert_indices = None
    if H == 24:
        hilbert_indices = hilbert_index_24_re
    elif H == 48:
        hilbert_indices = hilbert_index_48_re
    elif H == 96:
        hilbert_indices = hilbert_index_96_re
    elif H == 12:
        hilbert_indices = hilbert_index_12_re

    output = torch.zeros((B, C, H, W), device=input.device, dtype=input.dtype)

    x_indices = hilbert_indices[:, 1].long()
    y_indices = hilbert_indices[:, 0].long()

    output[:, :, x_indices, y_indices] += input[:, 0, :, torch.arange(input.size(3))]  # xs_1
    flip_x_indices = H - 1 - x_indices
    output[:, :, flip_x_indices, y_indices] += input[:, 1, :, torch.arange(input.size(3))]  # xs_2
    output[:, :, x_indices, y_indices] += input[:, 2, :, (H * W - 1 - torch.arange(input.size(3)))]  # xs_1_flip
    output[:, :, flip_x_indices, y_indices] += input[:, 3, :, (H * W - 1 - torch.arange(input.size(3)))]  # xs_2_flip

    return output.contiguous()  # torch.Size([B, C, H, W])


class CrossScan_Hilbert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0:4] = hilbert_scan(x)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        y = hilbert_reverse_scan(ys[:, 0:4], (B, C, H, W))
        return y.view(B, -1, H, W)


class CrossMerge_Hilbert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = hilbert_reverse_scan(ys[:, 0:4], (B, D, H, W))
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0:4] = hilbert_scan(x.view(B, C, H, W))
        xs = xs.view(B, 4, C, H, W)
        return xs


################################################
def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).contiguous().reshape(B, C, H * W)


def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).contiguous().reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2).contiguous()
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2).contiguous()
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan_Diagonal(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)
        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge_Diagonal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3).contiguous()
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)


class CrossScan_DS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        index = spiral_line_index[str(H)]
        xs = x.new_empty((B, 8, C, H * W))
        xs[:, 0:4] = get_line_scan(x, index)

        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        index = spiral_line_index[str(H)]
        L = H * W
        y_h = get_line_re_scan(ys[:, 0:4], (B, C, H, W), index)
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))
        y = y_h + y_da
        return y.view(B, -1, H, W)


class CrossMerge_DS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        index = spiral_line_index[str(H)]
        ys = ys.view(B, K, D, -1)
        y_h = get_line_re_scan(ys[:, 0:4], (B, D, H, W), index)
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))
        y = y_h + y_da
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        index = spiral_line_index[str(H)]
        xs = x.new_empty((B, 8, C, L))

        xs[:, 0:4] = get_line_scan(x.view(B, C, H, W), index)
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        xs = xs.view(B, 8, C, H, W)
        return xs


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb;
            pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn, verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops
