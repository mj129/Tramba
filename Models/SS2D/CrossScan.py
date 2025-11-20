import torch
import pickle
import os
# ================HilBert Scan=============
def xy2d(n, x, y, d):
    """将 (x, y) 坐标转换为 Hilbert 曲线距离 d"""
    s = 1
    t = d
    rx = ry = 0
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rotate(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def d2xy(n, d):
    """将 Hilbert 曲线距离 d 转换为 (x, y) 坐标"""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rotate(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def rotate(n, x, y, rx, ry):
    """旋转操作"""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def generate_hilbert_indices(n):
    """生成多个方向的 Hilbert 曲线索引"""
    hilbert_indices = {}
    for direction in range(4):
        indices = []
        for d in range(n * n):
            x, y = d2xy(n, d)
            if direction % 2 == 1:
                x, y = n - 1 - x, y
            if direction // 2 == 1:
                x, y = y, x
            indices.append((x, y))
        hilbert_indices[direction] = indices
    return hilbert_indices


def precompute_hilbert_indices(sizes, save_path='/root/autodl-tmp/M3Net/Models/selective_scan'):
    """预先计算指定尺寸的 Hilbert 曲线索引"""
    hilbert_indices_dict = {}
    for size in sizes:
        hilbert_indices = generate_hilbert_indices(size)
        hilbert_indices_dict[size] = hilbert_indices
    return hilbert_indices_dict


def hilbert_scan(x, hilbert_indices_dict):
    B, C, H, W = x.shape
    assert H == W, "H 和 W 必须相等"
    assert H in hilbert_indices_dict, "H 的值必须是预计算的尺寸之一"
    # 从字典中获取预先计算好的 Hilbert 曲线索引
    hilbert_indices = hilbert_indices_dict[H]
    num_directions = len(hilbert_indices)
    # 初始化包含多个方向的 Hilbert 曲线扫描结果的张量
    xs = x.new_empty((B, num_directions, C, H * W))
    # 对每个方向进行 Hilbert 曲线扫描
    for direction in range(num_directions):
        indices = hilbert_indices[direction]
        linear_indices = [y * H + x for x, y in indices]
        xs[:, direction] = x.view(B, C, H * W).index_select(2, torch.tensor(linear_indices).to(x.device))
    return xs  # torch.Size([B, 4, C, H*W])

# ============Morton Scan==============
def interleave_bits(x, y):
    z = 0
    for i in range(32):  # Assuming 32 bits is enough for H and W
        z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
    return z

def morton_order_indices(H, W):
    indices = torch.zeros(H, W, dtype=torch.long).cuda()
    for y in range(H):
        for x in range(W):
            indices[y, x] = interleave_bits(x, y)
    return indices

def rotate_indices(indices, direction):
    if direction == 'up':
        return torch.rot90(indices, k=2, dims=(0, 1))
    elif direction == 'down':
        return indices
    elif direction == 'left':
        return torch.rot90(indices, k=1, dims=(0, 1))
    elif direction == 'right':
        return torch.rot90(indices, k=-1, dims=(0, 1))
    else:
        raise ValueError("Invalid direction. Supported directions are 'up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right'.")

# Precompute and store Morton order indices for fixed sizes
def precompute_morton_indices(sizes):
    indices_dict = {}
    for size in sizes:
        H, W = size, size
        base_indices = morton_order_indices(H, W)
        indices_dict[size] = {
            'down': base_indices.reshape(-1),
            'up': rotate_indices(base_indices, 'up').reshape(-1),
            'left': rotate_indices(base_indices, 'left').reshape(-1),
            'right': rotate_indices(base_indices, 'right').reshape(-1),
        }
    return indices_dict

# Flatten tensor using precomputed indices
def z_order_flatten(tensor, indices_dict, direction='down'):
    device = tensor.device
    B, C, H, W = tensor.size()
    indices = indices_dict[H][direction]
    tensor = tensor.view(B, C, H * W)
    sorted_indices = indices.argsort()
    return tensor.index_select(2, sorted_indices)

def morton_scan(tensor, indices_dict):
    directions = ['down', 'right']
    flattened_tensors = [z_order_flatten(tensor, indices_dict, direction) for direction in directions]
    combined_tensor = torch.stack(flattened_tensors, dim=1)
    return combined_tensor

# ==========Other Scan==============

def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


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
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
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
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor, hilbert_indices_dict):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        # xs[:, 4] = hilbert_scan(x, hilbert_indices_dict)

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


class CrossMerge(torch.autograd.Function):
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
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.flatten(2, 3)
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


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys.sum(dim=1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1).contiguous()
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        y = ys.sum(dim=1).view(B, C, H, W)
        return y


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        y = ys.sum(dim=1).view(B, D, H * W)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.view(B, 1, C, L).repeat(1, 4, 1, 1).contiguous().view(B, 4, C, H, W)
        return xs


if __name__ == '__main__':
    import numpy as np
    # 示例数据

    B, C, H, W = 1, 1, 4, 4  # 确保 H 和 W 是2的幂
    data = np.arange(H * W).reshape(H, W)
    x = torch.tensor(data).unsqueeze(0).unsqueeze(0).cuda()
    # 按 Hilbert 曲线展开
    hilbert_dict = precompute_hilbert_indices([4, 8])
    xs = hilbert_scan(x, hilbert_dict)
    print(x)
    print(xs.shape)
    for i in range(xs.shape[1]):
        print(xs[:, i])

    # B, C, H, W = 1, 1, 4, 4  # 确保 H 和 W 是2的幂
    # data = np.arange(H * W).reshape(H, W)
    # x = torch.tensor(data).unsqueeze(0).unsqueeze(0).cuda()
    # print(x)
    # # 按 Hilbert 曲线展开
    # morton_index = precompute_morton_indices([4, 8])
    # xs = morton_scan(x, morton_index)
    #
    # print(xs.shape)
    #
    # for i in range(xs.shape[1]):
    #     print(xs[:, i])

    # B, C, H, W = x.shape
    # # 初始化包含四种扫描方向和 Hilbert 曲线扫描结果的张量
    # xs = x.new_empty((B, 4, C, H * W))
    #
    # # 添加横向和竖向的扫描
    # xs[:, 0] = x.flatten(2, 3)
    # xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
    # xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

    # 添加 Hilbert 曲线扫描
    # xs[:, 4] = hilbert_scan(x, hilbert_dict)
    # xs 的形状为 (B, C, H*W)
    # print(x)
    # for i in range(xs.shape[1]):
    #     print(xs[:, i].shape)
    #     scan_result = xs[:, i].cpu().numpy().reshape(H, W)
    #     print(f"扫描结果 {i}:")
    #     print(scan_result)


