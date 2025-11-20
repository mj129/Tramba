import torch

def generate_dilation_indices(H, W, dilation_rate=4):
    """
    Generate indices for dilation scan.

    Args:
    H: Height of the image.
    W: Width of the image.
    dilation_rate:

    Returns:
    A dictionary containing indices for normal and flipped scans.
    """

    coords = torch.zeros((4, H * W, 2), dtype=torch.int)
    for i in range(H):
        for j in range(W):
            coords[0][i * W + j] = torch.tensor([i, j])
    for i in range(H):
        for j in range(W):
            coords[1][i * W + j] = torch.tensor([j, i])
    coords[2] = torch.flip(coords[0], dims=[0])
    coords[3] = torch.flip(coords[1], dims=[0])

    dilated_masks = ((torch.arange(H * W) % dilation_rate).unsqueeze(0) == torch.arange(dilation_rate).unsqueeze(1))
    scan_point = []
    scan_point_flip = []
    scan_point_trans = []
    scan_point_trans_flip = []
    for mask in dilated_masks:
        index = torch.where(mask)[0]
        scan_point.extend(coords[0][index].tolist())
        scan_point_flip.extend(coords[1][index].tolist())
        scan_point_trans.extend(coords[2][index].tolist())
        scan_point_trans_flip.extend(coords[3][index].tolist())

    scan_point = torch.tensor(scan_point).cuda()
    scan_point_flip = torch.tensor(scan_point_flip).cuda()
    scan_point_trans = torch.tensor(scan_point_trans).cuda()
    scan_point_trans_flip = torch.tensor(scan_point_trans_flip).cuda()

    scans = [scan_point, scan_point_flip, scan_point_trans, scan_point_trans_flip]

    return scans


def get_dilation_scan(input: torch.Tensor, indices):
    """
    Get the scan results of a tensor along a series of lines to cover the entire tensor.

    Args:
    input: PyTorch tensor of shape (B, C, H, W).
    indices: Precomputed indices for Bresenham lines.

    Returns:
    A tensor of shape (B, C, 4, H * W) containing the values at the points along the lines.
    """
    B, C, H, W = input.shape

    xs = torch.zeros((B, 4, C, H * W), device=input.device, dtype=input.dtype)

    flat_input = input.view(B, C, -1)  # Flatten input tensor along H * W

    for i in range(4):
        idx = indices[i][:, 0] * H + indices[i][:, 1]
        xs[:, i] = torch.index_select(flat_input, 2, idx)

    return xs


def get_re_dilation_scan(xs, original_shape, indices):
    """
    Restore the original tensor from the scanned tensor.

    Args:
    xs: Tensor of shape (B, 4, C, H * W) containing the scanned values.
    original_shape: Tuple (B, C, H, W) specifying the original shape of the tensor.
    indices: Precomputed indices for Bresenham lines.

    Returns:
    A tensor of shape (B, C, H, W) containing the values restored from the scanned tensor.
    """
    B, C, H, W = original_shape

    output = torch.zeros((B, C, H * W), device=xs.device, dtype=xs.dtype)

    for k in range(4):
        idx = indices[k][:, 0]  # x indices
        idy = indices[k][:, 1]  # y indices
        flat_indices = idx * H + idy
        # Accumulate xs[k] into output using scatter_add
        flat_xs_k = xs[:, k].view(B, C, H * W)
        flat_indices = flat_indices.expand(B, C, H * W)
        output.scatter_add_(2, flat_indices, flat_xs_k)
    return output.view(B, C, H, W)


if __name__ == '__main__':
    import timeit

    H, W = 4, 4
    B, C = 1, 1
    # data = np.arange(H * W).reshape(H, W)
    # x = torch.tensor(data, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
    indices = generate_dilation_indices(H, W, dilation_rate=3)
    x = torch.arange(B * C * H * W).reshape(B, C, H, W).float().cuda()
    # x = torch.randn((B, C, H, W)).cuda()
    print(x)
    # print(x[0][0])
    xs = get_dilation_scan(x, indices)
    print(xs)
    y = get_re_dilation_scan(xs, (B, C, H, W), indices)
    print(y[0][0])
    print(torch.all(y == 4 * x))

    def run_code():
        xs = get_dilation_scan(x, indices)
        y = get_re_dilation_scan(xs, (B, C, H, W), indices)


    # Measure execution time
    execution_time = timeit.timeit(run_code, number=100)

    # Print the average execution time per iteration
    print(f"Average execution time per iteration: {execution_time / 100} seconds")
