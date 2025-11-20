import torch

def generate_window_indices(H, W, window_size=4):
    """
    Generate indices for window scan.

    Args:
    H: Height of the image.
    W: Width of the image.
    window_size:

    Returns:
    A dictionary containing indices for normal and flipped scans.
    """
    assert 0 < window_size < H
    if window_size > H or window_size > W:
        window_size = H
    horizontal_indices = []
    vertical_indices = []
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            # Collect indices for the current window
            horizontal_window_indices = [(i + x, j + y) for x in range(window_size) for y in range(window_size)]
            horizontal_indices.extend(horizontal_window_indices)
            # Collect indices for the current window in vertical order
            vertical_window_indices = [(j + x, i + y) for y in range(window_size) for x in range(window_size)]
            vertical_indices.extend(vertical_window_indices)
    horizontal_flip_indices = horizontal_indices[::-1]
    vertical_flip_indices = vertical_indices[::-1]
    horizontal_points = torch.tensor(horizontal_indices, dtype=torch.int64).cuda()
    horizontal_flip_points = torch.tensor(horizontal_flip_indices, dtype=torch.int64).cuda()
    vertical_points = torch.tensor(vertical_indices, dtype=torch.int64).cuda()
    vertical_flip_points = torch.tensor(vertical_flip_indices, dtype=torch.int64).cuda()
    scans = [horizontal_points, horizontal_flip_points, vertical_points, vertical_flip_points]
    return scans


def get_window_scan(input: torch.Tensor, indices):
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


def get_re_window_scan(xs, original_shape, indices):
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

    H, W = 8, 8
    B, C = 1, 1
    # data = np.arange(H * W).reshape(H, W)
    # x = torch.tensor(data, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
    indices = generate_window_indices(H, W, window_size=4)
    x = torch.arange(B * C * H * W).reshape(B, C, H, W).float().cuda()
    # x = torch.randn((B, C, H, W)).cuda()
    print(x)
    # print(x[0][0])
    xs = get_window_scan(x, indices)
    print(xs)
    y = get_re_window_scan(xs, (B, C, H, W), indices)
    print(y[0][0])
    print(torch.all(y == 4 * x))

    # def run_code():
    #     xs = get_dilation_scan(x, indices)
    #     y = get_re_dilation_scan(xs, (B, C, H, W), indices)
    #
    #
    # # Measure execution time
    # execution_time = timeit.timeit(run_code, number=100)
    #
    # # Print the average execution time per iteration
    # print(f"Average execution time per iteration: {execution_time / 100} seconds")
