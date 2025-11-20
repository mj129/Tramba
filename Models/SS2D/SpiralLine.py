import torch

def bresenham(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to generate a list of points (x, y) between (x0, y0) and (x1, y1)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def generate_indices(H, W):
    """
    Generate indices for Bresenham lines for a given image shape.

    Args:
    H: Height of the image.
    W: Width of the image.
    if_mask: If True, will use the mask to take an average of all elements that are **repeatedly** scanned

    Returns:
    A dictionary containing indices for normal and flipped scans.
    """
    line_points = []
    line_points_flip = []

    # Iterate over each row as the starting point
    for start_row in range(0, H, 2):
        points = bresenham(0, start_row, H - 1, W - 1 - start_row)
        line_points.extend(points)
        line_points_flip.extend(points[::-1])
    if H % 2 == 0:
        start = 0
    else:
        start = 2
    # Iterate over each column as the starting point
    for start_col in range(start, W, 2):
        points = bresenham(start_col, W - 1, H - 1 - start_col, 0)
        line_points.extend(points)
        line_points_flip.extend(points[::-1])

    line_points_1 = []
    line_points_1_flip = []

    for start_row in range(1, H, 2):
        points = bresenham(0, start_row, H - 1, W - 1 - start_row)
        line_points_1.extend(points)
        line_points_1_flip.extend(points[::-1])

    if H % 2 != 0:
        points = bresenham(0, W - 1, H - 1, 0)
        line_points_1.extend(points)
        line_points_1_flip.extend(points[::-1])

    # Iterate over each column as the starting point
    for start_col in range(1, W, 2):
        points = bresenham(start_col, W - 1, H - 1 - start_col, 0)
        line_points_1.extend(points)
        line_points_1_flip.extend(points[::-1])

    line_points = torch.tensor(line_points, dtype=torch.int64).cuda()
    line_points_flip = torch.tensor(line_points_flip, dtype=torch.int64).cuda()
    line_points_1 = torch.tensor(line_points_1, dtype=torch.int64).cuda()
    line_points_1_flip = torch.tensor(line_points_1_flip, dtype=torch.int64).cuda()

    scan_point = [line_points, line_points_flip, line_points_1, line_points_1_flip]
    return scan_point


def get_line_scan(input: torch.Tensor, indices):
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
        idx = indices[i][:, 0] + indices[i][:, 1] * H
        xs[:, i] = torch.index_select(flat_input, 2, idx)

    return xs


def get_line_re_scan(xs, original_shape, indices):
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
        flat_indices = idx + idy * H
        # Accumulate xs[k] into output using scatter_add
        flat_xs_k = xs[:, k].view(B, C, H * W)
        flat_indices = flat_indices.expand(B, C, H * W)
        output.scatter_add_(2, flat_indices, flat_xs_k)
    return output.view(B, C, H, W)


if __name__ == '__main__':
    import timeit

    H, W = 6, 6
    B, C = 1, 1
    # data = np.arange(H * W).reshape(H, W)
    # x = torch.tensor(data, dtype=torch.float).unsqueeze(0).unsqueeze(0).cuda()
    indices = generate_indices(H, W)
    x = torch.arange(B * C * H * W).reshape(B, C, H, W).float().cuda()
    # x = torch.ones_like(x)
    print(x)
    # print(x[0][0])
    xs = get_line_scan(x, indices)
    print(xs)
    y = get_line_re_scan(xs, (B, C, H, W), indices)
    print(y[0][0])


    def run_code():
        xs = get_line_scan(x, indices)
        y = get_line_re_scan(xs, (B, C, H, W), indices)


    # Measure execution time
    execution_time = timeit.timeit(run_code, number=100)

    # Print the average execution time per iteration
    print(f"Average execution time per iteration: {execution_time / 100} seconds")
