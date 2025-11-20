import torch

def spiral_scan_clockwise(H, W):
    # Initialize the boundaries of the matrix
    top, bottom = 0, H - 1
    left, right = 0, W - 1
    result = []

    while top <= bottom and left <= right:
        # Traverse from left to right
        for i in range(left, right + 1):
            result.append((top, i))
        top += 1

        # Traverse from top to bottom
        for i in range(top, bottom + 1):
            result.append((i, right))
        right -= 1

        if top <= bottom:
            # Traverse from right to left
            for i in range(right, left - 1, -1):
                result.append((bottom, i))
            bottom -= 1

        if left <= right:
            # Traverse from bottom to top
            for i in range(bottom, top - 1, -1):
                result.append((i, left))
            left += 1

    return result


def spiral_scan_counterclockwise(H, W):
    # Initialize the boundaries of the matrix
    top, bottom = 0, H - 1
    left, right = 0, W - 1
    result = []

    while top <= bottom and left <= right:
        # Traverse from right to left
        for i in range(right, left - 1, -1):
            result.append((bottom, i))
        bottom -= 1

        # Traverse from bottom to top
        for i in range(bottom, top - 1, -1):
            result.append((i, left))
        left += 1

        if top <= bottom:
            # Traverse from left to right
            for i in range(left, right + 1):
                result.append((top, i))
            top += 1

        if left <= right:
            # Traverse from top to bottom
            for i in range(top, bottom + 1):
                result.append((i, right))
            right -= 1

    return result


def spiral_scan(H, W):
    results_clockwise = spiral_scan_clockwise(H, W)
    results_counterclockwise = spiral_scan_counterclockwise(H, W)
    results_clockwise_re = []
    results_counterclockwise_re = []
    for (x, y) in results_clockwise:
        results_clockwise_re.append(y * H + x)
    for (x, y) in results_counterclockwise:
        results_counterclockwise_re.append(y * H + x)
    spiral_scan_list = []
    spiral_scan_re_list = []
    results_clockwise = torch.tensor(results_clockwise, dtype=torch.int).cuda()
    results_counterclockwise = torch.tensor(results_counterclockwise, dtype=torch.int).cuda()
    results_clockwise_re = torch.tensor(results_clockwise_re, dtype=torch.int).cuda()
    results_counterclockwise_re = torch.tensor(results_counterclockwise_re, dtype=torch.int).cuda()
    spiral_scan_list.append(results_clockwise_re)
    spiral_scan_list.append(results_counterclockwise_re)
    spiral_scan_re_list.append(results_clockwise)
    spiral_scan_re_list.append(results_counterclockwise)
    return spiral_scan_list, spiral_scan_re_list


if __name__ == '__main__':
    spiral_index_12, spiral_index_12_re = spiral_scan(6, 6)
    print(spiral_index_12_re)