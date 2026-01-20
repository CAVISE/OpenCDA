import torch

from opencood.utils import common_utils
from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_cuda


def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices


def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    # #######
    # import matplotlib.pyplot as plt
    # ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
    # ax.set_aspect('equal', 'box')
    # ax.set(xlim=(-50, 50),
    #        ylim=(-41.6, 41.6))
    # points0 = points[0].cpu().detach().numpy()
    # boxes0 = boxes[0].cpu().detach().numpy()
    # ax.plot(points0[:, 0], points0[:, 1], 'y.', markersize=3)
    # ax.plot(boxes0[:, 0], boxes0[:, 1], 'r.', markersize=10)
    # plt.xlabel('x')
    # plt.ylabel('y')
    #
    # plt.show()
    # plt.close()
    # ########
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    return box_idxs_of_pts
