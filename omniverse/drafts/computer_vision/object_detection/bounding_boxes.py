import torch
from typing import *
import numpy as np


def voc2coco(
    bboxes: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Convert pascal_voc to coco format.

    voc  => [xmin, ymin, xmax, ymax]
    coco => [xmin, ymin, w, h]

    Args:
        bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, xmax, ymax].

    Returns:
        coco_bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, w, h].
    """

    # careful in place can cause mutation
    bboxes[..., 2:4] -= bboxes[..., 0:2]

    return bboxes


def coco2voc(
    bboxes: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Convert coco to pascal_voc format.

    coco => [xmin, ymin, w, h]
    voc  => [xmin, ymin, xmax, ymax]


    Args:
        bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, w, h].

    Returns:
        voc_bboxes (torch.Tensor): Shape of (N, 4) where N is the number of samples and 4 is the coordinates [xmin, ymin, xmax, ymax].
    """

    # careful in place can cause mutation
    bboxes[..., 2:4] += bboxes[..., 0:2]

    return bboxes


def voc2yolo(
    bboxes: Union[np.ndarray, torch.Tensor],
    height: int = 720,
    width: int = 1280,
) -> Union[np.ndarray, torch.Tensor]:
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """

    # otherwise all value will be 0 as voc_pascal dtype is np.int
    # bboxes = bboxes.copy().astype(float)

    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height

    bboxes[..., 2] -= bboxes[..., 0]
    bboxes[..., 3] -= bboxes[..., 1]

    bboxes[..., 0] += bboxes[..., 2] / 2
    bboxes[..., 1] += bboxes[..., 3] / 2

    return bboxes


def yolo2voc(
    bboxes: Union[np.ndarray, torch.Tensor],
    height: int = 720,
    width: int = 1280,
) -> Union[np.ndarray, torch.Tensor]:
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    """

    # otherwise all value will be 0 as voc_pascal dtype is np.int
    # bboxes = bboxes.copy().astype(float)

    bboxes[..., 0] -= bboxes[..., 2] / 2
    bboxes[..., 1] -= bboxes[..., 3] / 2

    bboxes[..., 2] += bboxes[..., 0]
    bboxes[..., 3] += bboxes[..., 1]

    bboxes[..., 0::2] *= width
    bboxes[..., 1::2] *= height

    return bboxes
