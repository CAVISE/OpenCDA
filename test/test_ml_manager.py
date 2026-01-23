"""
Unit test for ML Manager.
"""

import os
import sys
import unittest
import types

import cv2
import numpy as np
from unittest.mock import patch

# temporary solution for relative imports in case opencda is not installed
# if opencda is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import opencda.customize.ml_libs.ml_manager as ml_manager_module
from opencda.customize.ml_libs.ml_manager import MLManager

class _FakeTensor:
    """Minimal torch.Tensor-like wrapper for what draw_2d_box needs."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
        self.is_cuda = False

    def cpu(self):
        return self
    
    def detach(self):
        return self
    
    def numpy(self):
        return self._arr

    def tolist(self):
        return self._arr.tolist()

    def __iter__(self):
        return iter(self._arr)

    def __getitem__(self, idx):
        return self._arr[idx]


class _FakeDetections:
    """
    Minimal YOLOv5-like detections object:
    - len(results) == 1 for single image
    - .render() returns list of images
    - .xyxy[0] returns Nx6 tensor-like with boxes [x1,y1,x2,y2,conf,cls]
    """

    def __init__(self, img):
        # Ensure we always have a valid image array to return
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
        self.ims = [img.copy()]

        # YOLOv5-style class names lookup used by MLManager.draw_2d_box:
        # label_name = result.names[label]
        self.names = {0: "obj"}

        h, w = img.shape[:2]
        # one bounding box inside image
        box = [0, 0, max(0, w - 1), max(0, h - 1), 0.9, 0]
        self.xyxy = [_FakeTensor([box])]

    def __len__(self):
        return 1

    def render(self):
        return self.ims


class _FakeDetector:
    """Callable model that returns fake detections for one image."""

    def __call__(self, img):
        return _FakeDetections(img)

 


class TestMlManager(unittest.TestCase):
    def setUp(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.data = cv2.imread(os.path.join(current_path, "data/test.jpg"))
        # Make sure torch.hub exists in the imported module even if torch is not real PyTorch
        if not hasattr(ml_manager_module.torch, "hub"):
            ml_manager_module.torch.hub = types.SimpleNamespace()

        # Patch hub.load so MLManager() doesn't download anything and works without real torch
        self._hub_load_patcher = patch.object(
            ml_manager_module.torch.hub,
            "load",
            return_value=_FakeDetector(),
            create=True,
        )
        self._hub_load_patcher.start()
        self.addCleanup(self._hub_load_patcher.stop)

        self.ml_manager = MLManager()

    def test_parameters(self):
        assert self.ml_manager.object_detector

    def test_draw_2d_bbx(self):
        results = self.ml_manager.object_detector(self.data)
        assert len(results) == 1
        assert self.ml_manager.draw_2d_box(results, self.data, 0).shape == self.data.shape
