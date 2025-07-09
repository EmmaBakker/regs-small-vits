# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import json
import torchvision
import numpy as np
import skimage.io

from PIL import Image
from tqdm import tqdm
from torchvision import transforms as pth_transforms

# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        pth_transforms.Resize(224),        
        pth_transforms.CenterCrop(224),     
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class VOC2007Dataset:
    """
    Minimal version that loads only VOC2007.
    """
    def __init__(self, root_path="datasets/VOC2007", image_set="train"):
        # image_set can be "train", "val", "trainval", or "test"
        self.root_path = root_path
        self.year = "2007"
        self.set = image_set
        self.name = f"VOC07_{self.set}"
        # Build the pytorch dataset
        self.dataloader = torchvision.datasets.VOCDetection(
            self.root_path,
            year=self.year,
            image_set=self.set,
            transform=transform,
            download=False,
        )
    
    def get_image_name(self, inp):
        """Return the image name (filename) for each sample."""
        return inp["annotation"]["filename"]

    def load_image(self, im_name):
        """Load the image from disk using `skimage`."""
        full_path = os.path.join(self.root_path, "VOCdevkit/VOC2007/JPEGImages", im_name)
        return skimage.io.imread(full_path)

    def extract_gt(self, targets, im_name):
        """Extract the ground-truth bounding boxes."""
        objects = targets["annotation"]["object"]
        gt_bbxs = []
        gt_clss = []
        # skip "truncated" or "difficult" if you like
        for obj in objects:
            if obj["truncated"] == "1" or obj["difficult"] == "1":
                # skip these if you want (optional)
                continue
            gt_clss.append(obj["name"])
            bb = obj["bndbox"]
            # VOC are 1-based
            x1 = int(bb["xmin"]) - 1
            y1 = int(bb["ymin"]) - 1
            x2 = int(bb["xmax"]) - 1
            y2 = int(bb["ymax"]) - 1
            gt_bbxs.append([x1, y1, x2, y2])
        if len(gt_bbxs) == 0:
            return None, None
        return np.array(gt_bbxs), gt_clss

    def transform_bboxes(self, bboxes, original_size, transformed_size, padding, resize=True, center_crop=True):
        """
        Apply resizing, padding, and center cropping to bounding boxes.
        
        Parameters:
        - bboxes: numpy array of shape (N, 4) in [x1, y1, x2, y2] format.
        - original_size: tuple (H, W) of original image size.
        - transformed_size: int, size after resizing.
        - padding: tuple (pad_H, pad_W) applied after resizing.
        - resize: bool, whether to resize the bounding boxes.
        - center_crop: bool, whether to apply center cropping.
        
        Returns:
        - Transformed bounding boxes as a numpy array of shape (N, 4).
        """
        orig_h, orig_w = original_size
        new_h, new_w = transformed_size, transformed_size

        # Resize
        if resize:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            bboxes = bboxes.astype(np.float32)
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y

        # Apply padding (if any)
        pad_H, pad_W = padding
        if pad_H > 0 or pad_W > 0:
            bboxes[:, [0, 2]] += 0  # Assuming padding is applied to the right and bottom
            bboxes[:, [1, 3]] += 0

        # Apply center crop
        if center_crop:
            # Calculate the cropping offset
            crop_h = transformed_size
            crop_w = transformed_size
            img_h, img_w = new_h + pad_H, new_w + pad_W
            start_y = (img_h - crop_h) // 2
            start_x = (img_w - crop_w) // 2

            bboxes[:, [0, 2]] -= start_x
            bboxes[:, [1, 3]] -= start_y

            # Clip the boxes to be within the cropped area
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, crop_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, crop_h)

        return bboxes




def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

