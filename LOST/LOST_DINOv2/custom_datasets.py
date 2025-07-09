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
from torch.utils.data import Subset


from PIL import Image
from tqdm import tqdm
from torchvision import transforms as pth_transforms

# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class ImageDataset:
    def __init__(self, image_path):
        
        self.image_path = image_path
        self.name = image_path.split("/")[-1]

        # Read the image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Build a dataloader
        img = transform(img)
        self.dataloader = [[img, image_path]]

    def get_image_name(self, *args, **kwargs):
        return self.image_path.split("/")[-1].split(".")[0]

    def load_image(self, *args, **kwargs):
        return skimage.io.imread(self.image_path)

class Dataset:
    def __init__(self, dataset_name, dataset_set, remove_hards):
        """
        Build the dataloader for the chosen dataset.
        If remove_hards=True (and it's VOC), we skip images that are "hard-only."
        """
        self.dataset_name = dataset_name
        self.set = dataset_set
        self.remove_hards = remove_hards

        # 1) Figure out root paths
        if dataset_name == "VOC07":
            self.root_path = "datasets/VOC2007"
            self.year = "2007"
        elif dataset_name == "VOC12":
            self.root_path = "datasets/VOC2012"
            self.year = "2012"
        elif dataset_name == "COCO20k":
            self.year = "2014"
            self.root_path = f"datasets/COCO/images/{dataset_set}{self.year}"
            self.sel20k = 'datasets/coco_20k_filenames.txt'
            self.all_annfile = "datasets/COCO/annotations/instances_train2014.json"
            self.annfile = "datasets/instances_train2014_sel20k.json"
            if not os.path.exists(self.annfile):
                select_coco_20k(self.sel20k, self.all_annfile)
        else:
            raise ValueError("Unknown dataset name.")

        if not os.path.exists(self.root_path):
            raise ValueError("Please follow the README to set up the datasets properly.")

        # 2) Build the underlying dataset
        if "VOC" in dataset_name:
            base_dataset = torchvision.datasets.VOCDetection(
                self.root_path,
                year=self.year,
                image_set=self.set,
                transform=transform,
                download=False,
            )
        elif "COCO20k" in dataset_name:
            base_dataset = torchvision.datasets.CocoDetection(
                self.root_path, annFile=self.annfile, transform=transform
            )
        else:
            raise ValueError("Unknown dataset name")

        # 3) Optionally skip "hard-only" images if it's VOC
        if remove_hards and "VOC" in dataset_name:
            print("Discovering 'hard-only' images for VOC ...")
            self.hards = discard_hard_voc(base_dataset)
            print(f"Found {len(self.hards)} 'hard-only' images in {dataset_name} {self.set}")

            all_indices = range(len(base_dataset))
            valid_indices = [idx for idx in all_indices if idx not in self.hards]

            # Build a Subset omitting the hard-only indices
            self.dataloader = Subset(base_dataset, valid_indices)
            self.name = f"{self.dataset_name}_{self.set}-nohards"
        else:
            # No changes
            self.dataloader = base_dataset
            self.hards = []
            self.name = f"{self.dataset_name}_{self.set}"

    def load_image(self, im_name):
        """
        For visualization usage: load the original image from disk
        """
        if "VOC" in self.dataset_name:
            # e.g. "000005.jpg" => load from /datasets_local/VOC2007/JPEGImages/
            return skimage.io.imread(f"/datasets_local/VOC{self.year}/JPEGImages/{im_name}")
        elif "COCO" in self.dataset_name:
            # If you want to load from COCO local path, you'd need a mapping
            # Not shown here
            pass
        else:
            raise ValueError("Unknown dataset name.")
        return None

    def get_image_name(self, inp):
        """
        Return the image name from the annotation.
        - For VOC: inp["annotation"]["filename"]
        - For COCO: str(inp[0]["image_id"])
        """
        if "VOC" in self.dataset_name:
            im_name = inp["annotation"]["filename"]
        elif "COCO" in self.dataset_name:
            im_name = str(inp[0]["image_id"])
        else:
            im_name = None
        return im_name

    def extract_gt(self, targets, im_name):
        # Parse bounding boxes
        if "VOC" in self.dataset_name:
            return extract_gt_VOC(targets, remove_hards=self.remove_hards)
        elif "COCO" in self.dataset_name:
            return extract_gt_COCO(targets, remove_iscrowd=True)
        else:
            raise ValueError("Unknown dataset name.")


def discard_hard_voc(base_dataset):
    """
    Returns a list of indices that correspond to images with *only* truncated/difficult objects.
    """
    hards = []
    for idx in range(len(base_dataset)):
        # base_dataset[idx] => (img_tensor, annotation_dict)
        _, ann = base_dataset[idx]
        objects = ann["annotation"]["object"]
        all_hard = True
        for obj in objects:
            if obj["truncated"] != "1" and obj["difficult"] != "1":
                all_hard = False
                break
        if all_hard:
            hards.append(idx)
    return hards



def extract_gt_COCO(targets, remove_iscrowd=True):
    objects = targets
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        # Remove iscrowd boxes
        if remove_iscrowd and objects[o]["iscrowd"] == 1:
            continue
        gt_cls = objects[o]["category_id"]
        gt_clss.append(gt_cls)
        bbx = objects[o]["bbox"]
        x1y1x2y2 = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        x1y1x2y2 = [int(round(x)) for x in x1y1x2y2]
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss


def extract_gt_VOC(targets, remove_hards=False):
    objects = targets["annotation"]["object"]
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        if remove_hards and (
            objects[o]["truncated"] == "1" or objects[o]["difficult"] == "1"
        ):
            continue
        gt_cls = objects[o]["name"]
        gt_clss.append(gt_cls)
        obj = objects[o]["bndbox"]
        x1y1x2y2 = [
            int(obj["xmin"]),
            int(obj["ymin"]),
            int(obj["xmax"]),
            int(obj["ymax"]),
        ]
        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        x1y1x2y2[0] -= 1
        x1y1x2y2[1] -= 1
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss


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

def select_coco_20k(sel_file, all_annotations_file):
    print('Building COCO 20k dataset.')

    # load all annotations
    with open(all_annotations_file, "r") as f:
        train2014 = json.load(f)

    # load selected images
    with open(sel_file, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]
    im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]

    new_anno = []
    new_images = []

    for i in tqdm(im20k):
        new_anno.extend(
            [a for a in train2014["annotations"] if a["image_id"] == int(i)]
        )
        new_images.extend([a for a in train2014["images"] if a["id"] == int(i)])

    train2014_20k = {}
    train2014_20k["images"] = new_images
    train2014_20k["annotations"] = new_anno
    train2014_20k["categories"] = train2014["categories"]

    with open("datasets/instances_train2014_sel20k.json", "w") as outfile:
        json.dump(train2014_20k, outfile)

    print('Done.')