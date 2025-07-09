#!/usr/bin/env python3
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
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from models_v2_source import deit_small_patch16_LS as create_source_model
from models_v2_reg import deit_small_patch16_LS as create_reg_model

from custom_datasets import VOC2007Dataset, bbox_iou
from object_discovery import lost, dino_seg

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LOST on VOC2007 with DeiT models")

    parser.add_argument("--voc_root", default="datasets/VOC2007", type=str,
                        help="Path to the VOC2007 folder (must contain Annotations/, JPEGImages/, etc.)")
    parser.add_argument("--voc_set", default="trainval", type=str,
                        choices=["train", "val", "trainval", "test"],
                        help="Which split of VOC2007 to run on.")

    # Make checkpoint_path optional: if you do "pretrained" mode, you can skip it
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="Path to a local .pth checkpoint (ignored if model_type=pretrained).")

    parser.add_argument("--model_type", default="source", type=str,
                        choices=["source", "reg", "pretrained"],
                        help=(
                            "'source': no register tokens + load local checkpoint.\n"
                            "'reg':    4 register tokens + load local checkpoint.\n"
                            "'pretrained': use official DeiT-III weights from URL (ignore local checkpoint)."
                        ))
    parser.add_argument("--k_patches", default=100, type=int,
                        help="Number of patches with lowest degree to consider in LOST.")
    parser.add_argument("--no_evaluation", action="store_true",
                        help="If set, do not compute CorLoc evaluation.")

    args = parser.parse_args()

    # 1) Build the dataset for VOC2007
    dataset = VOC2007Dataset(root_path=args.voc_root, image_set=args.voc_set)

    # 2) Build the model
    if args.model_type == "source":
        # no register tokens, do NOT load pretrained from the URL by default
        model = create_source_model(img_size=224, num_classes=1000, pretrained=False)

    elif args.model_type == "reg":
        # 4 register tokens, also not from the official pretrained
        model = create_reg_model(img_size=224, num_classes=1000, num_regs=4, pretrained=False)

    else:
        # "pretrained" => official DeiT-III from URL, no local checkpoint load
        model = create_source_model(img_size=224, num_classes=1000, pretrained=True)

    # 2b) If we are "source" or "reg", load from local checkpoint
    if args.model_type in ["source", "reg"]:
        if not args.checkpoint_path:
            raise ValueError(
                "You must provide --checkpoint_path when using model_type=source or reg."
            )
        print(f"[INFO] Loading local checkpoint from {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
    else:
        # model_type == "pretrained"
        # We skip loading from local checkpoint entirely
        if args.checkpoint_path is not None:
            print(
                f"[WARNING] model_type=pretrained was chosen: ignoring checkpoint_path={args.checkpoint_path}"
            )

    # Put on device
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) We iterate over the dataset
    corloc = []
    preds_dict = {}
    pbar = tqdm(dataset.dataloader)

    for im_id, (img, target_dict) in enumerate(pbar):
        im_name = dataset.get_image_name(target_dict)
        if im_name is None:
            continue

        # Original image size
        original_image = dataset.load_image(im_name)
        original_size = (original_image.shape[0], original_image.shape[1])

        img = img.to(device)

        # Compute padding so that height & width become multiples of 16
        pad_H = int(np.ceil(img.shape[1] / 16.0) * 16) - img.shape[1]
        pad_W = int(np.ceil(img.shape[2] / 16.0) * 16) - img.shape[2]
        if pad_H > 0 or pad_W > 0:
            img = F.pad(img, (0, pad_W, 0, pad_H), mode="constant", value=0)

        # We do the forward pass to get last self-attention
        with torch.no_grad():
            input_for_model = img.unsqueeze(0)  # [1,3,H,W]
            attentions = model.get_last_selfattention(input_for_model)

        B, nh, N, _ = attentions.shape
        # Typically: B=1, nh=6 (DeiT small), N=197 (1 cls token + 196 patches at 14x14)

        # We also capture QKV from the last block
        feat_out = {}
        def save_qkv(module, inp, out):
            feat_out["qkv"] = out

        # Attach forward hook to the last block's qkv
        last_block = model.blocks[-1]
        handle = last_block.attn.qkv.register_forward_hook(save_qkv)

        # Standard forward pass (so the hook above is triggered)
        _ = model(input_for_model)
        handle.remove()

        # qkv_raw shape => [batch_size*N, 3*embed_dim], typically [197, 3*384] for DeiT small
        qkv_raw = feat_out["qkv"]

        embed_dim = model.embed_dim  # e.g. 384 for deit_small
        num_heads = last_block.attn.num_heads
        tokens = N  # 197

        # Reshape qkv
        # => qkv shape: [3, B, num_heads, tokens, embed_dim//num_heads]
        # but currently qkv_raw is [B*N, 3*embed_dim], so we carefully reshape
        qkv = qkv_raw.reshape(B, tokens, 3, num_heads, embed_dim // num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v each => [B, num_heads, tokens, dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # We'll use v as in the LOST paper
        feats = v  # shape [B, num_heads, tokens, dim_per_head]

        # Flatten across all heads
        feats = feats.transpose(1, 2).reshape(B, tokens, -1)
        # => [B, tokens, num_heads*dim_per_head]

        # If "reg" model, we skip the 4 register tokens
        if args.model_type == "reg":
            num_reg_tokens = 4
            feats = feats[:, 1 + num_reg_tokens :, :]
        else:
            # For "source" or "pretrained", skip only the 1 CLS token
            feats = feats[:, 1:, :]

        # w_featmap, h_featmap => how many patch tokens horizontally / vertically
        # For a 224×224 image, we have 14×14 patches => w_featmap=14, h_featmap=14
        w_featmap = img.shape[-2] // 16
        h_featmap = img.shape[-1] // 16

        # scales = [patch_size_x, patch_size_y]
        scales = [16, 16]

        # Apply LOST
        pred_box, A, scores, seed = lost(
            feats, [w_featmap, h_featmap],
            scales, original_size, k_patches=args.k_patches
        )
        preds_dict[im_name] = pred_box

        # Evaluate with CorLoc if requested
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(target_dict, im_name)
            if gt_bbxs is None or len(gt_bbxs) == 0:
                # This image may have "difficult" or no objects
                continue

            # We must transform ground‐truth boxes the same way we transform images
            transformed_gt_bbxs = dataset.transform_bboxes(
                gt_bbxs,
                original_size=original_size,
                transformed_size=224,  # because of the Resize(224) in transforms
                padding=(pad_H, pad_W)
            )

            # Now compute IoU
            ious = bbox_iou(
                torch.from_numpy(pred_box),       # shape [4]
                torch.from_numpy(transformed_gt_bbxs)  # shape [num_gt_boxes, 4]
            )
            # If *any* of the ground‐truth boxes is IoU ≥ 0.5 => corloc=1
            corloc.append(1 if torch.any(ious >= 0.5) else 0)

    # 4) End of dataset iteration => print CorLoc results
    if not args.no_evaluation and len(corloc) > 0:
        score = 100.0 * np.mean(corloc)
        print(f"Final CorLoc on VOC2007/{args.voc_set}: {score:.2f}%  ({sum(corloc)}/{len(corloc)})")
    else:
        print("No evaluation requested or no ground-truth found.")

    # 5) Optionally save predictions
    out_path = "preds_lost.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(preds_dict, f)
    print("Saved predictions to", out_path)
