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

import torch
import torch.nn as nn
import os

try:
    from dinov2.models.vision_transformer import vit_large
except ImportError:
    raise ImportError(
        "Could not import DINOv2. Ensure 'dinov2/' is in your Python path or working directory."
    )


def get_model(arch, patch_size, resnet_dilate, device):

    # You can remove these lines if you already imported vit_large
    # from dinov2.models.vision_transformer import vit_large

    if arch == "dinov2_vitl14":
        print("==> Building DINOv2 ViT-L/14 (no registers)")
        model = vit_large(
            patch_size=patch_size,
            img_size=518,
            num_register_tokens=0,  # no registers
            init_values=1.0,
            block_chunks=0,
        )
        ckpt_path = "dinov2/models/dinov2_vitl14_pretrain.pth"

    elif arch == "dinov2_vitl14_reg":
        print("==> Building DINOv2 ViT-L/14 (with registers)")
        model = vit_large(
            patch_size=patch_size,
            img_size=518,
            num_register_tokens=4,  # 4 registers
            init_values=1.0,f
            block_chunks=0,
        )
        ckpt_path = "dinov2/models/dinov2_vitl14_reg4_pretrain.pth"

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    print(f"Loading checkpoint from {ckpt_path} ...")

    # 1) Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # 2) Print any key that involves "register_tokens" to confirm the checkpoint includes them
    print("\n[DEBUG] Checking if 'register_tokens' appears in checkpoint keys:")
    for k in checkpoint.keys():
        if "register_tokens" in k:
            print(f"  -> Found {k}, shape={checkpoint[k].shape}")

    # 3) Load the weights
    missing = model.load_state_dict(checkpoint, strict=False)
    print("load_state_dict report:", missing)

    # 4) Print out the shape of the model's register tokens *after* loading:
    #    This is the single most important check to confirm your model actually has the registers.
    #if model.register_tokens is not None:
    #    print("[DEBUG] model.register_tokens.shape:", model.register_tokens.shape)

    # 5) Freeze
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    return model

