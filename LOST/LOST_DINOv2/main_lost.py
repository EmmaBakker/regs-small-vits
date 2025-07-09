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

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model 
from custom_datasets import Dataset, bbox_iou, ImageDataset
from object_discovery import lost, detect_box, dino_seg
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion

def main():
    parser = argparse.ArgumentParser("LOST on DINOv2 L/14 with or without registers.")

    parser.add_argument(
        "--arch",
        default="dinov2_vitl14",
        type=str,
        choices=["dinov2_vitl14", "dinov2_vitl14_reg"],
        help="DINOv2 L/14 arch: no registers or with registers"
    )
    parser.add_argument("--patch_size", default=14, type=int,
                        help="Patch size for DINOv2 L/14 is 14")
    parser.add_argument("--dataset", default=None, type=str,
                        choices=[None, "VOC07", "VOC12", "COCO20k"],
                        help="Dataset or None if single-image mode")
    parser.add_argument("--set", default="train", type=str,
                        choices=["val", "train", "trainval", "test"],
                        help="Which subset of dataset to use")
    parser.add_argument("--image_path", default=None, type=str,
                        help="Path to a single image (overrides dataset usage)")
    parser.add_argument("--output_dir", default="outputs", type=str,
                        help="Where to store results")
    parser.add_argument("--which_features", default="k", type=str,
                        choices=["k", "q", "v"],
                        help="Which of the qkv to use for LOST")
    parser.add_argument("--k_patches", default=100, type=int,
                        help="Number of patches with the lowest degree for seed expansion")
    parser.add_argument("--visualize", type=str,
                        choices=["fms", "seed_expansion", "pred", None],
                        default=None,
                        help="Type of LOST visualization (optional)")
    parser.add_argument("--no_evaluation", action="store_true",
                        help="Skip corloc evaluation (e.g. single image use)")
    parser.add_argument("--save_predictions", default=True, type=bool,
                        help="Save bounding boxes in preds.pkl file")
    parser.add_argument("--no_hard", action="store_true",
                        help="If set, remove images that only contain truncated/difficult objects (VOC).")

    args = parser.parse_args()

    # Single image => ignore dataset
    if args.image_path is not None:
        args.dataset = None
        args.no_evaluation = True
        args.save_predictions = False

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    if args.dataset is None:
        dataset = ImageDataset(args.image_path)
    else:
        dataset = Dataset(args.dataset, args.set, remove_hards=args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.arch, args.patch_size, resnet_dilate=-1, device=device)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    print(f"Model loaded with architecture: {args.arch}")
    if args.arch == "dinov2_vitl14_reg":
        print(f"Number of register tokens: {model.num_register_tokens}")
        print(f"Initial register token values: {model.register_tokens.mean(dim=-1)}")

    # Make output dir
    if args.dataset is not None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Experiment name
    exp_name = f"LOST-{args.arch}-p{args.patch_size}-{args.which_features}"
    print(f"Running LOST on arch={args.arch} patch={args.patch_size} -> {exp_name}")

    # Visualization
    if args.visualize:
        vis_folder = os.path.join(args.output_dir, "visualizations", exp_name)
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images

    n_data = len(dataset.dataloader)
    corloc = np.zeros(n_data)
    preds_dict = {}
    cnt = 0

    pbar = tqdm(dataset.dataloader)
    for im_id, sample in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = sample[0]  # shape (C,H,W)
        init_size = img.shape
        im_name = dataset.get_image_name(sample[1])
        if im_name is None:
            continue

        # Pad to multiple of patch_size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, :img.shape[1], :img.shape[2]] = img
        img = paded.to(device)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # Debugging: Log image dimensions and feature map sizes
        #print(f"Image dimensions: {img.shape}")
        #print(f"Feature map size: ({w_featmap}, {h_featmap})")

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(sample[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue
        
        # ------------ EXTRACT FEATURES -------------------------------------------

        # ------------ FORWARD PASS -------------------------------------------

        # Hook on final block's qkv
        feat_out = {}
        def hook_fn(module, inp, out):
            feat_out["qkv"] = out
        model.blocks[-1].attn.qkv.register_forward_hook(hook_fn)

        with torch.no_grad():
            # get last attentions
            attentions = model.get_last_selfattention(img[None, :, :, :])
        
        # Scaling factor
        scales = [args.patch_size, args.patch_size]

        # Debugging: Log attention maps
        #print(f"Attention map shape: {attentions.shape}")

        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens
        # Extract the qkv features of the last attention layer
        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )# parse qkv

        # Debugging: Log qkv dimensions
        #print(f"QKV shape: {qkv.shape}")

        q, k, v = qkv[0], qkv[1], qkv[2]

        # flatten heads => (B, N, nHeads * dim_per_head)
        q = q.transpose(1,2).reshape(nb_im, nb_tokens, -1)
        k = k.transpose(1,2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1,2).reshape(nb_im, nb_tokens, -1)

        # Debugging: Log key features
        #print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # skip CLS & register tokens => feats[:, (1 + n_reg) :, :]
        n_reg = model.num_register_tokens
        if args.which_features == "k":
            feats = k[:, 1 + n_reg :, :]
        elif args.which_features == "q":
            feats = q[:, 1 + n_reg :, :]
        else:  # "v"
            feats = v[:, 1 + n_reg :, :]

        # Debugging: Log extracted features
        #print(f"Extracted features shape: {feats.shape}")

        # ------------ Apply LOST -------------------------------------------
        pred, A, scores, seed = lost(
            feats,
            [w_featmap, h_featmap],
            scales,
            init_size,
            k_patches=args.k_patches
        )

        # Debugging: Log LOST outputs
        #print(f"Predicted box: {pred}, Seed: {seed}, Scores: {scores[:10]}")

        # ------------ Visualizations -------------------------------------------
        if args.visualize == "pred":
            raw_img = dataset.load_image(im_name)
            visualize_predictions(raw_img, pred, seed,
                                  [args.patch_size, args.patch_size],
                                  [w_featmap, h_featmap],
                                  vis_folder, im_name)
        elif args.visualize == "fms":
            visualize_fms(A.detach().cpu().numpy(), seed, scores,
                          [w_featmap, h_featmap],
                          [args.patch_size, args.patch_size],
                          vis_folder, im_name)
        elif args.visualize == "seed_expansion":
            raw_img = dataset.load_image(im_name)
            pred_seed, _ = detect_box(A[seed,:], seed,
                                      [w_featmap, h_featmap],
                                      [args.patch_size, args.patch_size],
                                      init_size[1:])
            visualize_seed_expansion(raw_img, pred, seed, pred_seed,
                                     [args.patch_size, args.patch_size],
                                     [w_featmap, h_featmap],
                                     vis_folder, im_name)

        # store
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    # save predictions
    if args.save_predictions:
        folder = os.path.join(args.output_dir, exp_name)
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("LOST predictions saved at", filename)

    # final corloc
    if not args.no_evaluation and cnt > 0:
        corloc_val = 100.0 * corloc.sum() / cnt
        print(f"CorLoc on {cnt} images: {corloc_val:.2f}% ({int(corloc.sum())}/{cnt})")
        res_file = os.path.join(folder, "results.txt")
        with open(res_file, "w") as f:
            f.write(f"corloc: {corloc_val:.2f}\n")
        print("Results file saved at:", res_file)

if __name__ == "__main__":
    main()
