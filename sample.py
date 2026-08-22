import argparse
import os
import sys
import time
import warnings

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from tqdm import tqdm

from mask_process import process

sys.path.append(os.path.dirname(__file__))
from utils import (
    save_image,
    load_config,
    load_checkpoint,
    create_model,
    sample_batch,
    create_dataloader,
    normalize_zero_to_one,
)

warnings.filterwarnings("ignore")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified inference script for the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/PanNuke.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["image", "mask"],
        help="Sampling task: image or mask.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="PanNuke",
        help="Data type used by the dataset loader.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint directory. If omitted, uses checkpoint_dir/latest.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to save. If None, save all samples.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of inference steps during sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (overrides config device).",
    )
    return parser.parse_args(argv)


def save_mask(mask_tensor: torch.Tensor, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.detach().cpu().numpy()
    else:
        mask_np = np.array(mask_tensor)

    if mask_np.ndim == 3:
        mask_np = np.transpose(mask_np, (1, 2, 0))
    if mask_np.ndim != 3:
        raise ValueError(f"Expected mask tensor shape (H, W, 5) or (5, H, W), got {mask_np.shape}")

    # A nonzero channel stores the instance ID for its corresponding cell type.
    type_raw = np.argmax(mask_np, axis=-1)
    background = mask_np.max(axis=-1) == 0
    type_map = type_raw + 1
    type_map[background] = 0

    instance_map = np.zeros(type_map.shape, dtype=np.int32)
    for t in range(mask_np.shape[2]):
        channel = mask_np[:, :, t].astype(np.int32)
        if np.any(channel > 0):
            instance_map[channel > 0] = channel[channel > 0] + (t + 1) * 100000

    base_path = os.path.splitext(out_path)[0]
    sio.savemat(base_path + ".mat", {"mask": mask_np, "type_map": type_map, "instance_map": instance_map})

    type_colors = np.array(
        [
            [0, 0, 0],
            [65, 105, 225],
            [219, 112, 147],
            [244, 164, 96],
            [0, 201, 167],
            [238, 130, 238],
            [104, 118, 132],
            [128, 128, 0],
            [255, 241, 67],
            [190, 190, 190],
            [253, 94, 83],
            [126, 220, 48],
            [42, 164, 220],
            [64, 176, 128],
        ],
        dtype=np.uint8,
    )
    type_img = type_colors[type_map]
    Image.fromarray(type_img).save(base_path + "_type.png")

    instance_img = np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]
    rng = np.random.default_rng(42)
    colors = {int(uid): tuple(rng.integers(0, 256, size=3).astype(np.uint8).tolist()) for uid in unique_ids}
    for uid, color in colors.items():
        instance_img[instance_map == uid] = color
    Image.fromarray(instance_img).save(base_path + "_instance.png")


def build_output_dir(config: dict, args: argparse.Namespace):
    config_name = os.path.basename(args.config_path).replace(".yaml", "")
    output_name = f"fm_epoch_{args.start_epoch}_num_inference_{args.num_inference_steps}_{config_name}"
    output_dir = os.path.join(config["train_args"]["checkpoint_dir"], output_name)
    if os.path.exists(output_dir):
        print(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def post_process_mask(final_img: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    pred_map = final_img.cpu().numpy().transpose(1, 2, 0)
    mask_point = masks.cpu().numpy()
    gen_img = process(pred_map, nr_types=3, point=mask_point)

    for cell_type in range(gen_img.shape[-1]):
        mask_point_ = mask_point
        cell_id = np.unique(mask_point_[cell_type, :, :])[1:]
        for vv in cell_id:
            for kk in range(gen_img.shape[-1]):
                gen_img[:, :, cell_type] = np.where(
                    gen_img[:, :, kk] == vv, gen_img[:, :, kk], gen_img[:, :, cell_type]
                )
            for dd in range(gen_img.shape[-1]):
                if dd != cell_type:
                    gen_img[:, :, dd] = np.where(gen_img[:, :, cell_type] == vv, 0, gen_img[:, :, dd])

    return torch.from_numpy(gen_img.transpose(2, 0, 1))


def run_sampling(argv=None):
    args = parse_args(argv)
    config = load_config(args.config_path)

    task = args.task if args.task is not None else config["data_args"].get("task", "image")
    data_path = config["data_args"].get("data_path", "/home/PanNuke/")
    data_type = args.data_type

    device_name = args.device if args.device is not None else config["train_args"].get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print("Inference device:", device)

    model = create_model(config["model_args"], device=device, task=task)

    model_path = args.model_path if args.model_path is not None else os.path.join(config["train_args"]["checkpoint_dir"], "latest")
    start_epoch, _ = load_checkpoint(
        model=model,
        optimizer=None,
        checkpoint_dir=model_path,
        device=device,
        valid_only=True,
    )
    args.start_epoch = start_epoch

    output_dir = build_output_dir(config, args)
    print(f"Saving generated samples to: {output_dir}")

    val_loader = create_dataloader(
        data_path=data_path,
        task=task,
        split="test",
        batch_size=config["train_args"]["batch_size"],
        shuffle=False,
        data_type=data_type,
    )

    dataset_size = len(val_loader.dataset)
    num_samples = dataset_size if args.num_samples is None else args.num_samples
    print(f"Number of samples to save: {num_samples}")

    step_size = 1.0 / args.num_inference_steps
    print(f"Step size: {step_size}")

    solver_config = {
        "method": config.get("solver_args", {}).get("method", "midpoint"),
        "step_size": step_size,
        "time_points": config.get("solver_args", {}).get("time_points", 10),
    }

    count = 0
    for batch in tqdm(val_loader, desc="Generating Samples"):
        if count >= num_samples:
            break

        final_imgs = sample_batch(
            model=model,
            solver_config=solver_config,
            batch=batch,
            device=device,
            class_conditioning=config["model_args"].get("class_conditioning", False),
            mask_conditioning=config["model_args"].get("mask_conditioning", False),
        )

        images = batch.get("images")
        masks = batch.get("masks")
        classes = batch.get("classes")
        file_names = batch.get("file_name")

        batch_size = final_imgs.shape[0]
        for i in range(batch_size):
            if count >= num_samples:
                break

            name = os.path.basename(file_names[i])
            if task == "image":
                name = name.replace(".jpg", ".png").replace(".jpeg", ".png")
                gen_img = normalize_zero_to_one(final_imgs[i])
                save_image(gen_img, os.path.join(output_dir, name))
            else:
                name = name.replace(".mat", ".png")
                mask_pred = post_process_mask(final_imgs[i], masks[i])
                save_mask(mask_pred, os.path.join(output_dir, name))

            count += 1

    print(f"Saved {count} samples to {output_dir}")


def main(argv=None):
    run_sampling(argv)


if __name__ == "__main__":
    main()
