import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from utils import load_config, create_dataloader, save_checkpoint, load_checkpoint, create_model, validate_and_save_samples


def main():
    # Parse arguments and load config
    parser = argparse.ArgumentParser(description="Train our MMCFM-PNDA model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/PanNuke.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path)

    # Read core settings from config
    num_epochs = config["train_args"]["num_epochs"]
    num_val_samples = config["train_args"].get("num_val_samples", 5)
    batch_size = config["train_args"]["batch_size"]
    lr = config["train_args"]["lr"]
    print_every = config["train_args"].get("print_every", 1)
    val_freq = config["train_args"].get("val_freq", 5)
    root_ckpt_dir = config["train_args"]["checkpoint_dir"]

    # Decide which device to use
    device = (
        torch.device(config["train_args"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    # Model configuration flags
    mask_conditioning = config["general_args"]["mask_conditioning"]
    class_conditioning = config["general_args"]["class_conditioning"]

    # Build model
    model = create_model(config["model_args"], device=device)

    # Prepare data
    data_config = config["data_args"]

    train_loader = create_dataloader(
        data_path=data_config["data_path"],
        task=data_config["task"],
        split='train',
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        data_path=data_config["data_path"],
        task=data_config["task"],
        split='valid',
        batch_size=batch_size,
        shuffle=False,
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Load the latest checkpoint if available
    if os.path.exists(os.path.join(root_ckpt_dir, "latest")):
        
        latest_ckpt_dir = os.path.join(root_ckpt_dir, "latest")
        print(f"Loading checkpoint from {latest_ckpt_dir}...")
        start_epoch, loaded_config = load_checkpoint(model, optimizer, checkpoint_dir=latest_ckpt_dir, device=device, valid_only=False)

            
    # Define path object (scheduler included)
    path = AffineProbPath(scheduler=CondOTScheduler())

    solver_config = config["solver_args"]
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
     
        # Use tqdm for the train loader to get a per-batch progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            
            im_batch = batch["images"].to(device)
            mask_batch = batch["masks"].to(device) #nuclei multi-class label as condition
            classes_batch = batch["classes"]if class_conditioning else None
            
            classes_batch[0] = classes_batch[0].to(device) #### SSL_Embeds
            classes_batch[1] = classes_batch[1].to(device) #### Text_Embeds

            # Sample random initial noise, and random t
            x_0 = torch.randn_like(im_batch)
            t = torch.rand(im_batch.shape[0], device=device)

            # Sample the path from x_0 to x_batch
            sample_info = path.sample(t=t, x_0=x_0, x_1=im_batch)

            # Predict velocity and compute loss
            v_pred = model(
                x=sample_info.x_t,
                t=sample_info.t,
                masks=mask_batch,
                cond=classes_batch,
            )
            loss = F.mse_loss(v_pred, sample_info.dx_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Logging
        avg_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s")

        # Validation & checkpoint saving
        if (epoch + 1) % val_freq == 0:
            epoch_ckpt_dir = os.path.join(root_ckpt_dir, f"epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch + 1, config, epoch_ckpt_dir)
            save_checkpoint(model, optimizer, epoch + 1, config, latest_ckpt_dir)

            # Validation
            validate_and_save_samples(
                model=model,
                val_loader=val_loader,
                device=device,
                checkpoint_dir=epoch_ckpt_dir,
                epoch=epoch + 1,
                solver_config=solver_config,
                max_samples=num_val_samples,
                class_map=None,
                mask_conditioning=mask_conditioning,
                class_conditioning=class_conditioning,
            )

    print("Training complete!")


if __name__ == "__main__":
    main()
