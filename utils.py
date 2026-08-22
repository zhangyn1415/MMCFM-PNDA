import os
import yaml
import torch

import matplotlib.pyplot as plt
from scipy.ndimage import measurements
from skimage import morphology as morph  
from torch.utils.data import DataLoader, Dataset
import json, glob, numpy as np
from PIL import Image
import scipy.io as sio
import torch.nn as  nn
from models.control_net import ControlNet
from models.diffusion_net import DiffusionModelUNet
from tqdm import  tqdm
from flow_matching.solver import ODESolver


def load_config(config_path: str = "*.yaml"):
    """
    Loads a YAML config file from the given path.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

####### nuclei multi-class label - pathology image pair dataset ############
class PathoDataset(Dataset):
    def __init__(self, data_type='PanNuke', path="", split='train'):

        self.data_type = data_type
        self.split = split

        self.data_file, self.mask_file, self.ssl_file, self.text_file = self.load_mask_img_pairs(f"{path}/{split}/") 

    def load_mask_img_pairs(self, data_dir):
    
        filenames = []
        masknames = []
        featnames = []
        textualnames = []

        SSL_Embed = torch.load(f"{data_dir}/ssl.pt")
        Text_embed = torch.load(f"{data_dir}/text.pt")
        
        for idx_k, img in enumerate(sorted(glob.glob(f"{data_dir}/imgs/*.jpg"))):

            mask = img.replace("imgs", "masks").replace(".jpg", ".mat")
            filenames.append(img)
            masknames.append(mask)    
            featnames.append(SSL_Embed[idx_k])
            idx_k = img[img.rfind("/")+1:-4]
            textualnames.append(Text_embed[idx_k])
            

        return filenames, masknames, featnames, textualnames 


    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        
        item = self.data_file[idx]
        mask = self.mask_file[idx]
        prompt = self.ssl_file[idx].numpy()
        text_embed = self.text_file[idx]

        img = np.array( Image.open(item).convert('RGB') )
        # Normalize target images to [-1, 1].
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.copy(img)

        if self.data_type == "PanNuke":
            raw_dict = sio.loadmat(mask)
            maps = raw_dict["inst_map"]   
            cond = np.where(maps[:,:,:-1] > 0, 1, 0)         
 
        if self.split == "train":                    
            # Augmentation (random horizontal/vertical flip)
            if np.random.rand() > 0.5: # horizontal
                img = np.flip(img, axis=0)
                cond = np.flip(cond, axis=0)
            if np.random.rand() > 0.5: # vertical
                img = np.flip(img, axis=1)
                cond = np.flip(cond, axis=1)

        img = np.copy(img).transpose(2,0,1)
        cond = np.copy(cond).transpose(2,0,1).astype(np.float32)
        
        return dict(images=img, masks=cond, classes=[text_embed.astype(np.float32), prompt.astype(np.float32)], file_name=item )


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def get_hv(ann):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = fixed_ann = crop_ann = ann.copy()  # instance ID map
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map]) # (h, w, 2)
    return hv_map

####### nuclei center point -  multi-class label dataset ############
class NucleiMaskDataset(Dataset):
    def __init__(self, data_type='PanNuke', path="", split='train'):

        self.data_type = data_type
        self.split = split

        self.mask_file, self.text_file = self.load_nuclei_masks(f"{path}/{split}/")

    def load_nuclei_masks(self, data_dir):

        masknames = []
        featnames = []

        Text_Embeds = torch.load(f"{data_dir}/text.pt")
        
        for idx_k, mask in enumerate(sorted(glob.glob(data_dir+"masks/*.mat"))):

            masknames.append(mask)  
            idx_k = mask[mask.rfind("/")+1:-4]  
            featnames.append(Text_Embeds[idx_k])

        return masknames, featnames 

    def __len__(self):
        return len(self.mask_file)

    def __getitem__(self, idx):
        mask = self.mask_file[idx]
        prompt = self.text_file[idx]

        raw_dict = sio.loadmat(mask)
        ## nuclei center point map
        mask_point = raw_dict["point"]
                    
        if self.data_type == "PanNuke":
            
            maps = raw_dict["inst_map"]
            inst_map = np.max(maps[:,:,:-1], axis=-1).astype(np.int32)
            cond = np.where(maps[:,:,:]>0, 1, 0)
                                                
        # Augmentation (random horizontal/vertical flip)
        if self.split == "train":
            if np.random.rand() > 0.5: # horizontal
                cond = np.flip(cond, axis=0)
                mask_point = np.flip(mask_point, axis=0)
                inst_map = np.flip(inst_map, axis=0)
            if np.random.rand() > 0.5: # vertical
                cond = np.flip(cond, axis=1)
                mask_point = np.flip(mask_point, axis=1)
                inst_map = np.flip(inst_map, axis=1)
            if np.random.rand() > 0.5: # transpose
                cond = cond.transpose(1,0,2)
                mask_point = mask_point.transpose(1,0,2)
                inst_map = inst_map.transpose(1,0)
    
        hv_ = get_hv( inst_map )
        cond = np.concatenate([cond, hv_], axis=-1)
        np.clip(mask_point, 0, 1)
        
        cond = np.copy(cond).transpose(2,0,1).astype(np.float32)
        mask_point = np.copy(mask_point).transpose(2,0,1).astype(np.float32)
                            
        return dict(images=cond, masks=mask_point, classes=prompt.astype(np.float32), file_name=mask)


def normalize_zero_to_one(tensor: torch.Tensor):
    """
    Normalizes a tensor to the range [0, 1].
    """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_minusone_to_one(tensor: torch.Tensor):
    """
    Normalizes a tensor to the range [-1, 1].
    """
    return 2 * (tensor - tensor.min()) / (tensor.max() - tensor.min()) - 1


def move_to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        return [item.to(device) if torch.is_tensor(item) else item for item in obj]
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def create_dataloader(data_path="/home/PanNuke/", task='image', split=None, batch_size=8, shuffle=True, data_type='PanNuke'):
    """
    Creates a DataLoader from Images, and optionally Masks and classes if given.
    """
    if task == 'image':
        dataset = PathoDataset(split=split, path=data_path, data_type=data_type)
    else:
        dataset = NucleiMaskDataset(split=split, path=data_path,data_type=data_type)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


###############################################################################
# Checkpointing
###############################################################################
def save_checkpoint(
    model,
    optimizer,
    epoch,
    config,
    checkpoint_dir: str,
    scheduler=None,
):
    """
    Saves model/optimizer states + config to "checkpoint.pth" inside 'checkpoint_dir'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved -> {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_dir,
    device,
    valid_only=False,
    scheduler=None,
):
    """
    Loads model/optimizer states + config from 'checkpoint.pth' in 'checkpoint_dir', if present.
    Returns (epoch, config).

    If 'valid_only' is True, optimizer state is not loaded (for pure inference).
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, None

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
                        
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if not valid_only and optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and not valid_only:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    config_loaded = checkpoint.get("config", None)

    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, config_loaded


###############################################################################
# Image Saving
###############################################################################
def save_image(img_tensor, out_path):
    """
    Saves a single 2D image (assumed shape [1, H, W] or [H, W]) as PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # remove batch/channel dims if present
    if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.squeeze(0)
    
    Image.fromarray( (img_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1,2,0) ).save(out_path)
    # plt.figure()
    # plt.imshow(img_tensor.permute(1,2,0).detach().cpu().numpy())
    # plt.axis("off")
    # plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    # plt.close()

class MergedModel(nn.Module):
    """
    Merged model that wraps a UNet and an optional ControlNet.
    Takes in x, time in [0,1], and (optionally) a ControlNet condition.
    """

    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep

        # If controlnet is None, we won't do anything special in forward.
        self.has_controlnet = controlnet is not None
        self.has_conditioning = unet.with_conditioning
            

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: tuple[torch.Tensor, torch.Tensor] = None,
        masks: torch.Tensor = None,
    ):
        """
        Args:
            x: input image tensor [B, C, H, W].
            t: timesteps in [0,1], will be scaled to [0, max_timestep - 1].
            cond: [B,1 , conditions_dim].
            masks: [B, C, H, W] masks for conditioning.

        Returns:
            The network output (e.g. velocity, noise, or predicted epsilon).
        """
        # Scale continuous t -> discrete timesteps(If you dont want to change the embedding function in the UNet)
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        # If t is scalar, expand to batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if self.has_controlnet:
            # cond is expected to be a ControlNet conditioning, e.g. mask
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=x, timesteps=t, controlnet_cond=masks, context=cond
            )
            output = self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            # If no ControlNet, cond might be cross-attention or None
            output = self.unet(x=x, timesteps=t, context=cond)

        return output


def create_model(model_config: dict, device: torch.device = None, task: str = "image") -> MergedModel:

    """Create and return a `MergedModel` (UNet + optional ControlNet).

    Args:
        model_config: configuration dictionary from YAML.
        device: device to move the model to.
        task: either 'image' or 'mask' — used to adjust conditioning behavior.

    The function remains backward compatible with the previous signature.
    """
    mc = model_config.copy()

    mask_conditioning = mc.pop("mask_conditioning", False)
    class_conditioning = mc.pop("class_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", [8])
    cond_embed_in_channels = mc.pop("conditioning_embedding_in_channels", [8])

    # Build UNet
    unet = DiffusionModelUNet(**mc)

    # If mask conditioning is requested, build a ControlNet that shares weights
    controlnet = None
    if mask_conditioning:
        # ControlNet expects a slightly different config shape for conditioning embeddings.
        tmp_mc = mc.copy()
        # remove out_channels if present to avoid duplicate arg issues
        tmp_mc.pop("out_channels", None)
        # Allow overriding conditioning in-channels per task if needed
        if task == "mask":
            tmp_mc.update({"conditioning_embedding_in_channels": cond_embed_in_channels})
        else:
            tmp_mc.update({"conditioning_embedding_in_channels": cond_embed_in_channels})
        controlnet = ControlNet(**tmp_mc, conditioning_embedding_num_channels=cond_embed_channels)
        # initialize controlnet from unet weights where possible
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)

    return model.to(device)


def sample_with_solver(
    model,
    x_init,
    solver_config,
    cond=None,
    masks=None,
):
    """
    Uses ODESolver (flow-matching) to sample from x_init -> final output.
    solver_config might contain keys:
        {
          "method": "midpoint"/"rk4"/etc.,
          "step_size": float,
          "time_points": int,
        }

    Returns either the full trajectory [time_points, B, C, H, W] if return_intermediates=True
    or just the final state [B, C, H, W].
    """
    solver = ODESolver(velocity_model=model)

    time_points = solver_config.get("time_points", 10)
    T = torch.linspace(0, 1, time_points, device=x_init.device)

    method = solver_config.get("method", "midpoint")
    step_size = solver_config.get("step_size", 0.02)

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        method=method,
        step_size=step_size,
        return_intermediates=True,
        cond=cond,
        masks=masks,
    )
    return sol


def plot_solver_steps(sol, im_batch, mask_batch, class_batch, class_map, outdir, max_plot=4):
    if sol.dim() != 5:  # No intermediates to plot
        return
    n_samples = min(sol.shape[1], max_plot)
    n_steps = sol.shape[0]
    if mask_batch is not None:
        fig, axes = plt.subplots(n_samples, n_steps + 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(20, 8))
    if n_samples == 1:
        axes = [axes]
    
    def normalize_image(img_tensor):
        """将张量归一化到 [0, 1] 范围"""
        img = img_tensor.detach().cpu().numpy().squeeze()
        # 方法A：线性归一化到 [0, 1]
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 1e-8:  # 避免除零
            img = (img - img_min) / (img_max - img_min)
        # 方法B：如果知道理论范围（如 [-1, 1]），可以直接转换
        # img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        return img
    
    for i in range(n_samples):
        for t in range(n_steps):
            img_normalized = normalize_image(sol[t, i].permute(1,2,0))
            axes[i][t].imshow(img_normalized, cmap="gray")  # 使用归一化后的图像
            axes[i][t].axis("off")
            if i == 0:
                axes[i][t].set_title(f"Step {t}")
        col = n_steps
        if mask_batch is not None:
            mask_img = normalize_image(mask_batch[i].permute(1,2,0))
            axes[i][col].imshow(mask_img, cmap="gray")
            axes[i][col].axis("off")
            if i == 0:
                axes[i][col].set_title("Mask")
            col += 1
        real_img = normalize_image(im_batch[i].permute(1,2,0))
        axes[i][col].imshow(real_img, cmap="gray")
        axes[i][col].axis("off")
        if i == 0:
            axes[i][col].set_title("Real")
        if class_map and class_batch is not None:
            idx = class_batch[i].argmax().item()
            cls = class_map[idx] if idx < len(class_map) else str(idx)
            axes[i][col].text(
                0.5,
                -0.15,
                f"Class: {cls}",
                ha="center",
                va="top",
                transform=axes[i][col].transAxes,
                color="red",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "solver_steps.png"), bbox_inches="tight", pad_inches=0)
    plt.close()


COLORS = torch.tensor(
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
    ]
)


def validate_and_save(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_dir: str,
    epoch: int,
    solver_config: dict,
    max_samples=16,
    mask_conditioning=True,
    class_conditioning=False,
    task="image",
):
    model.eval()
    outdir = os.path.join(checkpoint_dir, f"val_samples_epoch_{epoch}")
    os.makedirs(outdir, exist_ok=True)
    count, step_plot_done = 0, False

    for batch in tqdm(val_loader, desc="Validating"):
        imgs = batch["images"].to(device)
        masks = batch["masks"].to(device) if mask_conditioning else None
        cond = batch["classes"] if class_conditioning else None
        # For mask task, context[0] (e.g. SSL) may be unused — avoid moving it unnecessarily
        if cond is not None and task == "mask":
            # keep text embedding  and delete SSL embedding
            text_emb = cond
            text_emb = move_to_device(text_emb, device)
            cond = [text_emb, None]
        else:
            cond = move_to_device(cond, device)

        x_init = torch.randn_like(imgs)
        sol = sample_with_solver(
            model,
            x_init,
            solver_config,
            cond=cond,
            masks=masks,
        )
        final_imgs = sol[-1] if sol.dim() == 5 else sol

        for i in range(final_imgs.size(0)):
            if count >= max_samples:
                break

            sdir = os.path.join(outdir, f"sample_{count+1:03d}")
            os.makedirs(sdir, exist_ok=True)

            if task == "mask":
                gen_img = torch.argmax(final_imgs[i][:-2, :, :], dim=0).detach().cpu()
                real_img = torch.argmax(imgs[i][:-2, :, :], dim=0).detach().cpu()
                gen_img = COLORS[gen_img, :] / 255.0
                real_img = COLORS[real_img, :] / 255.0
                save_image(gen_img.permute(2, 0, 1), os.path.join(sdir, "gen.png"))
                save_image(real_img.permute(2, 0, 1), os.path.join(sdir, "real.png"))
            else:
                gen_img = normalize_zero_to_one(final_imgs[i])
                real_img = normalize_zero_to_one(imgs[i])
                save_image(gen_img, os.path.join(sdir, "gen.png"))
                save_image(real_img, os.path.join(sdir, "real.png"))

            count += 1

        if not step_plot_done and task == "image":
            plot_solver_steps(sol, imgs, None, None, None, outdir)
            step_plot_done = True

        if count >= max_samples:
            break

    print(f"Validation samples saved in: {outdir}")


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    solver_config: dict,
    batch: torch.Tensor,
    device: torch.device,
    class_conditioning: bool = False,
    mask_conditioning: bool = False,
):
    model.eval()
    imgs = batch["images"].to(device)
    cond = batch.get("classes") if class_conditioning else None
    masks = batch.get("masks") if mask_conditioning else None

    if isinstance(cond, (list, tuple)):
        cond = [item.to(device) if torch.is_tensor(item) else item for item in cond]
    elif torch.is_tensor(cond):
        cond = [cond.to(device), None]

    if torch.is_tensor(masks):
        masks = masks.to(device)

    x_init = torch.randn_like(imgs)
    sol = sample_with_solver(
        model=model, solver_config=solver_config, x_init=x_init, cond=cond, masks=masks
    )
    final_imgs = sol[-1] if sol.dim() == 5 else sol
    return final_imgs