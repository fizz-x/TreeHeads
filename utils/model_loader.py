import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange

import torch.nn as nn
import os
import json
from datetime import datetime

# Central hyperparameter config
# raytune / keras 
global_config = {
    'seed': 47,
    'patch_size': 32,
    'num_bands': 15,        # change based on input (13+1 for fmask, +1 for mask channel)
    'batch_size': 48,
    'learning_rate': 8e-4,
    'weight_decay': 2e-4,
    'scheduler_type': 'ReduceLROnPlateau',
    'scheduler_patience':15,
    'scheduler_factor':0.5,
    'scheduler_min_lr':1e-6,
    'early_stopping_patience': 75,
    'epochs': 500,
    'huber_delta': 1.35,
    'device':  'mps' if torch.backends.mps.is_available() else 'cpu'
}


class S2CanopyHeightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()               # (N, num_bands, 32, 32)
        self.y = torch.from_numpy(y).float()               # (N, 1, 32, 32), not need to unsqueeze here
        # NaN mask across bands → shape: (N, 1, 32, 32)
        # A pixel is valid if *not all bands* in X are NaN and y is not NaN
        x_valid = ~torch.isnan(self.X).any(dim=1, keepdim=True)  # (N, num_bands, 32, 32)
        y_valid = ~torch.isnan(self.y).any(dim=1, keepdim=True)  # (N, 1, 32, 32)
        self.mask = x_valid & y_valid

        # Replace NaNs in input and target with -1.0 wherever mask is False
        # self.X[self.mask.expand_as(self.X) == 0] = -1.0
        # self.y[self.mask.expand_as(self.y) == 0] = -7.0 #dont do this with y
        # Replace NaNs in input with -1.0 or some other value
        self.X[torch.isnan(self.X)] = -1.0 
        self.y[torch.isnan(self.y)] = -10.0 

    def __len__(self):
        return self.X.shape[0]
    # def __getitem__(self, idx):
    #     return self.X[idx], self.y[idx]
    def validshare(self):
        # Returns the fraction of valid pixels (mask == True) across the entire dataset
        total_pixels = self.mask.numel()
        valid_pixels = self.mask.sum().item()
        return valid_pixels / total_pixels
    def getRGB(self, idx):
        # Assumes bands 3, 2, 1 correspond to R, G, B respectively (0-indexed)
        RGB_indices = [10, 3, 0]  # basic setup 
        max_aux_layer = 4 
        #(seasons*quantiles*bands, H, W) - seasons can be 3 dim or 1, quantiles 3 or 1. make sure to select the right bands for RGB
        if self.X.shape[1] < 13:
            raise ValueError("Dataset does not have enough bands to extract RGB.")
        elif self.X.shape[1] <= 13 + max_aux_layer:
            RGB_indices = RGB_indices
        elif self.X.shape[1] <= 39 + max_aux_layer:
            RGB_indices = [x + 13 for x in [10, 3, 0]]  # Adjust if necessary based on actual band order
        elif self.X.shape[1] == 117 + max_aux_layer:
            RGB_indices = [x + 13*4 for x in [10, 3, 0]]  # Adjust if necessary based on actual band order
        rgb = self.X[idx, RGB_indices, :, :]  # (3, 32, 32)
        # Normalize to [0, 1] for visualization
        rgb_no_nan = torch.nan_to_num(rgb, nan=float('-inf'))
        rgb_min = torch.min(rgb_no_nan)
        rgb_max = torch.max(rgb_no_nan)
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)  # Avoid division by zero
        return rgb.to("cpu").numpy()
    
    def __getitem__(self, idx):
        x = self.X[idx]                         # (num_bands, 32, 32)
        m = self.mask[idx].float()             # (1, 32, 32)
        x_with_mask = torch.cat([x, m], dim=0) # (num_bands + 1, 32, 32)
        return x_with_mask, self.y[idx], self.mask[idx]  # keep mask for loss too

# train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
# test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
def build_unet(in_channels, out_channels):
    cin = in_channels + 1
    model = UNet(in_channels=cin, out_channels=out_channels)  # +1 in channel for nan-mask
    return model

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, dropout=0.2):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.decoder2(torch.cat([self.upconv2(bottleneck), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        return self.final(dec1)

# For debugging: set mask to all ones to match nn.HuberLoss behavior
def masked_huber_loss(pred, target, mask, delta=global_config['huber_delta']):
    """
    Computes the masked Huber loss (Smooth L1 loss).
    If mask is all ones, this should match nn.HuberLoss.
    """
    # Uncomment the next line to force mask to all ones for testing
    #mask = torch.ones_like(target)
    mask = mask.float()
    error = pred - target
    abs_error = torch.abs(error)

    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=pred.device))
    linear = abs_error - quadratic

    loss = 0.5 * quadratic**2 + delta * linear
    masked_loss = loss * mask

    return masked_loss.sum() / mask.sum().clamp(min=1)

def compute_losses(outputs, y_batch, mask, cfg):
    """
    Computes the total weighted loss for multi-output models with masking.
    """
    losses = []
    start_idx = 0
    for name, out_cfg in cfg["outputs"].items():
        out_ch = out_cfg.get("out_channels", 1)
        output_slice = outputs[:, start_idx:start_idx+out_ch, :, :]
        target_slice = y_batch[:, start_idx:start_idx+out_ch, :, :]
        mask_slice = mask[:, start_idx:start_idx+out_ch, :, :] if mask.shape[1] > 1 else mask
        weight = out_cfg.get("weight", 1.0)

        if out_cfg["loss"] == "huber":
            loss = masked_huber_loss(output_slice, target_slice, mask_slice)
        elif out_cfg["loss"] == "bce":
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                output_slice, target_slice, reduction='none'
            )
            masked_bce = bce * mask_slice
            loss = masked_bce.sum() / mask_slice.sum().clamp(min=1)
        elif out_cfg["loss"] == "crossentropy":
            ce = torch.nn.functional.cross_entropy(
                output_slice, target_slice.squeeze(1).long(), reduction='none'
            )
            masked_ce = ce * mask_slice.squeeze(1)
            loss = masked_ce.sum() / mask_slice.sum().clamp(min=1)
        elif out_cfg["loss"] == "kl":
            # Discretize the continuous target into bins and convert both prediction and target to probability distributions
            num_bins = out_cfg.get("num_bins", 25)
            min_val = out_cfg.get("min_val", 0.0)
            max_val = out_cfg.get("max_val", 50.0)

            def bin_targets(target, num_bins, min_val, max_val):
                # target: (N, 1, H, W)
                bins = torch.linspace(min_val, max_val, num_bins + 1, device=target.device)
                # Digitize returns bin indices (0 to num_bins-1)
                target_flat = target.view(-1)
                bin_indices = torch.bucketize(target_flat, bins[:-1], right=False)
                # One-hot encode
                target_binned = torch.zeros(target_flat.size(0), num_bins, device=target.device)
                target_binned.scatter_(1, bin_indices.unsqueeze(1).clamp(max=num_bins-1), 1)
                # Reshape back to (N, H, W, num_bins)
                target_binned = target_binned.view(*target.shape[:-1], target.shape[-2], target.shape[-1], num_bins)
                # Move bin dim to channel position: (N, num_bins, H, W)
                target_binned = target_binned.permute(0, 4, 2, 3, 1).squeeze(-1)
                return target_binned

            # Output slice should have shape (N, num_bins, H, W)
            # If output_slice is (N, 1, H, W), expand to num_bins via a head in the model
            if output_slice.shape[1] != num_bins:
                raise ValueError(f"Output channels ({output_slice.shape[1]}) do not match num_bins ({num_bins}).")

            target_binned = bin_targets(target_slice, num_bins, min_val, max_val)
            # KLDiv expects log-probabilities for input, probabilities for target
            kld = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(output_slice, dim=1),
                torch.nn.functional.softmax(target_binned, dim=1),
                reduction='none'
            )
            masked_kld = kld * mask_slice
            loss = masked_kld.sum() / mask_slice.sum().clamp(min=1)
            masked_kld = kld * mask_slice
            loss = masked_kld.sum() / mask_slice.sum().clamp(min=1)
        else:
            raise ValueError(f"Unknown loss {out_cfg['loss']}")
        losses.append(weight * loss)
        start_idx += out_ch        
    return sum(losses)

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, train_loader, val_loader, cfg, global_config):
    device = global_config['device']
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=global_config['learning_rate'], weight_decay=global_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=global_config['scheduler_patience'], factor=global_config['scheduler_factor'], min_lr=global_config['scheduler_min_lr'])
    early_stopping = EarlyStopping(patience=global_config['early_stopping_patience'], verbose=False)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_config['epochs'], eta_min=1e-6)

    # build loss functions from config
    # loss_fns = {}
    # for name, out_cfg in cfg["outputs"].items():
    #     if out_cfg["loss"] == "huber":
    #         loss_fns[name] = masked_huber_loss #torch.nn.SmoothL1Loss()
    #     elif out_cfg["loss"] == "bce":
    #         loss_fns[name] = torch.nn.BCEWithLogitsLoss()
    #     elif out_cfg["loss"] == "crossentropy":
    #         loss_fns[name] = torch.nn.CrossEntropyLoss()
    #     elif out_cfg["loss"] == "kl":
    #         loss_fns[name] = torch.nn.KLDivLoss(reduction="batchmean")
    #     else:
    #         raise ValueError(f"Unknown loss {out_cfg['loss']}")

    # best_val_loss = float('inf')

    logs = {'train_loss': [], 'val_loss': []}

    for epoch in trange(global_config['epochs'],desc="Epochs"):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch, mask in train_loader:
            X_batch, y_batch, mask = X_batch.to(device), y_batch.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = compute_losses(outputs, y_batch, mask, cfg)

            loss.backward()

            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        logs['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, mask in val_loader:
                X_batch, y_batch, mask = X_batch.to(device), y_batch.to(device), mask.to(device)
                outputs = model(X_batch)
                #loss = masked_huber_loss(outputs, y_batch, mask)
                loss = compute_losses(outputs, y_batch, mask, cfg)
                total_val_loss += loss.item() * X_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        logs['val_loss'].append(avg_val_loss)

        scheduler.step(avg_val_loss)
        early_stopping.step(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')

        #print(f"Epoch [{epoch+1}/{global_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, logs

def generate_run_id():
        
    today = datetime.now().strftime("%y%m%d")
    idx = 0

    # Check existing folders
    results_dir = "../results"
    if os.path.exists(results_dir):
        folders = [f for f in os.listdir(results_dir) if f.startswith(today)]
        if folders:
            # Extract indices from existing folders and get max
            indices = [int(f.split('_')[1]) for f in folders]
            idx = max(indices) + 1

    run_id = f"{today}_{idx}"
    return run_id

def save_results(model, val_loader, test_loader, normparams, logs, cfg, run_id=None):

    if run_id is None:
        run_id = generate_run_id()

    out_dir = os.path.join("../results", run_id, 'train', cfg['exp'])
    os.makedirs(out_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pth"))
    torch.save(model, os.path.join(out_dir, "model.pth"))

    # Save logs and cfg as JSON
    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump(logs, f)
    with open(os.path.join(out_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f)

    # Optionally, save predictions and targets for val/test sets
    preds_val, targets_val, maskval = get_predictions_and_targets(val_loader, model, normparams)
    preds_test, targets_test, masktest = get_predictions_and_targets(test_loader, model, normparams)

    rgb_test = np.stack([test_loader.dataset.getRGB(i) for i in range(len(test_loader.dataset))], axis=0)
    np.savez(os.path.join(out_dir, "test_rgb.npz"), rgb_test=rgb_test)

    # Zip predictions and targets for val/test sets and save as .npz files
    np.savez(os.path.join(out_dir, "val_preds_targets.npz"), preds_val=preds_val, targets_val=targets_val, maskval=maskval)
    np.savez(os.path.join(out_dir, "test_preds_targets.npz"), preds_test=preds_test, targets_test=targets_test, masktest=masktest)

    print("Results saved to:", out_dir)

def denorm_chm(chm, params):
    """
    Denormalizes the canopy height model (CHM) using provided normalization parameters.
    
    Args:
        chm (np.ndarray): Normalized CHM array.
        params (dict): Dictionary containing 'mean' and 'std' for denormalization.
    
    Returns:
        np.ndarray: Denormalized CHM array.
    """
    mean = params['mu']
    std = params['std']
    return chm * std + mean

def get_predictions_and_targets(loader, model, normparams):

    model.eval()
    device = next(model.parameters()).device
    preds = []
    targets = []
    masklayer = []
    with torch.no_grad():
        for X_batch, y_batch, mask in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu())
            targets.append(y_batch.cpu())
            masklayer.append(mask.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    masklayer = torch.cat(masklayer, dim=0).numpy()

    if preds.shape[1] > 1:
        # multi-output: only denorm the first channel (canopy height)
        preds[:, 0:1, :, :] = denorm_chm(preds[:, 0:1, :, :], normparams)
        targets[:, 0:1, :, :] = denorm_chm(targets[:, 0:1, :, :], normparams)
    else:
        preds = denorm_chm(preds, normparams)
        targets = denorm_chm(targets, normparams)

    return preds, targets, masklayer

def load_results(exp_dir, run_id=None):
    """
    Loads model weights, logs, and cfg from the experiment results folder.
    """
    if run_id is not None:
        out_dir = os.path.join("../results", run_id, 'train', exp_dir)
    else:
        out_dir = os.path.join("../results/train", exp_dir)

    model_weights = torch.load(os.path.join(out_dir, "model_weights.pth"))
    with open(os.path.join(out_dir, "logs.json"), "r") as f:
        logs = json.load(f)
    with open(os.path.join(out_dir, "cfg.json"), "r") as f:
        cfg = json.load(f)
    return model_weights, logs, cfg

def load_np_stacks(exp_dir, run_id=None):
    """
    Loads prediction and target numpy arrays from the experiment results folder.
    Returns preds_val, targets_val, preds_test, targets_test as numpy arrays.
    """
    out_dir = os.path.join("../results/train", exp_dir)
    if run_id is not None:
        out_dir = os.path.join("../results", run_id, 'train', exp_dir)

    val_npz = np.load(os.path.join(out_dir, "val_preds_targets.npz"))
    test_npz = np.load(os.path.join(out_dir, "test_preds_targets.npz"))
    rgb_npz = np.load(os.path.join(out_dir, "test_rgb.npz")) if "test_rgb.npz" in os.listdir(out_dir) else None
    if rgb_npz is None:
        rgb_test = None
    else:
        rgb_test = rgb_npz["rgb_test"]
    preds_val = val_npz["preds_val"]
    targets_val = val_npz["targets_val"]
    preds_test = test_npz["preds_test"]
    targets_test = test_npz["targets_test"]
    maskval = val_npz["maskval"]
    masktest = test_npz["masktest"]
    return preds_val, targets_val, preds_test, targets_test, maskval, masktest, rgb_test

