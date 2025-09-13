import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange

import torch.nn as nn
import os
import json

# Central hyperparameter config
# raytune / keras 
global_config = {
    'patch_size': 32,
    'num_bands': 15,        # change based on input (13+1 for fmask, +1 for mask channel)
    'batch_size': 64,
    'learning_rate': 5e-4,
    'weight_decay': 5e-4,
    #'momentum': 0.9,
    'epochs': 150,
    'huber_delta': 0.8,
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

        # Replace NaNs in input with -1.0 or some other value
        self.X[torch.isnan(self.X)] = -1.0 
        self.y[torch.isnan(self.y)] = -1.0 
    def __len__(self):
        return self.X.shape[0]
    # def __getitem__(self, idx):
    #     return self.X[idx], self.y[idx]
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
            kld = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(output_slice, dim=1),
                torch.nn.functional.softmax(target_slice, dim=1),
                reduction='none'
            )
            masked_kld = kld * mask_slice
            loss = masked_kld.sum() / mask_slice.sum().clamp(min=1)
        else:
            raise ValueError(f"Unknown loss {out_cfg['loss']}")
        losses.append(weight * loss)
        start_idx += out_ch        
    return sum(losses)



def train_model(model, train_loader, val_loader, cfg, global_config):
    device = global_config['device']
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=global_config['learning_rate'], weight_decay=global_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, min_lr=1e-6)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_config['epochs'], eta_min=1e-6)

    # build loss functions from config
    loss_fns = {}
    for name, out_cfg in cfg["outputs"].items():
        if out_cfg["loss"] == "huber":
            loss_fns[name] = masked_huber_loss #torch.nn.SmoothL1Loss()
        elif out_cfg["loss"] == "bce":
            loss_fns[name] = torch.nn.BCEWithLogitsLoss()
        elif out_cfg["loss"] == "crossentropy":
            loss_fns[name] = torch.nn.CrossEntropyLoss()
        elif out_cfg["loss"] == "kl":
            loss_fns[name] = torch.nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError(f"Unknown loss {out_cfg['loss']}")

    best_val_loss = float('inf')
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

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')

        #print(f"Epoch [{epoch+1}/{global_config['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, logs

def save_results(model, logs, cfg):

    out_dir = os.path.join("../results/train", cfg['exp'])
    os.makedirs(out_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))

    # Save logs and cfg as JSON
    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump(logs, f)
    with open(os.path.join(out_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f)

    print("Results saved to:", out_dir)
    # Optionally, save predictions and targets for val/test sets
    # preds, targets = get_predictions_and_targets(val_loader, model)
    # np.save(os.path.join(out_dir, "predictions_val.npy"), preds)
    # np.save(os.path.join(out_dir, "targets_val.npy"), targets)