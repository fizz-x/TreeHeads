import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error
import torch

def plot_val_loss(train_losses, val_losses):
    """
    Plot training and validation loss over epochs.
    
    Parameters:
    - train_losses: list of training losses
    - val_losses: list of validation losses
    """
    plt.figure(figsize=(6, 2.2))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.show()

def load_normalization_params(json_path):
    """
    Load normalization parameters from a JSON file.

    Parameters:
    - json_path: str, path to the JSON file

    Returns:
    - mu: float, mean value
    - std: float, standard deviation
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    mu = params['mu']
    std = params['std']
    return mu, std

def denormalize(tensor, mean, std):
    return tensor * std + mean

def denorm_model_json(model, test_loader, json_path, config=None):
    mu, std = load_normalization_params(json_path)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs, targets, mask = inputs.to(config['device']), targets.to(config['device']), mask.to(config['device'])
            outputs = model(inputs)

            # Ensure mask is boolean and has the same shape as outputs/targets
            mask = mask.bool()
            if mask.shape != outputs.shape:
                mask = mask.view_as(outputs)
            masked_outputs = outputs[mask]
            masked_targets = targets[mask]
            all_preds.append(masked_outputs.cpu().numpy())
            all_targets.append(masked_targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize predictions and targets
    all_preds = denormalize(torch.from_numpy(all_preds), mu, std).numpy()
    all_targets = denormalize(torch.from_numpy(all_targets), mu, std).numpy()

    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
    r2 = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
    bias = np.mean(all_preds - all_targets)
    #print(f"[DEBUG] - Length all_preds: {len(all_preds)}; len all_targets: {len(all_targets)}, shape: {all_preds.shape}")
    print(f"ALS Denormalization dn = tensor * std + µ\n \tµ: \t{mu:.2f}m\n \tstd: \t{std:.2f}m")
    print("Metrics:")
    print(f"\tMAE: \t{mae:.2f}m\n \tRMSE: \t{rmse:.2f}m\n \tBias: \t{bias:.2f}m\n \tR2: \t{r2:.2f}") 
    print("----------------------------------------------")
    return all_preds, all_targets

def plot_real_pred_delta(model, dataloader, num_samples=5, device='cpu', json_path=None):
    """
    Plot S2 RGB, real ALS patches, model predictions, and their delta.

    Parameters:
    - y: np.ndarray, shape (N, 32, 32)
    - model: trained model
    - dataloader: DataLoader for test data
    - num_samples: int, number of samples to plot
    - device: torch device
    """
    model.eval()
    shown = 0
    mu, std = load_normalization_params(json_path)

    with torch.no_grad():
        for X_batch, y_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            #X_batch = X_batch.reshape([])
            y_batch = y_batch.cpu().numpy()
            preds = model(X_batch).cpu().numpy()
            # Denormalize 
            y_batch = denormalize(torch.from_numpy(y_batch), mu, std).numpy()
            preds = denormalize(torch.from_numpy(preds), mu, std).numpy()

            for i in range(X_batch.shape[0]):
                if shown >= num_samples:
                    return
                # S2 RGB: channels 10, 3, 0 (B, G, R) for Sentinel-2
                rgb = X_batch[i, [10, 3, 0]].cpu().numpy()
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                rgb = np.transpose(rgb, (1, 2, 0))
                gt = y_batch[i, 0] if y_batch.ndim == 4 else y_batch[i]
                pred = preds[i, 0] if preds.ndim == 4 else preds[i]
                delta = pred - gt
                plt.figure(figsize=(15, 3))
                # S2 RGB
                plt.subplot(1, 4, 1)
                plt.imshow(rgb)
                plt.title("S2 RGB")
                plt.axis('off')
                # ALS GT
                plt.subplot(1, 4, 2)
                plt.title("ALS Ground Truth")
                plt.axis('off')
                im = plt.imshow(gt, cmap='viridis')
                plt.colorbar(im, ax=plt.gca())  
                # pred 
                plt.subplot(1, 4, 3)
                im = plt.imshow(pred, cmap='viridis')
                plt.colorbar(im, ax=plt.gca())  
                plt.title("Model Prediction")
                plt.axis('off')
 
                plt.subplot(1, 4, 4)
                imd = plt.imshow(delta, cmap='bwr')
                plt.title("Prediction - GT (Delta)")
                plt.axis('off')
                plt.colorbar(imd, ax=plt.gca(), location='right')
                plt.tight_layout()
                plt.show()
                shown += 1
                #print(min(delta.flatten()), max(delta.flatten()))
# plot y vs. preds as heatmap
def plot_heatmap(y_true, y_pred, title="Heatmap of True vs Predicted"):
    """
    Plot a heatmap of true vs predicted values.

    Parameters:
    - y_true: np.ndarray, true values
    - y_pred: np.ndarray, predicted values
    - title: str, title of the plot
    """
    plt.figure(figsize=(7, 5))
    plt.hexbin(y_true.flatten(), y_pred.flatten(), gridsize=90, cmap='viridis', mincnt=2)
    plt.colorbar(label='Counts')
    plt.xlabel('Ground Truth Values [m]')
    plt.ylabel('Predicted Values [m]')
    plt.title(title)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.xlim(y_true.min(), 50)
    plt.ylim(y_true.min(), 50)
    plt.grid(True)
    plt.show()

def plot_compact_heatmap_val_test(y_val_true, y_val_pred, y_test_true, y_test_pred, title="Heatmap of Ground-Truth vs Predicted Canopy Height\n"):
    """
    Plot two heatmaps of true vs predicted values: left for validation, right for test set.

    Parameters:
    - y_val_true: np.ndarray, true values for validation set
    - y_val_pred: np.ndarray, predicted values for validation set
    - y_test_true: np.ndarray, true values for test set
    - y_test_pred: np.ndarray, predicted values for test set
    - title: str, title of the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    vmax = max(y_val_true.max(), y_val_pred.max(), y_test_true.max(), y_test_pred.max(), 50)
    vmin = min(y_val_true.min(), y_val_pred.min(), y_test_true.min(), y_test_pred.min())
    import matplotlib.colors as mcolors

    # Validation set heatmap
    axes[0].hexbin(y_val_true.flatten(), y_val_pred.flatten(), gridsize=120, cmap='viridis', mincnt=10) #norm=mcolors.LogNorm()
    axes[0].plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--')
    axes[0].set_xlabel('Ground Truth [m]')
    axes[0].set_ylabel('Predicted [m]')
    axes[0].set_title('Validation Set')
    axes[0].set_xlim(0, 50)
    axes[0].set_ylim(0, 50)
    axes[0].grid(True)

    # Test set heatmap
    hb = axes[1].hexbin(y_test_true.flatten(), y_test_pred.flatten(), gridsize=120, cmap='viridis', mincnt=10) # viridis
    axes[1].plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--')
    axes[1].set_xlabel('Ground Truth [m]')
    axes[1].set_ylabel('Predicted [m]')
    axes[1].set_title('Test Set')
    axes[1].set_xlim(0, 50)
    axes[1].set_ylim(0, 50)
    axes[1].grid(True)

    fig.colorbar(hb, ax=axes, orientation='vertical', fraction=0.03, pad=0.04, label='Counts',)
    fig.suptitle(title)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_eval_report(train_losses, val_losses, model, val_loader,test_loader, json_path=None, config=None):
    """
    Plot evaluation report including loss curves and heatmap of true vs predicted values.

    Parameters:
    - train_losses: list of training losses
    - val_losses: list of validation losses
    - y_true: np.ndarray, true values
    - y_pred: np.ndarray, predicted values
    - json_path: str, path to normalization parameters JSON file (optional)
    """
    import utils.eval as eval
    from utils.eval import plot_heatmap, plot_compact_heatmap_val_test, denorm_model_json, plot_real_pred_delta
    
    plot_val_loss(train_losses, val_losses)

    print("VALIDATION METRICS")
    predictions, targets = denorm_model_json(model, val_loader, json_path, config=config)
    print("TEST METRICS")
    pred_test, target_test = denorm_model_json(model, test_loader, json_path, config=config)
    #a = simpletest(predictions)
    plot_compact_heatmap_val_test(targets, predictions, target_test, pred_test, title="Heatmap of Ground-Truth vs Predicted Canopy Height\n")
    #plot_heatmap(targets, predictions, title="Val-Set: \nHeatmap of True vs Predicted Canopy Height")
    #plot_heatmap(target_test, pred_test, title="Test-Set: \nHeatmap of True vs Predicted Canopy Height")
    #plot_compact_heatmap_val_test(targets, predictions, target_test, pred_test)
    plot_real_pred_delta(model, val_loader, num_samples=3, device=config['device'],json_path=json_path)
    

    # print all config parameters 
    print("Configuration Parameters:")
    for key, value in config.items():
        print(f"{key}: \t{value}")

    print("-------------------------------")
    #print("Model Architecture:")
    #print(model)

    print("Evaluation report completed.")

