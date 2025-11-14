import os
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_absolute_error
import torch
import seaborn as sns
from matplotlib import cm
import glob

def plot_val_loss(train_losses, val_losses, title="Training and Validation Loss", report=None):
    """
    Plot training and validation loss over epochs.
    
    Parameters:
    - train_losses: list of training losses
    - val_losses: list of validation losses
    """
    plt.figure(figsize=(4, 1.6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    ymax = np.percentile(train_losses, 99)
    ymin = min(np.percentile(val_losses, 1), np.percentile(train_losses, 1))*0.97
    plt.ylim(ymin, ymax)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    #if report is not None:
        # Save BEFORE plt.show(), so the figure is not cleared
        #save_plot(plt.gcf(), "train_val_loss", report)
    #plt.show()
    return plt.gcf()

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

def denorm_model_json(model, test_loader, json_path, config=None,verbose=False):
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

    if verbose:
        print(f"ALS Denormalization dn = tensor * std + µ\n \tµ: \t{mu:.2f}m\n \tstd: \t{std:.2f}m")

    return all_preds, all_targets

def write_metrics_to_df(report, sites, global_config, df=None):

    experiment_name = report.get("experiment_name", "Unknown")

    metrics = {}
    pv = report["predictions"]["validation"].copy()
    tv = report["targets"]["validation"].copy()
    pt = report["predictions"]["test"].copy()
    tt = report["targets"]["test"].copy()
    mv = report["masks"]["validation"].copy()
    mt = report["masks"]["test"].copy()

    pv[~mv] = np.nan
    tv[~mv] = np.nan
    pt[~mt] = np.nan
    tt[~mt] = np.nan

    pv_ = pv.copy()
    tv_ = tv.copy()
    pt_ = pt.copy()
    tt_ = tt.copy()

    do_min = True
    if do_min:
        repl = np.nan
        thresh = 5
        mask_val = tv_ < thresh
        tv_[mask_val] = repl
        pv_[mask_val] = repl
        mask_test = tt_ < thresh
        tt_[mask_test] = repl
        pt_[mask_test] = repl
        mae_val_, nmae_val_, rmse_val_, bias_val_, r2_val_ = get_metrics(pv_, tv_, verbose=False)
        mae_test_, nmae_test_, rmse_test_, bias_test_, r2_test_ = get_metrics(pt_, tt_, verbose=False)

    mae_val, nmae_val, rmse_val, bias_val, r2_val = get_metrics(pv, tv, verbose=False)
    mae_test, nmae_test, rmse_test, bias_test, r2_test = get_metrics(pt, tt, verbose=False)
    f1_ch1, f1_macro = get_aux_metrics(report, verbose=False)

    metrics = {
        "Experiment": experiment_name,
        "MAE [m] (Val)": round(mae_val, 2),
        "MAE [m] (Test)": round(mae_test, 2),
        "nMAE [%] (Val)": round(nmae_val, 2),
        "nMAE [%] (Test)": round(nmae_test, 2),
        "RMSE [m] (Val)": round(rmse_val, 2),
        "RMSE [m] (Test)": round(rmse_test, 2),
        "Bias [m] (Val)": round(bias_val, 2),
        "Bias [m] (Test)": round(bias_test, 2),
        "R2 [-] (Val)": round(r2_val, 2),
        "R2 [-] (Test)": round(r2_test, 2),
        "F1 FMASK (Test)": round(f1_ch1, 4),
        "F1 DLT (Test)": round(f1_macro, 4)
       # "---": "---",
    }

    if do_min:
        metrics.update(
        {
        f"MAE [m] (Val, >{thresh}m)": round(mae_val_, 2),
        f"MAE [m] (Test, >{thresh}m)": round(mae_test_, 2),
        f"RMSE [m] (Val, >{thresh}m)": round(rmse_val_, 2),
        f"RMSE [m] (Test, >{thresh}m)": round(rmse_test_, 2),
        f"Bias [m] (Val, >{thresh}m)": round(bias_val_, 2),
        f"Bias [m] (Test, >{thresh}m)": round(bias_test_, 2),
        f"R2 [-] (Val, >{thresh}m)": round(r2_val_, 2),
        f"R2 [-] (Test, >{thresh}m)": round(r2_test_, 2)
        }
        )

    metrics.update(global_config)

    if df is None:
        df = pd.DataFrame([metrics])
    else:
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)


    if df is None:
        df = pd.DataFrame({
            'Metric': ['MAE [m]', 'RMSE [m]', 'Bias [m]', 'R2 [-]'],
        })

    if 'combo' in df.columns:
        df['combo'] = df['combo'].astype(str)
        df.loc[~df['combo'].str.startswith('_'), 'combo'] = '_' + df['combo']
        df['combo'] = df['combo'].astype("category") # ensure it remains categorical
    return df

def get_metrics(all_preds, all_targets, verbose = True):

    # Flatten and mask out NaNs in both arrays
    preds_flat = all_preds.flatten()
    targets_flat = all_targets.flatten()
    mask = ~np.isnan(preds_flat) & ~np.isnan(targets_flat)
    preds_flat = preds_flat[mask]
    targets_flat = targets_flat[mask]

    mae_abs = mean_absolute_error(targets_flat, preds_flat)
    rmse = np.sqrt(np.mean((targets_flat - preds_flat) ** 2))
    r2 = 1 - np.sum((targets_flat - preds_flat) ** 2) / np.sum((targets_flat - np.mean(targets_flat)) ** 2)
    bias = np.mean(preds_flat - targets_flat)

    ## nmae old
    # Calculate normalized MAE as percentage using binning (excluding 0-2m range)
    bins = np.arange(2, 60, 2)  # Create bins from 2m to 60m in 2m steps
    bin_indices = np.digitize(targets_flat, bins) - 1
    mae_percent_per_bin = np.zeros(len(bins))
    counts_per_bin = np.zeros(len(bins))
    
    # for i in range(len(bins)): #
    #     bin_mask = bin_indices == i
    #     if np.sum(bin_mask) > 0:  # Only calculate if bin has values
    #         # Calculate MAE as percentage of the bin center value
    #         bin_center = bins[i] + 1  # Center of the 2m bin
    #         mae = mean_absolute_error(targets_flat[bin_mask], preds_flat[bin_mask])
    #         mae_percent_per_bin[i] = (mae / bin_center) * 100
    #         counts_per_bin[i] = np.sum(bin_mask)
    
    # # Calculate normalized MAE as average percentage across valid bins
    # valid_bins = counts_per_bin > 0
    # nMAE = np.average(mae_percent_per_bin[valid_bins], weights=counts_per_bin[valid_bins])

    # nmae new: 
    # Mask targets within [2, 60] and calculate the flat nMAE
    mask_nmae = (targets_flat >= 2) & (targets_flat <= 60)
    nmae_flat = np.mean(np.abs(preds_flat[mask_nmae] - targets_flat[mask_nmae]) / targets_flat[mask_nmae]) * 100

    #print(f"[DEBUG] - Length all_preds: {len(all_preds)}; len all_targets: {len(all_targets)}, shape: {all_preds.shape}")
    if verbose:
        print("Metrics:")
        print(f"\tMAE: \t{mae_abs:.2f}m\n \tRMSE: \t{rmse:.2f}m\n \tBias: \t{bias:.2f}m\n \tR2: \t{r2:.2f}") 
        print("----------------------------------------------")
    return mae_abs, nmae_flat, rmse, bias, r2

def get_aux_metrics(report, verbose = True):
    from sklearn.metrics import f1_score

    if not report.get("aux") or not report["aux"].get("targets") or report["aux"]["targets"].get("test") is None:
        if verbose:
            print("No auxiliary predictions/targets found in report.")
        return np.nan, np.nan

    pt = report["aux"]["predictions"]["test"].copy()
    tt = report["aux"]["targets"]["test"].copy()
    mask = report["masks"]["test"].copy()
    mask = mask.astype(bool).flatten()

    # Channel 1: F1 score (binary classification)
    fmask_gt_ch1 = tt[:, 0, :, :].flatten()
    fmask_gt_ch1 = fmask_gt_ch1[mask]
    fmask_pred_ch1 = (pt[:, 0, :, :].flatten() > 0.5).astype(np.uint8)
    fmask_pred_ch1 = fmask_pred_ch1[mask]
    f1_ch1 = f1_score(fmask_gt_ch1, fmask_pred_ch1, average='binary')

    # Channel 2: Multi-class classification (0, 1, 2)
    fmask_gt_ch2 = tt[:, 1, :, :].flatten()
    fmask_gt_ch2 = fmask_gt_ch2[mask]
    fmask_pred_ch2 = pt[:, 1, :, :].flatten()
    fmask_pred_ch2 = fmask_pred_ch2[mask]
    f1_micro = f1_score(fmask_gt_ch2, fmask_pred_ch2, average='micro')

    if verbose:
        print("Auxiliary Metrics:")
        print(f"\tF1 Score Channel 1 (Binary): \t{f1_ch1:.4f}\n \tF1 Score Channel 2 (Micro): \t{f1_micro:.4f}") 
        print("----------------------------------------------")
    return f1_ch1, f1_micro

def print_all_metrics(report, sites, cfg, above2m = False):
    """
    Print all metrics for validation and test sets.

    Parameters:
    - predictions: np.ndarray, predicted values for validation set
    - targets: np.ndarray, true values for validation set
    - pred_test: np.ndarray, predicted values for test set
    - target_test: np.ndarray, true values for test set
    """
    #print("Validation Set Metrics:")
    df = pd.DataFrame({
        'Metric': ['MAE [m]', 'RMSE [m]', 'Bias [m]', 'R2 [-]', 'norm_mu [m]', 'norm_std [m]'],
    })
    if cfg["strategy"] == "all_in_model":
        mu = sites['SITE1']['CHM_norm_params']['mu']
        std = sites['SITE1']['CHM_norm_params']['std']
    else:
        mu = -1
        std = -1

    pv = report["predictions"]["validation"].copy()
    tv = report["targets"]["validation"].copy()
    pt = report["predictions"]["test"].copy()
    tt = report["targets"]["test"].copy()

    repl = np.nan
    
    if above2m:
        # Mask out everything below 2m in targets, and mask preds at the same positions
        mask_val = tv < 2
        tv[mask_val] = repl
        pv[mask_val] = repl 
        mask_test = tt < 2
        tt[mask_test] = repl
        pt[mask_test] = repl
        #pv, tv, pt, tt = pv[pv > 2], tv[tv > 2], pt[pt > 2], tt[tt > 2]
    #print("Minimum val / test: ", np.nanmin(pv), np.nanmin(tv), np.nanmin(pt), np.nanmin(tt))
    
    mae, rmse, bias, r2 = get_metrics(pv, tv,verbose=False)
    df['Validation'] = [mae, rmse, bias, r2, mu, std]
    mae, rmse, bias, r2 = get_metrics(pt, tt,verbose=False)
    df['Test'] = [mae, rmse, bias, r2, mu, std]
    #print(df)

    print(df.to_string(index=False, header = True, col_space=8, float_format="{:.2f}".format),)

def plot_real_pred_delta(report, num_samples=5, device='cpu'):
    """
    Plot S2 RGB, real ALS patches, model predictions, and their delta.

    Parameters:
    - y: np.ndarray, shape (N, 32, 32)
    - model: trained model
    - dataloader: DataLoader for test data
    - num_samples: int, number of samples to plot
    - device: torch device
    """
    rgb = report["rgb"]["test"].copy()
    preds = report["predictions"]["test"].copy()
    targets = report["targets"]["test"].copy()
    #model = report["model_weights"].copy()
    mask = report["masks"]["test"].copy()
    shown = 0
    name = report["experiment_name"]

    for i in range(len(rgb)):
        if shown >= num_samples:
            break
            
        # Get sample
        rgb_sample = rgb[i, [2,1,0]] # Assuming RGB channels are last 3
        #rgb_sample = (rgb_sample - rgb_sample.min()) / (rgb_sample.max() - rgb_sample.min() + 1e-6)
        rgb_sample = np.transpose(rgb_sample, (1, 2, 0))
        
        pred_sample = preds[i]
        target_sample = targets[i] 
        mask_sample = mask[i]
        
        # Apply mask
        pred_sample = np.where(mask_sample, pred_sample, np.nan)
        target_sample = np.where(mask_sample, target_sample, np.nan)
        delta = pred_sample - target_sample
        delta = np.where(mask_sample, delta, np.nan)

        # Create plot
        plt.figure(figsize=(15, 3))
        
        # S2 RGB
        plt.subplot(1, 4, 1)
        plt.imshow(rgb_sample)
        plt.title("S2 RGB")
        plt.axis('off')
        
        # ALS Ground Truth
        plt.subplot(1, 4, 2)
        plt.title("ALS Ground Truth [m]")
        plt.axis('off')
        min_val = 0 #np.nanmin(target_sample)
        max_val = np.nanmax(target_sample)*0.95
        im = plt.imshow(target_sample, cmap='viridis', vmin=min_val, vmax=max_val)
        plt.colorbar(im, ax=plt.gca())
        
        # Prediction
        plt.subplot(1, 4, 3)
        im = plt.imshow(pred_sample, cmap='viridis', vmin=min_val, vmax=max_val)
        plt.colorbar(im, ax=plt.gca())
        plt.title("Model Prediction [m]")
        plt.axis('off')
        
        # Delta/Error
        plt.subplot(1, 4, 4)
        vmax = np.nanmax(np.abs(delta))
        imd = plt.imshow(delta, cmap='bwr', vmin=-vmax, vmax=vmax)
        plt.title("Error = Prediction - GT [m]")
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

def plot_compact_heatmap_val_test(report, title="Heatmap of Ground-Truth vs Predicted Canopy Height\n"):
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

    # Validation set
    y_val_true = report["targets"]["validation"].copy()
    y_val_pred = report["predictions"]["validation"].copy()
    # Test set
    y_test_true = report["targets"]["test"].copy()
    y_test_pred = report["predictions"]["test"].copy()

    maskval = report["masks"]["validation"].copy()
    masktest = report["masks"]["test"].copy()

    # Flatten and drop NaNs for min/max calculation
    # def nanminmax(*arrays):
    #     arr = np.concatenate([a.flatten() for a in arrays])
    #     arr = arr[~np.isnan(arr)]
    #     return np.nanmin(arr), np.nanmax(arr)

    # vmin, vmax = nanminmax(y_val_true, y_val_pred, y_test_true, y_test_pred)
    vmin, vmax = 0, 50

    maskval = maskval.flatten()
    masktest = masktest.flatten()
    maskval = maskval[~np.isnan(maskval)]
    masktest = masktest[~np.isnan(masktest)]

    # # Mask out y values depending on their mask
    y_val_true = np.where(maskval, y_val_true.flatten(), np.nan)
    y_val_pred = np.where(maskval, y_val_pred.flatten(), np.nan)
    y_test_true = np.where(masktest, y_test_true.flatten(), np.nan)
    y_test_pred = np.where(masktest, y_test_pred.flatten(), np.nan)

    vmax = max(vmax, 50)
    import matplotlib.colors as mcolors
    # Use logarithmic normalization for color intensity to better visualize dense regions
    norm = mcolors.LogNorm(vmin=50, vmax=5000)

    # Validation set heatmap
    axes[0].hexbin(
        y_val_true.flatten(), y_val_pred.flatten(),
        gridsize=100, cmap='viridis', mincnt=15, norm=norm
    )
    axes[0].plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--')
    axes[0].set_xlabel('Ground Truth [m]')
    axes[0].set_ylabel('Predicted [m]')
    axes[0].set_title('Validation Set')
    axes[0].set_xlim(0, 50)
    axes[0].set_ylim(0, 50)
    axes[0].grid(True)

    # Test set heatmap
    hb = axes[1].hexbin(
        y_test_true.flatten(), y_test_pred.flatten(),
        gridsize=100, cmap='viridis', mincnt=15, norm=norm
    )
    axes[1].plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--')
    axes[1].set_xlabel('Ground Truth [m]')
    axes[1].set_ylabel('Predicted [m]')
    axes[1].set_title('Test Set')
    axes[1].set_xlim(0, 50)
    axes[1].set_ylim(0, 50)
    axes[1].grid(True)
    
    if False:
        # Validation set heatmap
        axes[0].hexbin(y_val_true.flatten(), y_val_pred.flatten(), gridsize=100, cmap='viridis', mincnt=15) #norm=mcolors.LogNorm()
        axes[0].plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--')
        axes[0].set_xlabel('Ground Truth [m]')
        axes[0].set_ylabel('Predicted [m]')
        axes[0].set_title('Validation Set')
        axes[0].set_xlim(0, 50)
        axes[0].set_ylim(0, 50)
        axes[0].grid(True)

        # Test set heatmap
        hb = axes[1].hexbin(y_test_true.flatten(), y_test_pred.flatten(), gridsize=100, cmap='viridis', mincnt=15) # viridis
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
    # save_plot(plt.gcf(), "heatmap", report)
    # plt.show()
    return fig

def plot_error_over_frequency(report, bins=60, title = "Error vs. GT Distribution"):
    """
    Plot the error distribution over frequency for test set.

    Parameters:
    - report: dict, containing targets, predictions, and masks
    - bins: int, number of bins for histogram
    - title: str, title of the plot
    """

    y_test_true = report["targets"]["test"].copy()
    y_test_pred = report["predictions"]["test"].copy()
    
    test_errors = y_test_pred - y_test_true
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Test set - Absolute Error
    test_bin_means, test_bin_edges, _ = stats.binned_statistic(
        y_test_true.flatten(), test_errors.flatten(), statistic='mean', bins=bins
    )
    test_bin_counts, _, _ = stats.binned_statistic(
        y_test_true.flatten(), y_test_true.flatten(), statistic='count', bins=test_bin_edges
    )
    # mask out test_bin_means where counts are less than 10
    mask = test_bin_counts < 10
    test_bin_means[mask] = np.nan

    bin_centers = (test_bin_edges[:-1] + test_bin_edges[1:]) / 2
    axes[0].fill_between(bin_centers, 0, test_bin_means, alpha=0.3, color='green', label='Mean Error')
    axes[0].plot(
        bin_centers, test_bin_means, label='Mean Error', color='green', linewidth=2
    )
    axes[0].set_xlabel('Ground Truth [m]')
    axes[0].set_ylabel('Mean Absolute Error [m]')
    axes[0].set_title('Test: Absolute Error vs. Ground Truth')
    axes[0].grid(True)
    axes[0].set_xlim(0, 50)
    axes[0].set_ylim(-20, 20)
    # Overlay ground truth distribution (dashed line, scaled for visibility)
    gt_dist = test_bin_counts / test_bin_counts.max() * 18
    axes[0].plot(
        bin_centers, gt_dist, '--', color='gray', label='GT Distribution (scaled)'
    )
    axes[0].legend(loc='lower left')

    # Test set - Normalized MAE (nMAE) - only for heights in range [2, 60]m
    abs_errors = np.abs(test_errors.flatten())
    gt_flat = y_test_true.flatten()
    
    # Mask to only include values in [2, 60] range
    mask_range = (gt_flat >= 2) & (gt_flat <= 60)
    abs_errors_masked = abs_errors[mask_range]
    gt_masked = gt_flat[mask_range]
    
    nmae_values = (abs_errors_masked / gt_masked) * 100  # percentage
    
    test_bin_nmae, test_bin_edges_nmae, _ = stats.binned_statistic(
        gt_masked, nmae_values, statistic='mean', bins=bins
    )
    test_bin_counts_nmae, _, _ = stats.binned_statistic(
        gt_masked, gt_masked, statistic='count', bins=test_bin_edges_nmae
    )
    # mask out test_bin_nmae where counts are less than 10
    mask_nmae = test_bin_counts_nmae < 10
    test_bin_nmae[mask_nmae] = np.nan

    axes[1].plot(
        (test_bin_edges_nmae[:-1] + test_bin_edges_nmae[1:]) / 2, test_bin_nmae, label='Mean nMAE', color='orange'
    )
    axes[1].set_xlabel('Ground Truth [m]')
    axes[1].set_ylabel('Normalized MAE [%]')
    axes[1].set_title('Test: Normalized Error vs. Ground Truth (2-60m)')
    axes[1].grid(True)
    #axes[1].set_xlim(2, 60)
    # Overlay ground truth distribution (dashed line, scaled for visibility)
    gt_dist_nmae = test_bin_counts_nmae / test_bin_counts_nmae.max() * np.nanmax(test_bin_nmae) * 0.5
    axes[1].plot(
        (test_bin_edges_nmae[:-1] + test_bin_edges_nmae[1:]) / 2, gt_dist_nmae, '--', color='gray', label='GT Distribution (scaled)'
    )
    axes[1].legend(loc='lower left')

    fig.suptitle(title)
    plt.tight_layout()
    return fig

def plot_eval_report(train_losses, val_losses, model, val_loader, test_loader, json_path=None, config=None):
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

    
    #print("VALIDATION METRICS")
    predictions, targets = denorm_model_json(model, val_loader, json_path, config=config)
    #mae, rmse, bias, r2 = eval.get_metrics(predictions, targets)
    #print("TEST METRICS")
    pred_test, target_test = denorm_model_json(model, test_loader, json_path, config=config)
    #mae, rmse, bias, r2 = eval.get_metrics(pred_test, target_test)
    
    print("METRIC REPORT:")
    eval.print_all_metrics(predictions, targets, pred_test, target_test, json_path)
    plot_val_loss(train_losses, val_losses)

    plot_error_over_frequency(targets, predictions, target_test, pred_test, bins=50)
    plot_compact_heatmap_val_test(targets, predictions, target_test, pred_test, title="Heatmap of Ground-Truth vs Predicted Canopy Height\n")
    #plot_heatmap(targets, predictions, title="Val-Set: \nHeatmap of True vs Predicted Canopy Height")
    #plot_heatmap(target_test, pred_test, title="Test-Set: \nHeatmap of True vs Predicted Canopy Height")
    #plot_compact_heatmap_val_test(targets, predictions, target_test, pred_test)
    plot_real_pred_delta(model, val_loader, num_samples=3, device=config['device'],json_path=json_path)
    

    # print all config parameters 
    print("Configuration Parameters:")
    for key, value in config.items():
        if key != 'lr':
            print(f"{key}: \t{value}")
        else:
            print(f"{key}: \t\t{value:.2e}")  # format learning rate in scientific notation

    print("-------------------------------")
    # print the model size and number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = num_params * 4 / (1024 ** 2)  #
    print(f"Model size: \t\t{model_size_mb:.2f} MB \nNumber of parameters: \t{num_params:.2e}")
    #print("Model Architecture:")
    #print(model)
    print("-------------------------------")
    print("Evaluation report completed.")

def ziptheresults(exp_name, model_weights, logs, cfg, preds_val, targets_val, preds_test, targets_test, maskval, masktest, rgb_test):
    """
    Compile a new evaluation report for the given experiment.
    """

    auxreport = {}
    if preds_val.shape[1] > 1: # check if aux channels exist
        auxreport = {
            "predictions": {
                "validation": preds_val[:,1:,:,:], 
                "test": preds_test[:,1:,:,:]
            },
            "targets": {
                "validation": targets_val[:,1:,:,:],
                "test": targets_test[:,1:,:,:]
            }
        }
    report = {
        "experiment_name": exp_name,
        "model_weights": model_weights,
        "logs": logs,
        "config": cfg,
        "predictions": {
            "validation": preds_val[:,0,:,:],
            "test": preds_test[:,0,:,:]
        },
        "targets": {
            "validation": targets_val[:,0,:,:],
            "test": targets_test[:,0,:,:]
        },
        "masks": {
            "validation": maskval[:,0,:,:],
            "test": masktest[:,0,:,:]
        },
        "rgb": {
            #"validation": rgb_val,
            "test": rgb_test
        },
        "aux": {
            **auxreport
        }

    }

    return report

def save_plot(fig, plotname, report, run_id=None):
    if run_id is None:
        outdir = os.path.join('../results', 'eval')
    else:
        outdir = os.path.join('../results', run_id, 'eval')
    os.makedirs(outdir, exist_ok=True)
    fname = f"{report['experiment_name']}+{plotname}.png"
    fig.savefig(os.path.join(outdir, fname), bbox_inches='tight')
    plt.close(fig)

def printout_eval_report(report, sites, cfg, run_id):
    """
    Print the evaluation report in a readable format.
    """
    #print("Evaluation Report:")
    #print("-------------------------------")
    #print(f"Experiment Name: \t{report['experiment_name']}")
    #print_all_metrics(report,sites,cfg,above2m=False)
    #plot_compact_heatmap_val_test(report, title=f"{report['experiment_name']} - Heatmap of Ground-Truth vs Predicted Canopy Height\n")
    #plot_error_over_frequency(report, bins=80, title = f"{report['experiment_name']} - Error vs. GT Distribution")
    # Save figures to disk instead of showing
    fig = plot_val_loss(report["logs"]["train_loss"], report["logs"]["val_loss"], title=f"{report['experiment_name']} - Training and Validation Loss", report=report)
    save_plot(fig, "train_val_loss", report, run_id)
    # Heatmap
    fig = plot_compact_heatmap_val_test(report, title=f"{report['experiment_name']} - Heatmap of Ground-Truth vs Predicted Canopy Height\n")
    save_plot(fig, "heatmap", report, run_id)
    # Error vs GT Distribution
    fig = plot_error_over_frequency(report, bins=80, title = f"{report['experiment_name']} - Error vs. GT Distribution")
    save_plot(fig, "error_vs_gt", report, run_id)

def save_df_result_to_csv(df_result, run_id, override=True):
    # Save the transposed DataFrame to a CSV file with optional override
    path = f"../results/{run_id}/metrics/"
    path_csv = path + "results_summary.csv"
    path_plt = path + "results_summary.png"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plot_experiment_metrics_test_only(df_result, printout=False)

    if override:
        df_result.transpose().to_csv(path_csv)
        fig.savefig(path_plt, bbox_inches='tight')
        plt.close(fig)
        print("Results saved to", path_csv)
    else:
        base, ext = os.path.splitext(path_csv)
        suffix = 1
        new_path = f"{base}_{suffix}{ext}"
        while os.path.exists(new_path):
            suffix += 1
            new_path = f"{base}_{suffix}{ext}"
        df_result.transpose().to_csv(new_path)
        print("Results saved to", new_path)

def read_multiple_csv_to_df(run_ids):
    """
    Read CSV metrics files from multiple run_ids and combine them into one DataFrame.
    
    Args:
        run_ids (list): List of run IDs to process
        
    Returns:
        pd.DataFrame: Combined DataFrame with all metrics
    """
    df_list = []
    for run_id in run_ids:
        path = f"../results/{run_id}/metrics/"
        file = os.path.join(path, "results_summary.csv")
        if os.path.exists(file):
            df = pd.read_csv(file).transpose()
            df.columns = df.iloc[0]  # Set first row as column names
            df = df.iloc[1:]  # Remove first row since it's now column names
            df_list.append(df)
            
    if df_list:
        df_result = pd.concat(df_list, axis=0, ignore_index=True)
        # Cast metrics columns to numeric, excluding the Experiment name column
        #metric_columns = [col for col in df_result.columns if col != 'Experiment']
        #metric_columns = first 11 columns after 'Experiment'
        metric_columns = df_result.columns[1:19].tolist()
        df_result[metric_columns] = df_result[metric_columns].apply(pd.to_numeric)

        return df_result
    else:
        print("No CSV files found in any of the specified run_ids")
        return None

def plot_experiment_metrics_test_only(df_result, title=None, printout=False):

    """
    Plot only the Test metrics (MAE, nMAE, RMSE, Bias) for each experiment as barplots.
    Args:
        df (pd.DataFrame): DataFrame containing metrics for different experiments.
    """
    df = df_result.iloc[:, :11]
    metrics = [
        "MAE [m] (Test)",
        "nMAE [%] (Test)",
        "RMSE [m] (Test)",
        "Bias [m] (Test)",
        "R2 [-] (Test)"
    ]
    metric_labels = ["MAE [m]", "nMAE [%]", "RMSE [m]", "Bias [m]", "R2 [-]"]

    exp_col = df.columns[0]
    exp_names = df[exp_col].tolist()
    

    # Shorten experiment names for x-axis
    short_names = [str(i+1) for i in range(len(exp_names))]
    exp_name_map = dict(zip(exp_names, short_names))
    df_short = df.copy()
    df_short[exp_col] = df_short[exp_col].map(exp_name_map)

    # Define experiment groups and colors
    group_map = {
        0: "Baseline",
        1: "CompositeA",
        2: "CompositeB",
        3: "CompositeC",
        4: "AuxLayerA",
        5: "AuxLayerB",
        6: "MidTraining",
    }
    viridis = plt.get_cmap('nipy_spectral')
    color_indices = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9]
    group_colors = {group: viridis(idx) for group, idx in zip(group_map.values(), color_indices)}
    exp_to_group = {short_names[i]: group_map.get(i, "Other") for i in range(len(short_names))}
    palette = {name: group_colors[exp_to_group[name]] for name in short_names}
    # TUM COLORMAP
    palette = [0x072140, 0x3070B3, 0x8F81EA, 0xB55CA5, 0xFED702, 0xF7B11E, 0x9FBA36]
    palette = {str(i+1): f'#{palette[i]:06x}' for i in range(len(palette))}
    
    # Prepare melted dataframe for plotting
    df_melted = pd.melt(
        df_short,
        id_vars=exp_col,
        value_vars=metrics,
        var_name="MetricType",
        value_name="Value"
    )
    df_melted["Metric"] = df_melted["MetricType"].apply(lambda x: x.split()[0] + " " + x.split()[1])
    df_melted["Group"] = df_melted[exp_col].map(exp_to_group)
    # if each experiment has multiple entries, we can show error bars (stddev)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # Initialize handles and labels for legend outside the loop
    handles, labels = None, None
    for i, label in enumerate(metric_labels):
        ax = axes[i // 3, i % 3]
        ax.grid(True, which='both', linestyle='--', linewidth=0.5,alpha=0.7)
        metric_group = df_melted[df_melted["Metric"] == label]

                # Plot bars
        bars = ax.bar(range(len(exp_names)), metric_group['Value'], 
                     #yerr=stats['std'], capsize=5,
                     color=[palette[str(i+1)] for i in range(len(exp_names))])

        # Add value annotations
        for bar, mean in zip(bars, metric_group['Value']):
            label_ = f'{mean:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   label_, ha='center', va='bottom', fontsize=9)

        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ymax = metric_group["Value"].max()
        ymin = min(metric_group["Value"].min(),0)  # Ensure ymin is not above zero
        #print("ymax:", ymax, "ymin:", ymin)
        ax.set_ylim(ymin * 1.1, ymax * 1.2 + 0.15)  # Set y-limit to 10% above max value

        # Set x-tick rotation safely
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha('center')
            
        #if handles and labels:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            # manual legend from the palette (one patch per short name)
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(color=palette[name], label=name) for name in palette.keys()]
            labels = exp_names #list(palette.keys())
        #print(handles, labels)
    # remove the last empty subplot if exists
    if len(axes.flatten()) > len(metric_labels):
        fig.delaxes(axes.flatten()[len(metric_labels)])
    fig.legend(handles, labels, title="Experiment", loc="center left", bbox_to_anchor=(0.7, 0.25), fontsize='large')
    if title is not None:
        plt.suptitle(title, fontsize='x-large')
    else:
        plt.suptitle("Evaluation Metrics by Experiment", fontsize='x-large')
    plt.tight_layout()
    if printout:
        plt.show()
    else:
        return fig
    
def plot_experiment_metrics_multiple_runs(df_result, title=None, printout=False):
    """
    Plot Test metrics for each experiment as barplots, showing mean and standard deviation
    across multiple runs.

    Args:
        df_result (pd.DataFrame): DataFrame containing metrics for different experiments and runs
        title (str, optional): Plot title 
        printout (bool): Whether to show the plot directly

    Returns:
        matplotlib.figure.Figure: The figure object if printout=False
    """
    # Get unique experiment names
    exp_names = df_result['Experiment'].unique()

    # Select metrics columns
    metrics = [
        "MAE [m] (Test)",
        "nMAE [%] (Test)",
        "RMSE [m] (Test)", 
        #"Bias [m] (Test, >5m)",
        "Bias [m] (Test)",
        "R2 [-] (Test)"
    ]
    metric_labels = ["MAE [m]", "nMAE [%]", "RMSE [m]", "Bias [m]", "R2 [-]"] #

    # Create short names for x-axis
    short_names = [str(i+1) for i in range(len(exp_names))]
    exp_name_map = dict(zip(exp_names, short_names))

    # Define color scheme
    group_map = {
        0: "Baseline", 
        1: "CompositeA",
        2: "CompositeB",
        3: "CompositeC", 
        4: "AuxLayerA",
        5: "AuxLayerB",
        6: "MidTraining"
    }

    # Create color palette
    viridis = plt.get_cmap('nipy_spectral')
    color_indices = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9]
    group_colors = {group: viridis(idx) for group, idx in zip(group_map.values(), color_indices)}
    exp_to_group = {short_names[i]: group_map.get(i, "Other") for i in range(len(short_names))}
    palette = {name: group_colors[exp_to_group[name]] for name in short_names}
    # TUM COLORMAP
    palette = [0x072140, 0x3070B3, 0x8F81EA, 0xB55CA5, 0xFED702, 0xF7B11E, 0x9FBA36]
    palette = {str(i+1): f'#{palette[i]:06x}' for i in range(len(palette))}
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Plot metrics
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i // 3, i % 3]
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Calculate mean and std per experiment
        stats = df_result.groupby('Experiment')[metric].agg(['mean', 'std']).reset_index()
        
        # Plot bars
        bars = ax.bar(range(len(exp_names)), stats['mean'], 
                     yerr=stats['std'], capsize=5,
                     color=[palette[str(i+1)] for i in range(len(exp_names))])

        # Add value annotations
        for bar, mean, std in zip(bars, stats['mean'], stats['std']):
            label_ = f'{mean:.2f}\n±{std:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std * 1.05,
                   label_, ha='center', va='bottom', fontsize=8)

        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(short_names, rotation=0, ha='center')

        # Set y limits
        ymax = (stats['mean'] + stats['std']).max()
        ymin = 0 
        if metric == "Bias [m] (Test)": #Bias can be negative
            ymax = max(ymax, 1)
            ymin = min(np.nanmin(stats['mean'] - stats['std']), -1)
            #ymin = np.nanmin(ymin, -1)
        ax.set_ylim(ymin * 1.05, ymax * 1.2)

    # Remove empty subplot if exists  
    if len(axes.flatten()) > len(metrics):
        fig.delaxes(axes.flatten()[len(metrics)])


    # Add legend
    handles = [plt.Rectangle((0,0), 1, 1, color=palette[name]) for name in short_names]
    fig.legend(handles, exp_names, title="Experiment",
              loc="center left", bbox_to_anchor=(0.7, 0.3), fontsize='large')

    if "07_aux_task" in exp_names:
        #print("Including auxiliary task F1 scores in the plot.")
        # print a text box including F1 scores for aux tasks mean and std in the bottom right corner
        f1_fmask_mean = df_result["F1 FMASK (Test)"].mean()
        f1_fmask_std = df_result["F1 FMASK (Test)"].std()
        f1_dlt_mean = df_result["F1 DLT (Test)"].mean()
        f1_dlt_std = df_result["F1 DLT (Test)"].std()
        textstr = (
            "Auxiliary Task Scores:\n"
            f'F1 FMASK:  {f1_fmask_mean:.2f} ± {f1_fmask_std:.2f}\n'
            f'F1 DLT:    {f1_dlt_mean:.2f} ± {f1_dlt_std:.2f}'
        )
        props = dict(boxstyle='round', facecolor='#9FBA36', alpha=0.5)
        fig.text(0.71, 0.13, textstr, fontsize=12, bbox=props, verticalalignment='top', horizontalalignment='left')

    
    plt.suptitle(title or "Evaluation Metrics by Experiment", fontsize='x-large')
    plt.tight_layout()

    if printout:
        plt.show()
    else:
        return fig, stats

def save_big_df_stats(run_ids=None, big_df=None, target_folder="drafts"):

    if big_df is None:
        big_df = read_multiple_csv_to_df(run_ids)
    elif run_ids is None:
        raise ValueError("Either run_ids or big_df must be provided.")
    
    # Get unique experiment names and combos
    exp_names = big_df['Experiment'].unique()
    if 'combo' not in big_df.columns:
        big_df['combo'] = '_allin'  # default combo if not present
    combos = big_df["combo"].unique()

    # Select metrics columns
    metrics = [
        "MAE [m] (Test)",
        "nMAE [%] (Test)",
        "RMSE [m] (Test)", 
        "Bias [m] (Test)",
        "R2 [-] (Test)",
        "F1 FMASK (Test)",
        "F1 DLT (Test)"
    ]
    

    path = f"../results/{target_folder}/metrics/"
    path_csv = path + "results_summary.csv"
    path_text_csv = path + "results_text_table.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for combo in combos:
        path_plt = path + f"results_summary_{combo}.pdf"
        subset_df = big_df[big_df["combo"] == combo]
        os.makedirs(os.path.dirname(path_plt), exist_ok=True)
        fig, stats = plot_experiment_metrics_multiple_runs(subset_df, title="Metrics by Experiment for Spatial Config: " + combo, printout=False)
        # Save the plot
        fig.savefig(path_plt, bbox_inches='tight')
        plt.close(fig)

    all_stats = []
    for metric in metrics:
        # Group by both 'Experiment' and 'combo'
        stats_metric = big_df.groupby(['Experiment', 'combo'])[metric].agg(['mean', 'std', 'count']).reset_index()
        stats_metric = stats_metric.rename(columns={
            'mean': f'{metric} Mean', 
            'std': f'{metric} Std', 
            'count': f'{metric} Count'
        })

        all_stats.append(stats_metric)

    # Merge all metrics into a single DataFrame on the 'Experiment' and 'combo' columns
    combined_stats = all_stats[0]
    for stats_metric in all_stats[1:]:
        combined_stats = pd.merge(combined_stats, stats_metric, on=['Experiment', 'combo'])

    # Save the combined statistics to a CSV file
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    combined_stats.to_csv(path_csv, index=False)
    print("Generalization results saved to", path_csv)

    coolstats = readable_metrics_pivot(combined_stats)
    os.makedirs(os.path.dirname(path_text_csv), exist_ok=True)
    coolstats.to_csv(path_text_csv)

    return combined_stats, coolstats

def readable_metrics_pivot(stats):
    """
    Returns a pivoted table: one row per Experiment, columns are metrics (Mean±Std).
    Removes (Test) from metric names and replaces NaN with empty string.
    Keeps 'combo' column as its own column.
    """
    table = []
    metric_prefixes = set()
    for col in stats.columns:
        if col.endswith(" Mean"):
            metric_prefixes.add(col[:-5])
    for idx, row in stats.iterrows():
        experiment = row["Experiment"] if "Experiment" in row else idx
        combo = row["combo"] if "combo" in stats.columns else ""
        row_dict = {"Experiment": experiment, "combo": combo}
        for metric in metric_prefixes:
            clean_metric = metric.replace(" (Test)", "")
            mean_col = f"{metric} Mean"
            std_col = f"{metric} Std"
            if mean_col in stats.columns and std_col in stats.columns:
                mean = row[mean_col]
                std = row[std_col]
                if pd.isna(mean) or pd.isna(std):
                    row_dict[clean_metric] = ""
                else:
                    row_dict[clean_metric] = f"{mean:.2f}±{std:.2f}"
        table.append(row_dict)
    df = pd.DataFrame(table)
    return df.set_index("Experiment")

def generalization_checker(gen_path, allin_path):

    if not os.path.exists(gen_path):
        print("Generalization path does not exist:", gen_path)
        return False

    if not os.path.exists(allin_path):
        print("All-in path does not exist:", allin_path)
        return False

    gen_df = pd.read_csv(gen_path)
    allin_df = pd.read_csv(allin_path)

    def genscore(metric_gen, metric_allin):
        gscore = (1 - abs(metric_gen - metric_allin) / metric_allin) * 100  # percent
        return gscore
    
    def std_prop(std_gen, std_allin):
        sprop = std_gen / std_allin
        return sprop

    # Initialize an empty DataFrame to store the Genscore for each Experiment x Metric x Combo
    metrics = ['MAE [m] (Test)', 'nMAE [%] (Test)', 'RMSE [m] (Test)', 'R2 [-] (Test)']
    genscore_df = pd.DataFrame(columns=['Experiment', 'Metric', 'combo', 'Score', 'Score_Type'])

    for _, gen_row in gen_df.iterrows():
        experiment = gen_row['Experiment']
        combo = gen_row['combo']
        allin_row = allin_df[allin_df['Experiment'] == experiment]
        if allin_row.empty:
            print(f"Experiment {experiment} not found in all-in data.")
            continue

        allin_row = allin_row.iloc[0]
        for metric in metrics:
            gen_mean = gen_row[f'{metric} Mean']
            allin_mean = allin_row[f'{metric} Mean']
            gen_std = gen_row[f'{metric} Std']
            allin_std = allin_row[f'{metric} Std']

            # Calculate mean score
            mean_score = genscore(gen_mean, allin_mean)
            genscore_df = pd.concat([genscore_df, pd.DataFrame([{
                'Experiment': experiment,
                'Metric': metric,
                'combo': combo,
                'Score': mean_score,
                'Score_Type': 'mean_score'
            }])], ignore_index=True)

            # Calculate std factor
            std_factor = std_prop(gen_std, allin_std)
            genscore_df = pd.concat([genscore_df, pd.DataFrame([{
                'Experiment': experiment,
                'Metric': metric,
                'combo': combo,
                'Score': std_factor,
                'Score_Type': 'std_factor'
            }])], ignore_index=True)

    return genscore_df

def generalization_checker_sitespec(gen_path, allin_path):

    if not os.path.exists(gen_path):
        print("Generalization path does not exist:", gen_path)
        return False

    if not os.path.exists(allin_path):
        print("All-in path does not exist:", allin_path)
        return False

    gen_df = pd.read_csv(gen_path)
    allin_df = pd.read_csv(allin_path)

    def genscore(metric_gen, metric_allin):
        gscore = (1 - abs(metric_gen - metric_allin) / metric_allin) * 100  # percent
        return gscore
    
    def std_prop(std_gen, std_allin):
        sprop = std_gen / std_allin
        return sprop

    combo_sites = {
        "_011": 0, # Ebrach
        "_101": 1, # Waldbrunn
        "_110": 2  # Berchtesgaden
    }
    # Initialize an empty DataFrame to store the Genscore for each Experiment x Metric x Combo
    metrics = ['MAE [m] (Test)', 'nMAE [%] (Test)', 'RMSE [m] (Test)', 'R2 [-] (Test)']
    genscore_df = pd.DataFrame(columns=['Experiment', 'Metric', 'combo', 'Score', 'Score_Type'])

    for _, gen_row in gen_df.iterrows():
        experiment = gen_row['Experiment']
        combo = gen_row['combo']
        # allin_row = allin_df[allin_df['Experiment'] == experiment and allin_df['site_id'] == combo_sites[combo]]
        # if allin_row.empty:
        #     print(f"Experiment {experiment} not found in all-in data.")
        #     continue

        # allin_row = allin_row.iloc[0]
        for metric in metrics:
            gen_mean = gen_row[f'{metric} Mean']
            gen_std = gen_row[f'{metric} Std']

            # Find the correct row in allin_df for this experiment, site, and metric
            allin_metric_row = allin_df[
            (allin_df['experiment'] == experiment) &
            (allin_df['site_id'] == combo_sites[combo]) &
            (allin_df['metric'] == metric)
            ]
            if allin_metric_row.empty:
                print(f"Metric {metric} for experiment {experiment} and site {combo_sites[combo]} not found in all-in data.")
                allin_mean = np.nan
                allin_std = np.nan
            else:
                allin_mean = allin_metric_row['mean'].values[0]
                allin_std = allin_metric_row['std'].values[0]

            # Calculate mean score
            mean_score = genscore(gen_mean, allin_mean)
            genscore_df = pd.concat([genscore_df, pd.DataFrame([{
                'Experiment': experiment,
                'Metric': metric,
                'combo': combo,
                'Score': mean_score,
                'Score_Type': 'mean_score'
            }])], ignore_index=True)

            # Calculate std factor
            std_factor = std_prop(gen_std, allin_std)
            genscore_df = pd.concat([genscore_df, pd.DataFrame([{
                'Experiment': experiment,
                'Metric': metric,
                'combo': combo,
                'Score': std_factor,
                'Score_Type': 'std_factor'
            }])], ignore_index=True)

    return genscore_df

def plot_genscore(genscore_df, title="Generalization Score per Metric by Experiment and Train/Test Config", printout=False, savefig=False, targetfolder="drafts"):
    """
    Plot the Generalization Score for each experiment and metric.

    Args:
        genscore_df (pd.DataFrame): DataFrame containing 'Experiment', 'Metric', 'Score', and 'Score_Type'
        title (str): Title of the plot
        printout (bool): Whether to show the plot directly

    """
    import seaborn as sns

    combos = genscore_df['combo'].unique()
    metrics = genscore_df['Metric'].unique()
    combodict = {
        "_011": "Ebrach",
        "_101": "Waldbrunn",
        "_110": "Berchtesgaden"
    }

    # Custom color palette
    palette = [0x072140, 0x3070B3, 0x8F81EA, 0xB55CA5, 0xFED702, 0xF7B11E, 0x9FBA36]
    palette = [f'#{color:06X}' for color in palette]
    exp_names = genscore_df['Experiment'].unique()
    exp_palette = {exp: palette[i % len(palette)] for i, exp in enumerate(exp_names)}

    # Set up the matplotlib figure
    fig, axes = plt.subplots(len(combos), 1, figsize=(10, 3 * len(combos)), sharey=False)

    if len(combos) == 1:
        axes = [axes]  # Ensure axes is iterable for a single combo

    for i, combo in enumerate(combos):
        ax = axes[i]
        subset_df = genscore_df[genscore_df['combo'] == combo]
        sns.barplot(
            data=subset_df,
            x='Metric',
            y='Score',
            hue='Experiment',
            ax=ax,
            palette=[exp_palette[exp] for exp in subset_df['Experiment'].unique()]
        )
        ax.set_title(f'Train/Test Config: {combo} (Test Site: {combodict[combo]})')
        ax.set_ylabel('Generalization Score (%)')
        ax.set_xlabel('Metric')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylim(0, 100)
        # if subset_df['Score'].min() < 0:
        #     ax.set_ylim(max(-100, subset_df['Score'].min()), 100)
        ax.legend(title='Experiment', bbox_to_anchor=(1.02, 0.5), loc='center left')

    plt.suptitle(title, fontsize='x-large')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)


    if printout:
        plt.show()
    if savefig:
        path = f"../results/gen/{targetfolder}/gscore/"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        path_fig = path + "genscore"+ ".pdf"
        fig.savefig(path_fig, bbox_inches='tight')
        print("Generalization score plot saved to", path_fig)
    if printout:
        plt.show()
        return fig
    else:
        plt.close(fig)


def plot_comparison_all_gen(master_df, title="Comparison of Metrics Across Experiments", printout=False, savefig =True, targetfolder = "drafts"):
    """
    Plot comparison of metrics across experiments and combos, with one figure per combo
    and subplots for each metric. Includes a reference value for the "_allin" combo.

    Args:
        master_df (pd.DataFrame): DataFrame containing 'Experiment', 'combo', 'Metrics', and 'Value'
        title (str): Title of the plot
        printout (bool): Whether to show the plot directly
    """

    # Extract unique combos and metrics
    combos = master_df['combo'].unique()
    metrics = master_df['Metrics'].unique()
    exp_names = master_df['Experiment'].unique()
    # Create short names for x-axis
    short_names = [str(i+1) for i in range(len(exp_names))]
    combodict = {
        "_011": "Ebrach",
        "_101": "Waldbrunn",
        "_110": "Berchtesgaden"
    }
    globalmax =[12, 40, 14, 6, 2]
    globalmin =[0,0,0,-10,-2]

    # Custom color palette
    palette = [0x072140, 0x3070B3, 0x8F81EA, 0xB55CA5, 0xFED702, 0xF7B11E, 0x9FBA36]
    legend_palette = {str(i+1): f'#{palette[i]:06x}' for i in range(len(palette))}
    palette = [f'#{color:06X}' for color in palette]
    exp_names = master_df['Experiment'].unique()
    exp_palette = {exp: palette[i % len(palette)] for i, exp in enumerate(exp_names)}

    figures = []

    for combo in combos:
        if combo == "_allin":
            continue  # Skip "_allin" as it is used as a reference

        subset_df = master_df[master_df['combo'] == combo]
        allin_df = master_df[master_df['combo'] == "_allin"]

        # Separate mean and std values
        mean_df = subset_df[subset_df['Metrics'].str.contains('Mean')].copy()
        std_df = subset_df[subset_df['Metrics'].str.contains('Std')].copy()
        allin_mean_df = allin_df[allin_df['Metrics'].str.contains('Mean')].copy()

        # Merge mean and std for error bars
        mean_df['Metric'] = mean_df['Metrics'].str.replace(' Mean', '')
        std_df['Metric'] = std_df['Metrics'].str.replace(' Std', '')
        allin_mean_df['Metric'] = allin_mean_df['Metrics'].str.replace(' Mean', '')

        merged_df = pd.merge(mean_df, std_df, on=['Experiment', 'combo', 'Metric'], suffixes=('_mean', '_std'))
        merged_df = pd.merge(merged_df, allin_mean_df, on=['Experiment', 'Metric'], suffixes=('', '_allin'))
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        metric_labels = merged_df['Metric'].unique()
        metric_label_short = [label.split('(')[0].strip() for label in metric_labels]

        for i, (metric, metric_short) in enumerate(zip(metric_labels, metric_label_short)):
            if i>=5:
                continue  # Limit to first 5 metrics for 2x3 grid
            ax = axes[i // 3, i % 3]
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            metric_data = merged_df[merged_df['Metric'] == metric]

            # Plot bars
            bars = ax.bar(
                range(len(metric_data)),
                metric_data['Value_mean'],
                yerr=metric_data['Value_std'],
                capsize=5,
                color=[exp_palette[exp] for exp in metric_data['Experiment']]
            )

            # Add reference "_allin" values as crosses
            for j, (bar, mean, allin_mean) in enumerate(zip(bars, metric_data['Value_mean'], metric_data['Value'])):
                ax.scatter(bar.get_x() + bar.get_width() / 2, allin_mean, color='red', marker='x', label='_allin' if j == 0 else "")

            # Add value annotations
            for bar, mean, std in zip(bars, metric_data['Value_mean'], metric_data['Value_std']):
                label_ = f'{mean:.2f}\n±{std:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std * 1.05,
                        label_, ha='center', va='bottom', fontsize=8)

            ax.set_title(metric_short)
            ax.set_xlabel("")
            ax.set_ylabel(metric_short)
            ax.set_xticks(range(len(metric_data)))
            ax.set_xticklabels(short_names, rotation=0, ha='center')

            # Set y limits
            # ymax = max((metric_data['Value_mean'] + metric_data['Value_std']).max(), 1)
            # ymin = min((metric_data['Value_mean'] - metric_data['Value_std']).min(), 0)
            # ymin = min(ymin, metric_data["Value"].min())
            # ax.set_ylim(ymin * 1.05, ymax * 1.2)
            ymax = globalmax[i]
            ymin = globalmin[i]
            ax.set_ylim(ymin,ymax)

        # Remove empty subplot if exists
        if len(axes.flatten()) > 5: #len(metric_labels)
            fig.delaxes(axes.flatten()[5])

        # Add legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=legend_palette[name]) for name in short_names]
        handles.append(plt.Line2D([0], [0], color='red', marker='x', linestyle='', label='All-In Reference'))
        fig.legend(handles, exp_names.tolist() + ['All-In Reference'], title="Experiment",
                   loc="center left", bbox_to_anchor=(0.7, 0.3), fontsize='large')

        if "07_aux_task" in exp_names:
            #print("Including auxiliary task F1 scores in the plot.")
            # Helper function for safe extraction
            def get_metric_value(df, experiment, metric_name):
                row = df[(df["Experiment"] == experiment) & (df["Metrics"] == metric_name)]
                if not row.empty:
                    return row["Value"].values[0]
                else:
                    return np.nan

            f1_fmask_mean = get_metric_value(mean_df, "07_aux_task", "F1 FMASK (Test) Mean")
            f1_fmask_std = get_metric_value(std_df, "07_aux_task", "F1 FMASK (Test) Std")
            f1_dlt_mean = get_metric_value(mean_df, "07_aux_task", "F1 DLT (Test) Mean")
            f1_dlt_std = get_metric_value(std_df, "07_aux_task", "F1 DLT (Test) Std")
            textstr = (
                "Auxiliary Task Scores:\n"
                f'F1 FMASK:  {f1_fmask_mean:.2f} ± {f1_fmask_std:.2f}\n'
                f'F1 DLT:    {f1_dlt_mean:.2f} ± {f1_dlt_std:.2f}'
            )
            props = dict(boxstyle='round', facecolor='#9FBA36', alpha=0.5)
            fig.text(0.71, 0.1, textstr, fontsize=12, bbox=props, verticalalignment='top', horizontalalignment='left')


        plt.suptitle(title + (f' Train/Test Config: {combo} (Test Site: {combodict[combo]})'), fontsize='x-large')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)


        if savefig:
            figures.append(fig)
            path = f"../results/gen/{targetfolder}/gscore/"
            print(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            path_fig = path + "absolute_" + combo + ".pdf"
            fig.savefig(path_fig, bbox_inches='tight')
        if printout:
            plt.show()
        else: plt.close(fig)
    

    return figures


def plot_comparison_all_gen_sitespec(master_df, title="Comparison of Metrics Across Experiments", printout=False, savefig=True, targetfolder="drafts"):
    """
    Plot comparison of metrics across experiments and combos, with one figure per combo
    and subplots for each metric. Includes combo-specific reference values from Allin_Mean.

    Args:
        master_df (pd.DataFrame): DataFrame containing 'Experiment', 'combo', 'Metrics', 'Gen_Mean', 'Gen_Std', 'Allin_Mean', 'Allin_Std'
        title (str): Title of the plot
        printout (bool): Whether to show the plot directly
        savefig (bool): Whether to save figures to disk
        targetfolder (str): Target folder for saving figures
    """

    # Extract unique combos and experiments
    combos = master_df['combo'].unique()
    exp_names = master_df['Experiment'].unique()
    
    # Create short names for x-axis
    short_names = [str(i+1) for i in range(len(exp_names))]
    combodict = {
        "_011": "Ebrach",
        "_101": "Waldbrunn",
        "_110": "Berchtesgaden"
    }
    globalmax = [12, 40, 14, 6, 2]
    globalmin = [0, 0, 0, -10, -2]

    # Custom color palette
    palette = [0x072140, 0x3070B3, 0x8F81EA, 0xB55CA5, 0xFED702, 0xF7B11E, 0x9FBA36]
    legend_palette = {str(i+1): f'#{palette[i]:06x}' for i in range(len(palette))}
    palette = [f'#{color:06X}' for color in palette]
    exp_palette = {exp: palette[i % len(palette)] for i, exp in enumerate(exp_names)}

    figures = []

    for combo in combos:
        subset_df = master_df[master_df['combo'] == combo].copy()
        
        if subset_df.empty:
            continue

        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        metric_labels = subset_df['Metrics'].unique()
        metric_label_short = [label.split('(')[0].strip() for label in metric_labels]

        for i, (metric, metric_short) in enumerate(zip(metric_labels, metric_label_short)):
            if i >= 5:
                continue  # Limit to first 5 metrics for 2x3 grid
            ax = axes[i // 3, i % 3]
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            metric_data = subset_df[subset_df['Metrics'] == metric].copy()

            # Plot bars for Gen_Mean with Gen_Std as error bars
            bars = ax.bar(
                range(len(metric_data)),
                metric_data['Gen_Mean'],
                yerr=metric_data['Gen_Std'],
                capsize=5,
                color=[exp_palette[exp] for exp in metric_data['Experiment']]
            )

            # Add reference Allin_Mean values as crosses
            for j, (bar, gen_mean, allin_mean) in enumerate(zip(bars, metric_data['Gen_Mean'], metric_data['Allin_Mean'])):
                ax.scatter(bar.get_x() + bar.get_width() / 2, allin_mean, color='red', marker='x', s=100, linewidths=2, 
                          label='Allin Reference' if j == 0 else "")

            # Add value annotations
            for bar, gen_mean, gen_std in zip(bars, metric_data['Gen_Mean'], metric_data['Gen_Std']):
                label_ = f'{gen_mean:.2f}\n±{gen_std:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + gen_std * 1.05,
                        label_, ha='center', va='bottom', fontsize=8)

            ax.set_title(metric_short)
            ax.set_xlabel("")
            ax.set_ylabel(metric_short)
            ax.set_xticks(range(len(metric_data)))
            ax.set_xticklabels(short_names, rotation=0, ha='center')

            # Set y limits
            ymax = globalmax[i]
            ymin = globalmin[i]
            ax.set_ylim(ymin, ymax)

        # Remove empty subplot if exists
        if len(axes.flatten()) > 5:
            fig.delaxes(axes.flatten()[5])

        # Add legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=legend_palette[name]) for name in short_names]
        handles.append(plt.Line2D([0], [0], color='red', marker='x', linestyle='', markersize=8, markeredgewidth=2, label='Allin Reference'))
        fig.legend(handles, list(exp_names) + ['Allin Reference (Mean)'], title="Experiment",
                   loc="center left", bbox_to_anchor=(0.7, 0.3), fontsize='large')

        if "07_aux_task" in exp_names:
            # Helper function for safe extraction
            def get_metric_value(df, experiment, metric_name, allin=False):
                row = df[(df["Experiment"] == experiment) & (df["Metrics"] == metric_name)]
                if not row.empty:
                    return row["Gen_Mean"].values[0] if not allin else row["Allin_Mean"].values[0]
                else:
                    return np.nan

            def get_metric_std(df, experiment, metric_name, allin=False):
                row = df[(df["Experiment"] == experiment) & (df["Metrics"] == metric_name)]
                if not row.empty:
                    return row["Gen_Std"].values[0] if not allin else row["Allin_Std"].values[0]
                else:
                    return np.nan

            f1_fmask_mean = get_metric_value(subset_df, "07_aux_task", "F1 FMASK (Test)")
            f1_fmask_std = get_metric_std(subset_df, "07_aux_task", "F1 FMASK (Test)")
            f1_dlt_mean = get_metric_value(subset_df, "07_aux_task", "F1 DLT (Test)")
            f1_dlt_std = get_metric_std(subset_df, "07_aux_task", "F1 DLT (Test)")
            f1_fmask_allin = get_metric_value(subset_df, "07_aux_task", "F1 FMASK (Test)", allin=True)
            f1_fmask_allin_std = get_metric_std(subset_df, "07_aux_task", "F1 FMASK (Test)", allin=True)
            f1_dlt_allin = get_metric_value(subset_df, "07_aux_task", "F1 DLT (Test)", allin=True)
            f1_dlt_allin_std = get_metric_std(subset_df, "07_aux_task", "F1 DLT (Test)", allin=True)

            textstr = (
                "Auxiliary Task Scores:\n"
                f'F1 FMASK:  {f1_fmask_mean:.2f} ± {f1_fmask_std:.2f} All-In: {f1_fmask_allin:.2f} ± {f1_fmask_allin_std:.2f}\n'
                f'F1 DLT:    {f1_dlt_mean:.2f} ± {f1_dlt_std:.2f} All-In: {f1_dlt_allin:.2f} ± {f1_dlt_allin_std:.2f}'
            )
            props = dict(boxstyle='round', facecolor='#9FBA36', alpha=0.5)
            fig.text(0.71, 0.1, textstr, fontsize=12, bbox=props, verticalalignment='top', horizontalalignment='left')

        plt.suptitle(title + (f' Train/Test Config: {combo} (Test Site: {combodict.get(combo, combo)})'), fontsize='x-large')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if savefig:
            figures.append(fig)
            path = f"../results/gen/{targetfolder}/gscore/"
            print(path)
            os.makedirs(path, exist_ok=True)
            path_fig = path + "absolute_" + combo + ".pdf"
            fig.savefig(path_fig, bbox_inches='tight')
        
        if printout:
            plt.show()
        else:
            plt.close(fig)

    return figures

def plot_comparison_all_gen_backup(master_df, title="Comparison of Metrics Across Experiments and Combos", printout=False):
    """
    Plot comparison of metrics across experiments and combos, with one figure per combo
    and subplots for each metric.

    Args:
        master_df (pd.DataFrame): DataFrame containing 'Experiment', 'combo', 'Metrics', and 'Value'
        title (str): Title of the plot
        printout (bool): Whether to show the plot directly
    """

    # Extract unique combos and metrics
    combos = master_df['combo'].unique()
    metrics = master_df['Metrics'].unique()
    exp_names = master_df['Experiment'].unique()
    # Create short names for x-axis
    short_names = [str(i+1) for i in range(len(exp_names))]

    # Custom color palette
    palette = [0x072140, 0x3070B3, 0x8F81EA, 0xB55CA5, 0xFED702, 0xF7B11E, 0x9FBA36]
    legend_palette = {str(i+1): f'#{palette[i]:06x}' for i in range(len(palette))}
    palette = [f'#{color:06X}' for color in palette]
    exp_names = master_df['Experiment'].unique()
    exp_palette = {exp: palette[i % len(palette)] for i, exp in enumerate(exp_names)}

    figures = []

    for combo in combos:
        subset_df = master_df[master_df['combo'] == combo]

        # Separate mean and std values
        mean_df = subset_df[subset_df['Metrics'].str.contains('Mean')].copy()
        std_df = subset_df[subset_df['Metrics'].str.contains('Std')].copy()

        # Merge mean and std for error bars
        mean_df['Metric'] = mean_df['Metrics'].str.replace(' Mean', '')
        std_df['Metric'] = std_df['Metrics'].str.replace(' Std', '')
        merged_df = pd.merge(mean_df, std_df, on=['Experiment', 'combo', 'Metric'], suffixes=('_mean', '_std'))

        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        metric_labels = merged_df['Metric'].unique()

        for i, metric in enumerate(metric_labels):
            ax = axes[i // 3, i % 3]
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

            metric_data = merged_df[merged_df['Metric'] == metric]

            # Plot bars
            bars = ax.bar(
                range(len(metric_data)),
                metric_data['Value_mean'],
                yerr=metric_data['Value_std'],

                capsize=5,
                color=[exp_palette[exp] for exp in metric_data['Experiment']]
            )

            # Add value annotations
            for bar, mean, std in zip(bars, metric_data['Value_mean'], metric_data['Value_std']):
                label_ = f'{mean:.2f}\n±{std:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std * 1.05,
                        label_, ha='center', va='bottom', fontsize=8)

            ax.set_title(metric)
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(metric_data)))
            ax.set_xticklabels(short_names, rotation=0, ha='center')

            # Set y limits
            ymax = max((metric_data['Value_mean'] + metric_data['Value_std']).max(), 1)
            ymin = min((metric_data['Value_mean'] - metric_data['Value_std']).min(), 0)
            ax.set_ylim(ymin * 1.05, ymax * 1.2)

        # Remove empty subplot if exists
        if len(axes.flatten()) > len(metric_labels):
            fig.delaxes(axes.flatten()[len(metric_labels)])

            # Add legend
        handles = [plt.Rectangle((0,0), 1, 1, color=legend_palette[name]) for name in short_names]
        fig.legend(handles, exp_names, title="Experiment",
            loc="center left", bbox_to_anchor=(0.7, 0.25), fontsize='large')

        plt.suptitle(f"{title} - Combo: {combo}", fontsize='x-large')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if printout:
            plt.show()
        else:
            figures.append(fig)

    return figures

def plot_error_over_frequency_backup(report, bins=80, title = "Error vs. GT Distribution"):
    """
    Plot the error distribution over frequency for validation and test sets.

    Parameters:
    - y_val_true: np.ndarray, true values for validation set
    - y_val_pred: np.ndarray, predicted values for validation set
    - y_test_true: np.ndarray, true values for test set
    - y_test_pred: np.ndarray, predicted values for test set
    - bins: int, number of bins for histogram
    """

    y_val_true, y_val_pred, y_test_true, y_test_pred = report["targets"]["validation"].copy(), report["predictions"]["validation"].copy(), report["targets"]["test"].copy(), report["predictions"]["test"].copy()
    
    val_errors = y_val_pred - y_val_true
    test_errors = y_test_pred - y_test_true
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Validation set
    val_bin_means, val_bin_edges, _ = stats.binned_statistic(
        y_val_true.flatten(), val_errors.flatten(), statistic='mean', bins=bins
    )
    val_bin_counts, _, _ = stats.binned_statistic(
        y_val_true.flatten(), y_val_true.flatten(), statistic='count', bins=val_bin_edges
    )
    # mask out val_bin_means where counts are less than 5
    mask = val_bin_counts < 10
    val_bin_means[mask] = np.nan

    axes[0].plot(
        (val_bin_edges[:-1] + val_bin_edges[1:]) / 2, val_bin_means, label='Mean Error', color='blue'
    )
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Mean Error (Predicted - True)')
    axes[0].set_title('Validation Error vs. Ground Truth')
    axes[0].grid(True)
    axes[0].set_xlim(0, 50)
    axes[0].set_ylim(-20, 20)
    # Overlay ground truth distribution (dashed line, scaled for visibility)
    gt_dist = val_bin_counts / val_bin_counts.max() * 18
    axes[0].plot(
        (val_bin_edges[:-1] + val_bin_edges[1:]) / 2, gt_dist, '--', color='gray', label='GT Distribution (scaled)'
    )
    axes[0].legend(loc='lower left')

    # Test set
    test_bin_means, test_bin_edges, _ = stats.binned_statistic(
        y_test_true.flatten(), test_errors.flatten(), statistic='mean', bins=bins
    )
    test_bin_counts, _, _ = stats.binned_statistic(
        y_test_true.flatten(), y_test_true.flatten(), statistic='count', bins=test_bin_edges
    )
    # mask out val_bin_means where counts are less than 5
    mask = test_bin_counts < 10
    test_bin_means[mask] = np.nan

    axes[1].plot(
        (test_bin_edges[:-1] + test_bin_edges[1:]) / 2, test_bin_means, label='Mean Error', color='green'
    )
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Mean Error (Predicted - True)')
    axes[1].set_title('Test Error vs. Ground Truth')
    axes[1].grid(True)
    axes[1].set_xlim(0, 50)
    axes[1].set_ylim(-20, 20)
    # Overlay ground truth distribution (dashed line, scaled for visibility)
    #gt_dist_test = test_bin_counts / test_bin_counts.max() * np.nanmax(np.abs(test_bin_means))
    gt_dist_test = test_bin_counts / test_bin_counts.max() * 18

    axes[1].plot(
        (test_bin_edges[:-1] + test_bin_edges[1:]) / 2, gt_dist_test, '--', color='gray', label='GT Distribution (scaled)'
    )
    axes[1].legend(loc='lower left')

    fig.suptitle(title)
    plt.tight_layout()
    # save_plot(plt.gcf(), "error_vs_gt", report)
    # plt.show()
    return fig