import rasterio
import numpy as np
#from .config_loader import get_config
from sklearn.model_selection import train_test_split
#from .data_loader import load_raster, QUANTILE_IDX

# map quantile names to band indices
QUANTILE_IDX = {"Q25": 0, "Q50": 1, "Q75": 2, "AVG": 3, "STD": 4}

def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # shape (bands, H, W)
        # if src.nodata is not None:
        #     arr[arr == src.nodata] = np.nan
    return arr

def read_multiband_tif_as_stack(tif_path, bands=13, channels=5):
    """
    Reads a stacked multiband TIFF and returns a numpy array of shape (bands, channels, h, w).

    Args:
        tif_path (str): Path to the stacked multiband TIFF.
        bands (int): Number of bands.
        channels (int): Number of channels per band.

    Returns:
        np.ndarray: Array of shape (bands, channels, h, w).
    """
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        arr = src.read()  # shape: (bands*channels, h, w)
        # if src.nodata is not None:
        #     arr[arr == src.nodata] = np.nan
        arr = arr.reshape((bands, channels, h, w))
        desc = src.descriptions  # List of band names
    return arr, desc

def build_site_data(cfg, site_paths):
    stack = []

    # Spectral inputs
    seasons = cfg["spectral"]["seasons"]
    quantiles = cfg["spectral"]["quantiles"]
    for season in seasons:
        arr, desc = read_multiband_tif_as_stack(site_paths["S2_norm"][season])  # (bands, channels, H, W)
        arr = arr[:, [QUANTILE_IDX[q] for q in quantiles], :, :]           # select quantiles
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])                # (bands*quantiles, H, W)
        stack.append(arr)

    # Aux inputs
    for aux in cfg["aux_inputs"]:
        arr = load_raster(site_paths[aux])  # (1, H, W) or (H, W)
        if aux == "DEM":
            arr = arr / 2000.0
        # if aux == "DLT":
        #     arr = arr / 3.0
        stack.append(arr)

    X = np.concatenate(stack, axis=0)  # (C, H, W)

    # Outputs
    # Y = {}
    stacky = []
    for name, out_cfg in cfg["outputs"].items():
        target = out_cfg["target"]
        arr = load_raster(site_paths[target])
        if arr.ndim == 3:
            arr = arr[0]
        #Y[name] = arr
        stacky.append(arr)
        Y = np.stack(stacky, axis=0)  # (num_outputs, H, W)

    return X, Y




def extract_patches(X, Y, patch_size=32, nan_percent_allowed=8, fullmap=False):
    """
    Extract patches for the input data and corresponding labels.
    Args:
        X (np.ndarray): Input data of shape (C, H, W).
        Y (np.ndarray): Label data of shape (H, W) or (1, H, W).
        patch_size (int): Size of the square patches to extract.
        nan_percent_allowed (float): Percentage of NaN values allowed in a patch.
                                     Patches with more NaNs will be discarded.
    allow maximum nan percentage in each patch, <nan_percent_allowed> for X and Y seperately. only take the patch if it meets the criteria for both X and Y.
    """
    # Implementation goes here
    X_patches, Y_patches = [], []
    C, H, W = X.shape

    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            x_patch = X[:, i:i + patch_size, j:j + patch_size]
            y_patch = Y[:, i:i + patch_size, j:j + patch_size]

            # Check NaN percentage
            if (np.isnan(x_patch).mean() * 100 <= nan_percent_allowed and
                np.isnan(y_patch).mean() * 100 <= nan_percent_allowed):
                X_patches.append(x_patch)
                Y_patches.append(y_patch)
            elif fullmap:
                X_patches.append(np.full((C, patch_size, patch_size), np.nan, dtype=np.float32))
                Y_patches.append(np.full((Y.shape[0], patch_size, patch_size), np.nan, dtype=np.float32))

    return X_patches, Y_patches



def build_patched_dataset(cfg, sites_dict, patch_size=32):
    X_patches, Y_patches = [], []

    for site_name, site_paths in sites_dict.items():
        X, Y = build_site_data(cfg, site_paths)
        Xp, Yp = extract_patches(X, Y, patch_size)  # (N, C, ps, ps)

        X_patches.append(Xp)
        Y_patches.append(Yp)

    X_patches = np.concatenate(X_patches, axis=0)  # (N, C, ps, ps)
    Y_patches = np.concatenate(Y_patches, axis=0)  # (N, C, ps, ps)
    
    return X_patches, Y_patches

def split_dataset(X_patches, Y_patches, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    """
    Splits the dataset into train/val/test sets (70/15/15) using sklearn's train_test_split.
    Args:
        X_patches (np.ndarray): Input patches, shape (N, C, ps, ps).
        Y_patches (np.ndarray): Output patches, shape (N, C, ps, ps).
        train_size (float): Proportion of the dataset to include in the train split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    Returns:
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_patches, Y_patches, test_size=(val_size + test_size), random_state=seed
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=test_size/(val_size + test_size), random_state=seed
    )
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def reconstruct_from_patches(predictions_denorm, target_shape, patch_size=32, overlap=0):
    """
    Reconstruct the full prediction map from patches.
    
    Args:
        predictions_denorm (list of np.ndarray): List of predicted patches, each of shape (N, C, ps, ps). e.g. (441, 1, 32, 32)
        target_shape (tuple): Shape of the target full map (C, H, W). e.g. (1, 846, 1241)
        patch_size (int): Size of each patch (ps).
        overlap (int): Overlap between patches.
        
    Returns:
        np.ndarray: Reconstructed full prediction map of shape (C, H, W).
    """
    C = predictions_denorm.shape[1]
    H, W = target_shape[1], target_shape[2]
    full_map = np.zeros((C, H, W), dtype=np.float32)
    count_map = np.zeros((C, H, W), dtype=np.float32)

    step = patch_size - overlap
    idx = 0
    for i in range(0, H - patch_size + 1, step):
        for j in range(0, W - patch_size + 1, step):
            if idx < len(predictions_denorm):
                full_map[:, i:i + patch_size, j:j + patch_size] += predictions_denorm[idx]
                count_map[:, i:i + patch_size, j:j + patch_size] += 1
                idx += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1
    full_map /= count_map

    return full_map