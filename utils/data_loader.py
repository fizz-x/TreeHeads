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


# ## example that works:
    # QUANTILE_IDX = {"Q25": 0, "Q50": 1, "Q75": 2, "AVG": 3, "STD": 4}
    # stack = []
    # seasons = ["summer", "spring"]
    # quantiles = ["Q50", "Q75"]

    # for season in seasons:
    #     arr, desc = read_multiband_tif_as_stack(site_paths["SITE1"]["S2"][season])  # (bands, channels, H, W)
    #     arr = arr[:, [QUANTILE_IDX[q] for q in quantiles], :, :]                  # select quantiles
    #     arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])                       # (bands*quantiles, H, W)
    #     stack.append(arr)
    #     print(f"Season {season} shape:", arr.shape)

    # # Aux inputs
    # auxes = ['DLT', 'DEM']
    # for aux in auxes:
    #     arr = load_raster(site_paths["SITE1"][aux])                                     # (1, H, W)
    #     stack.append(arr)

    # X = np.concatenate(stack, axis=0)                                             # (C, H, W)
    # print("Final stack shape:", X.shape)


    # # function to be adapted:
    # stack = []

    # # Spectral inputs
    # seasons = cfg["spectral"]["seasons"]
    # quantiles = cfg["spectral"]["quantiles"]
    # for season in seasons:
    #     arr, desc = read_multiband_tif_as_stack(site_paths["S2"][season])  # (bands, channels, H, W)
    #     selected = [arr[:, QUANTILE_IDX[q], :, :] for q in quantiles]
    #     selected = [x.reshape(-1, x.shape[-2], x.shape[-1]) for x in selected]
    #     stack.extend(selected)

    # # Aux inputs
    # for aux in cfg["aux_inputs"]:
    #     arr = load_raster(site_paths[aux])  # (H,W)
    #     stack.append(arr)

    # X = np.stack(stack, axis=0)  # (C,H,W)


    # # Outputs
    # Y = {}
    # for name, out_cfg in cfg["outputs"].items():
    #     target = out_cfg["target"]
    #     arr = load_raster(site_paths[target])
    #     if arr.ndim == 3:
    #         arr = arr[0]
    #     Y[name] = arr

    # return X, Y


def extract_patches(X, Y, patch_size=32, nan_percent_allowed=8):
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

    return X_patches, Y_patches



def build_patched_dataset(cfg, sites_dict, patch_size=32):
    X_patches, Y_patches = [], []

    for site_name, site_paths in sites_dict.items():
        X, Y = build_site_data(cfg, site_paths)
        Xp, Yp = extract_patches(X, Y, patch_size)  # (N, C, ps, ps)
        #Yp = {k: extract_patches(v[np.newaxis,:,:], patch_size) for k,v in Y.items()}

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
