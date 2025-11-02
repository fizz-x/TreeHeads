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

            # Check NaN percentage for each channel
            x_nan_per_channel = np.isnan(x_patch).mean(axis=(1,2)) * 100  # (C,)
            y_nan_per_channel = np.isnan(y_patch).mean(axis=(1,2)) * 100  # (num_outputs,)
            if (np.all(x_nan_per_channel <= nan_percent_allowed) and 
                np.all(y_nan_per_channel <= nan_percent_allowed)):

                X_patches.append(x_patch)
                Y_patches.append(y_patch)
            elif fullmap:
                X_patches.append(np.full((C, patch_size, patch_size), np.nan, dtype=np.float32))
                Y_patches.append(np.full((Y.shape[0], patch_size, patch_size), np.nan, dtype=np.float32))

    return X_patches, Y_patches



def build_patched_dataset(cfg, sites_dict, patch_size=32, nan_percent_allowed=20):
    X_patches, Y_patches = [], []
    site_indices = {}
    current_index = 0
    site_nums = [] # will be an array with the length of the total patches, with the site number for each patch.
    site_num = 0

    for site_name, site_paths in sites_dict.items():
        X, Y = build_site_data(cfg, site_paths)
        Xp, Yp = extract_patches(X, Y, patch_size, nan_percent_allowed=nan_percent_allowed)  # (N, C, ps, ps)

        #print(f"Site {site_name}: extracted {len(Xp)} patches.")
        X_patches.append(Xp)
        Y_patches.append(Yp)
        site_indices[site_name] = (current_index, current_index + len(Xp))
        current_index += len(Xp)
        site_nums.extend([site_num] * len(Xp))
        site_num += 1

    X_patches = np.concatenate(X_patches, axis=0)  # (N, C, ps, ps)
    Y_patches = np.concatenate(Y_patches, axis=0)  # (N, C, ps, ps), e.g. (1138, 3, 32, 32)
    site_nums = np.array(site_nums)

    # # add one dim to Y ((1138, 3, 32, 32) --> (1138, 1, 3, 32, 32))
    # Y_patches = Y_patches[np.newaxis, ...]
    # # then write the site number to that new dim, so that each patch has the site number in that dim.
    # # e.g. if site1 has 400 patches, site2 has 300 patches, site3 has 438 patches, then the first 400 patches will have 0 in that dim, the next 300 patches will have 1 in that dim, and the last 438 patches will have 2 in that dim.
    # site_num = 0
    # for site_name, (start_idx, end_idx) in site_indices.items():
    #     Y_patches[start_idx:end_idx, :, :, :, :] = site_num
    #     site_num += 1

    #print(f"Total patches extracted: {len(X_patches)}, x32^2 = {len(X_patches)*32*32} pixels.")
    
    return X_patches, Y_patches, site_nums

def build_patched_dataset_generalization(cfg, sites, combo, patch_size=32, nan_percent_allowed=20):
    
    # combos can be ["110","101","011"] #1 is training data, 0 test. logic is LSB: 001 -> SITE1 is training data. 
    # Train data, where combo == 1
    test_sites = {}
    train_sites = sites.copy()
    site_names = list(sites.keys())
    # keep the site that is 0 in test_sites dict depending on combo.
    for i, c in enumerate(combo):
        if c == "0":
            site_name = site_names[i]
            test_sites[site_name] = sites[site_name]
            train_sites.pop(site_name, None)

            # now for both we overwrite the CHM_norm and CHM_norm_params depending on the combo
            # Overwrite CHM_norm and CHM_norm_params for test and train sites based on the combo
            if site_name in test_sites:
                test_sites[site_name]["CHM_norm"] = test_sites[site_name]["CHM_norm_combo"][f"_{combo}"]["path"]
                test_sites[site_name]["CHM_norm_params"] = {
                    "mu": test_sites[site_name]["CHM_norm_combo"][f"_{combo}"]["mu"],
                    "std": test_sites[site_name]["CHM_norm_combo"][f"_{combo}"]["std"],
                }
            elif site_name in train_sites:
                train_sites[site_name]["CHM_norm"] = train_sites[site_name]["CHM_norm_combo"][f"_{combo}"]["path"]
                train_sites[site_name]["CHM_norm_params"] = {
                    "mu": train_sites[site_name]["CHM_norm_combo"][f"_{combo}"]["mu"],
                    "std": train_sites[site_name]["CHM_norm_combo"][f"_{combo}"]["std"],
                }

    X_patch_train, Y_patch_train = [], []
    for site_name, site_paths in train_sites.items():
        X, Y = build_site_data(cfg, site_paths)
        Xp, Yp = extract_patches(X, Y, patch_size, nan_percent_allowed=nan_percent_allowed)  # (N, C, ps, ps)

        X_patch_train.append(Xp)
        Y_patch_train.append(Yp)

    X_patch_train = np.concatenate(X_patch_train, axis=0)  # (N, C, ps, ps)
    Y_patch_train = np.concatenate(Y_patch_train, axis=0)  # (N, C, ps, ps)

    # Test data, where combo == 0
    X_patch_test, Y_patch_test = [], []
    for site_name, site_paths in test_sites.items():
        X, Y = build_site_data(cfg, site_paths)
        Xp, Yp = extract_patches(X, Y, patch_size, nan_percent_allowed=nan_percent_allowed)  # (N, C, ps, ps)
        X_patch_test.append(Xp)
        Y_patch_test.append(Yp)

    X_patch_test = np.concatenate(X_patch_test, axis=0)  # (N, C, ps, ps)
    Y_patch_test = np.concatenate(Y_patch_test, axis=0)  # (N, C, ps, ps)

    return X_patch_train, Y_patch_train, X_patch_test, Y_patch_test

def split_dataset(X_patches, Y_patches, train_size=0.7, val_size=0.15, test_size=0.15, seed=42, site_indices=None):
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
    if site_indices is not None:
        _, temp = train_test_split(
            site_indices, test_size=(val_size + test_size), random_state=seed
        )
        _, site_indices_test = train_test_split(
            temp, test_size=test_size/(val_size + test_size), random_state=seed
        )
    else:
        site_indices_test = None

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), site_indices_test

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