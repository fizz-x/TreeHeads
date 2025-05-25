import os
import numpy as np
import rasterio

# load the stacked S2 and merged ALS rasters
def load_rasters(S2_stack_path, ALS_Path, verbose=True):
    """
    Load Sentinel-2 stacked raster and ALS raster, print their properties.
    Args:
        S2_stack_path (str): Path to the Sentinel-2 stacked raster.
        ALS_Path (str): Path to the ALS raster.
        verbose (bool): If True, print additional information.
    Returns:
        tuple: Numpy arrays of the S2 and ALS rasters.
    """
    
    with rasterio.open(S2_stack_path) as s2_src:
        s2np = s2_src.read().astype(np.float32)  # shape: (bands, height, width)
        s2res_x, s2res_y = s2_src.res  # (pixel width, pixel height in coordinate units)
        s2crs = s2_src.crs
    
    with rasterio.open(ALS_Path) as src1:
        alsnp = src1.read(1).astype(np.float32)  # shape: (height, width)
        ares_x, ares_y = src1.res  # (pixel width, pixel height in coordinate units)
        acrs = src1.crs

    if verbose:
        print(f"✅ Loaded S2: {s2np.shape}, ALS: {alsnp.shape}")

        print(f"{os.path.basename(S2_stack_path)}:")
        print(f"  CRS: {s2crs}")
        print(f"  Ground Sampling Distance (GSD): {s2res_x:.2f} x {s2res_y:.2f} {s2crs.linear_units}\n")

        print(f"{os.path.basename(ALS_Path)}:")
        print(f"  CRS: {acrs}")
        print(f"  Ground Sampling Distance (GSD): {ares_x:.2f} x {ares_y:.2f} {acrs.linear_units}\n")

    return s2np, alsnp 