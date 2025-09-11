import os
import numpy as np
import rasterio

# load the stacked S2 and merged ALS rasters
def load_rasters(S2_stack_path, ALS_Path, verbose=True, s2_channel="Q50"):
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
        band_names = s2_src.descriptions  # List of band names
        if s2_channel:
            selected_band_indices = [i + 1 for i, name in enumerate(band_names) if s2_channel in (name or "")]
            if not selected_band_indices:
                raise ValueError(f"No bands found containing '{s2_channel}' in their name.")
            s2np = s2_src.read(selected_band_indices).astype(np.float32)  # shape: (selected_bands, height, width)
            band_names = tuple(band_names[i - 1] for i in selected_band_indices)
        else:
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
        print(f"  Ground Sampling Distance (GSD): {s2res_x:.2f} x {s2res_y:.2f} {s2crs.linear_units}")
        print(f"  Band names: {band_names}\n")

        print(f"{os.path.basename(ALS_Path)}:")
        print(f"  CRS: {acrs}")
        print(f"  Ground Sampling Distance (GSD): {ares_x:.2f} x {ares_y:.2f} {acrs.linear_units}\n")

    return s2np, alsnp, band_names

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
        arr = arr.reshape((bands, channels, h, w))
        desc = src.descriptions  # List of band names
    return arr, desc

def read_tif_as_array(tif_path, verbose=True):
    """
    Reads a single-band TIFF and returns a numpy array.

    Args:
        tif_path (str): Path to the TIFF file.

    Returns:
        np.ndarray: Array of shape (h, w).
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1)  # Read the first band
        res_x, res_y = src.res  # (pixel width, pixel height in coordinate units)
        crs = src.crs
        dsc = src.descriptions  # List of band names

        if verbose:
            print(f"✅ Loaded {os.path.basename(tif_path)}: shape={arr.shape}, CRS={crs}, GSD={res_x:.2f} x {res_y:.2f} {crs.linear_units}")
            print(f"Band names: {dsc}")

    return arr.astype(np.float32)  # Ensure float32 type for consistency