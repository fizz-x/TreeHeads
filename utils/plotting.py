import numpy as np
import matplotlib.pyplot as plt
import rasterio
from utils.basics import load_rasters

def normalize_S2(s2, band_idxs=(10, 3, 0)):
    """
    Normalize Sentinel-2 data for visualization.

    Parameters:
    - s2: numpy array of S2 data (bands, height, width)
    - band_idxs: tuple of band indices for RGB visualization (default: (10, 3, 0))

    Returns:
    - rgb: numpy array of normalized RGB image (height, width, bands)
    """
    # Ensure S2 data is non-negative
    s2[s2 < 0] = np.nan
    # Clip to the 50th percentile to avoid extreme values
    s2 = np.clip(s2, 0, np.nanpercentile(s2, 50))
    # Normalize the data
    s2 = (s2 - np.nanmin(s2)) / (np.nanmax(s2) - np.nanmin(s2))
    
    # Create RGB image from specified bands
    rgb = s2[band_idxs, :, :].transpose(1, 2, 0)  # shape: (height, width, bands)
    
    return rgb

def plot_full_image(s2, als, band_idxs=(10, 3, 0)):
    """
    Plot the full image of S2 data (RGB) and ALS.

    Parameters:
    - s2: numpy array of S2 data (bands, height, width)
    - als_mean: numpy array of ALS data (height, width)
    - band_idxs: tuple of band indices for RGB visualization (default: (10, 3, 0))
    """

    rgb = normalize_S2(s2, band_idxs)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Plot S2 RGB
    axs[0].imshow(rgb)
    axs[0].set_title("S2 RGB")
    axs[0].axis("off")

    # Plot ALS mean
    im = axs[1].imshow(als, cmap='viridis')
    axs[1].set_title("Canopy Height (ALS Resampled)")
    # axs[1].set_xlabel("Meters") # Optional: add x-axis label
    # axs[1].set_ylabel("Meters") # Optional: add y-axis label
    # axs[1].set_aspect('equal')  # Ensure equal aspect ratio
    axs[1].axis("off")
    cbar = fig.colorbar(im, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Height (m)")

    plt.tight_layout()
    plt.show()


def plot_overlay(s2, als, band_idxs=(10, 3, 0), alpha=0.5):
    """
    Plot an overlay of the RGB image and ALS data.

    Parameters:
    - s2: numpy array of S2 data (bands, height, width)
    - als: numpy array of ALS data (height, width)
    - band_idxs: tuple of band indices for RGB visualization (default: (10, 3, 0))
    - alpha: transparency level for the ALS overlay (default: 0.5)
    """

    rgb = normalize_S2(s2, band_idxs)

    als[als == 0] = np.nan

    # Plot RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb, interpolation='none')
    plt.imshow(als, cmap='Reds', alpha=alpha, interpolation='none')  # Overlay ALS data
    plt.title("RGB Image with CHM Overlay")
    plt.axis("off")
    plt.colorbar(label="Canopy Height (m)", fraction=0.02,pad=0.04,)
    plt.tight_layout
    plt.show()

def plot_ALS_histogram(als_path):
    """
    Plot histogram of ALS data.
    """
    # plot histogram of ALS data
    with rasterio.open(als_path) as src:
        als_data = src.read(1).astype(np.float32)

    # === Mask out nodata and NaNs (e.g., 0 or -9999 can be nodata in ALS)
    als_data = als_data[~np.isnan(als_data)]
    # als_data = als_data[als_data <=120]  # optional: remove zeroes if they're nodata
    #als_data = als_data[als_data >= 0]  # optional: filter a known nodata

    # === Plot histogram ===
    percentiles = [0,5,10,90, 95, 97.5, 98, 99, 100]
    #cbar = 
    values = als_data
    perc_values = np.percentile(values, percentiles)

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=256, color='skyblue', edgecolor='black', alpha=0.7)
    for p, v in zip(percentiles, perc_values):
        plt.axvline(v, color='r', linestyle='--', label=f'{p}th: {v:.1f} m')
    plt.title("ALS Pixel Value Histogram with Percentiles")
    plt.xlabel("Canopy Height (m)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

