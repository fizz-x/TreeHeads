import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import rasterio
from utils.basics import load_rasters

def normalize_S2(s2, band_idxs=(10, 3, 0)):
    """
    Normalize Sentinel-2 data for visualization; this was a first draft for quick visualization.
    Better to use normalized S2 data from the preprocessing pipeline.

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

def plot_full_image(s2, als, band_idxs=(10, 3, 0),norm_rgb=False):
    """
    Plot the full image of S2 data (RGB) and ALS.

    Parameters:
    - s2: numpy array of S2 data (bands, height, width)
    - als_mean: numpy array of ALS data (height, width)
    - band_idxs: tuple of band indices for RGB visualization (default: (10, 3, 0))
    """
    if norm_rgb:
        rgb = normalize_S2(s2, band_idxs)
    else:
        rgb = s2[band_idxs, :, :].transpose(1, 2, 0)

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


def plot_overlay(s2, als, band_idxs=(10, 3, 0), alpha=0.5,norm_rgb=False, type = "CHM"):
    """
    Plot an overlay of the RGB image and ALS data.

    Parameters:
    - s2: numpy array of S2 data (bands, height, width)
    - als: numpy array of ALS data (height, width)
    - band_idxs: tuple of band indices for RGB visualization (default: (10, 3, 0))
    - alpha: transparency level for the ALS overlay (default: 0.5)
    - norm_rgb: boolean to normalize RGB image (default: False)
    - type: type of data to overlay, e.g., "CHM" for ALS CHM (default: "CHM") or FMASK for binary forest mask 
    """
    if norm_rgb:
        rgb = normalize_S2(s2, band_idxs)
    else:
        rgb = s2[band_idxs, :, :].transpose(1, 2, 0)

    als[als == 0] = np.nan

    # Plot RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb, interpolation='none')
    plt.imshow(als, cmap='Reds', alpha=alpha, interpolation='none')  # Overlay ALS data
    plt.axis("off")
    if type == "CHM":
        plt.title("RGB Image with CHM Overlay")
        plt.colorbar(label="Canopy Height (m)", fraction=0.02,pad=0.04,)
    elif type == "FMASK":
        plt.title("RGB Image with Forest Mask Overlay")
        plt.colorbar(label="Forest Mask (1 = Forest)", fraction=0.02, pad=0.04) 
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

def plot_two_density_and_percentiles(tif_path1, tif_path2, percentile_values=[0, 5, 10, 50, 75, 90, 95, 98, 99, 100]):
    """
    Plot density distributions of pixel values in two TIFF files and print specified percentiles for both.
    
    Args:
        tif_path1 (str): Path to the first TIFF file.
        tif_path2 (str): Path to the second TIFF file.
        percentile_values (list): List of percentiles to calculate and display.
    """
    def load_data(path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
        return data[~np.isnan(data)]
    
    data1 = load_data(tif_path1)
    data2 = load_data(tif_path2)
    
    perc1 = np.percentile(data1, percentile_values)
    perc2 = np.percentile(data2, percentile_values)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=128, density=True, alpha=0.6, color='r', label=os.path.basename(tif_path1))
    plt.hist(data2, bins=128, density=True, alpha=0.6, color='g', label=os.path.basename(tif_path2))
    
    plt.title("Density Plot Comparison before and after resampling")
    plt.xlabel("Canopy Height (m)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()
    

    # Print percentiles as a table
    table = pd.DataFrame({
        os.path.basename(tif_path1): np.round(perc1, 2),
        os.path.basename(tif_path2): np.round(perc2, 2),
        "delta": np.round(perc2 - perc1, 2)
    }, index=pd.Index([f"{p}th" for p in percentile_values], name="Percentile"))
    print("\nPercentiles Table:")
    #pd.set_option('display.float_format', '{:.2f}'.format)  # Set float format for better readability
    print(table)

def plot_s2_histograms_and_percentiles(s2_data, num_bands=13,band_names=None):
    """
    Plot histograms for each band in the S2 data and print nanmax, nanmin, nanmean, nanmedian per band.

    Parameters:
    - s2_data: numpy array of S2 data (bands, height, width)
    - num_bands: number of bands in the S2 data (default: 13)
    """
    plt.figure(figsize=(15, 10))
    #plt.figure()

    percentiles = [0, 0.1, 25, 50, 75, 95, 99.9, 100]
    table = []
    
    for i in range(num_bands):

        band_data = s2_data[i].flatten()
        band_data_nonan = band_data[~np.isnan(band_data)]
        band_data = s2_data[i].flatten()
        band_data_nonan = band_data[~np.isnan(band_data)]
        pct_values = np.percentile(band_data_nonan, percentiles)
        table.append(pct_values)
        # plots 
        plt.subplot(4, 4, i + 1)
        plt.hist(band_data_nonan, bins=100, color='blue', alpha=0.7)
        plt.title(f'B{i + 1}_{band_names[i]}' if band_names else f'na')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True)

    df = pd.DataFrame(
        np.array(table).T,
        index=[f"P{p}" for p in percentiles],
        columns=[f"B{i+1}_{band_names[i]}" if band_names else f'na' for i in range(num_bands)]
    )
    # append one row with nan ratio per band
    nan_counts = [np.sum(np.isnan(s2_data[i])) for i in range(num_bands)]
    total_counts = [s2_data[i].size for i in range(num_bands)]
    nan_ratios = [nan_counts[i] / total_counts[i] if total_counts[i] > 0 else np.nan for i in range(num_bands)]
    df.loc['NaN Ratio'] = nan_ratios
    # append one row with band names

    #print(df.round(2))
    display(df.round(2))
    plt.suptitle('S2 Band Histograms', fontsize=16)
    plt.tight_layout()
    plt.show()