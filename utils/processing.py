import os
import re
import subprocess
import glob
from collections import defaultdict
import numpy as np
import rasterio
import utils.basics as bsc
from rasterio.warp import reproject, Resampling
from types import SimpleNamespace
import json
import datetime

def compare_raster_metadata(file_A_path, file_B_path):
    """
    Compare CRS and transform between two raster files (A and B), printing their names and metadata.
    """
    with rasterio.open(file_A_path) as src_A:
        crs_A = src_A.crs
        transform_A = src_A.transform
        desc_A = src_A.descriptions
        shape_A = src_A.shape
        count_A = src_A.count

    with rasterio.open(file_B_path) as src_B:
        crs_B = src_B.crs
        transform_B = src_B.transform
        desc_B = src_B.descriptions
        shape_B = src_B.shape
        count_B = src_B.count

    print(f"\n--- Raster A: {os.path.basename(file_A_path)} ---")
    print(f"Descriptions: {desc_A}")
    print(f"Shape: {shape_A}")
    print(f"Band count: {count_A}")
    print(f"CRS: {crs_A}")
    print(f"Transform: {transform_A}")

    print(f"\n--- Raster B: {os.path.basename(file_B_path)} ---")
    print(f"Descriptions: {desc_B}")
    print(f"Shape: {shape_B}")
    print(f"Band count: {count_B}")
    print(f"CRS: {crs_B}")
    print(f"Transform: {transform_B}")

    print("\n--- Comparison ---")
    if crs_A == crs_B:
        print("✅ CRS Projections match!")
    else:
        print("❌ CRS Projections do not match!")

    if transform_A == transform_B:
        print("✅ Transformations match!")
    else:
        print("❌ Transformations do not match!")

# ALS CLEANUP AND PROCESSING FUNCTIONS
def clean_als_tif(input_path, output_path, min_value=0, max_value=75, override=False):
    """
    Cleans an ALS .tif file by setting values outside [min_value, max_value] to NaN and saves the result.

    Args:
        input_path (str): Path to the raw ALS .tif file.
        output_path (str): Path to save the cleaned .tif file.
        min_value (float): Minimum allowed value; deafults to 0.
        max_value (float): Maximum allowed value; defaults to 75.
    """
    # Extract the base filename without extension and append '_processed.tif'
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    processed_tif_path = os.path.join(output_path, f"{base_name}_P.tif")
    os.makedirs(output_path, exist_ok=True)
    # Check if the output file already exists
    if os.path.exists(processed_tif_path) and not override:
        print(f"✅  Output file already exists: {processed_tif_path}. Use override=True to overwrite.")
        return processed_tif_path

    with rasterio.open(input_path) as src:
        als_data = src.read(1).astype(np.float32)
        profile = src.profile

    als_data = np.where((als_data < min_value) | (als_data > max_value), np.nan, als_data)
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    # Save the processed data back to a new TIFF file

    with rasterio.open(processed_tif_path, 'w', **profile) as dst:
        dst.write(als_data, 1)

    print(f"✅ Processed TIFF saved to: {processed_tif_path}")
    return processed_tif_path

# S2 VRT CREATION FUNCTION
def create_s2_vrt(root_dir, vrt_output_dir, recursive = True):

    # === STEP 1: Gather all .tif files and preserve seasonality structure ===
    season_folders = ["spring", "summer", "autumn"]
    band_groups_by_season = defaultdict(lambda: defaultdict(list))

    for season in season_folders:
        season_path = os.path.join(root_dir, season)
        if not os.path.isdir(season_path):
            continue
        for root, _, files in os.walk(season_path):
            for f in files:
                if f.lower().endswith('.tif'):
                    full_path = os.path.join(root, f)
                    # === STEP 2: Group files by band suffix ===
                    pattern = re.compile(r"_SEN2L_(.+?)\.tif$", re.IGNORECASE)
                    match = pattern.search(os.path.basename(f))
                    if match:
                        band = match.group(1)
                        band_groups_by_season[season][band].append(full_path)
            if not recursive:
                break

    # Create VRTs in season subfolders inside vrt_output_dir
    for season, band_groups in band_groups_by_season.items():
        season_vrt_dir = os.path.join(vrt_output_dir, season)
        os.makedirs(season_vrt_dir, exist_ok=True)
        for band, files in band_groups.items():
            vrt_path = os.path.join(season_vrt_dir, f"{band}.vrt")
            print(f"Creating {band} VRT for {season} with {len(files)} files...")

            # Write file list
            list_path = os.path.join(season_vrt_dir, f"list_{band}.txt")
            with open(list_path, "w") as f:
                for tif in files:
                    f.write(f"{tif}\n")

            # Call gdalbuildvrt
            subprocess.run([
                "gdalbuildvrt",
                "-input_file_list", list_path,
                vrt_path
            ], check=True)

            os.remove(list_path)

            # Set band descriptions from the first tif file
            with rasterio.open(files[0]) as src:
                descriptions = src.descriptions
            with rasterio.open(vrt_path, 'r+') as vrt:
                for i, desc in enumerate(descriptions, start=1):
                    if desc:
                        vrt.set_band_description(i, desc)

        print(f"✅ Done! VRTs for {season} created.")


# S2 Crop the VRTs to ALS extent, handling season subfolders
def crop_and_stack_vrts_to_als(
    als_path, 
    vrt_folder, 
    output_folder,
    override=False
):
    """
    Crop and stack all VRT bands to the bounding box of a single ALS raster,
    handling season subfolders.

    Args:
        als_path (str): Path to the ALS raster (used for bounding box and CRS).
        vrt_folder (str): Folder containing season subfolders with VRT files.
        output_folder (str): Folder to save the cropped TIFF files (season subfolders will be created).
        override (bool): If True, overwrite existing files.
    """

    # Get bounding box and CRS from ALS raster
    with rasterio.open(als_path) as src:
        bounds = src.bounds
        crs = src.crs
    print(f"[DEBUG] ALS bounds: {bounds}")
    print(f"[DEBUG] ALS CRS: {crs}")
    print(f"[DEBUG] VRT folder: {vrt_folder}")
    print(f"[DEBUG] Output folder: {output_folder}")
    season_folders = ["spring", "summer", "autumn"]

    for season in season_folders:
        season_vrt_dir = os.path.join(vrt_folder, season)
        if not os.path.isdir(season_vrt_dir):
            continue
        season_output_dir = os.path.join(output_folder, season)
        os.makedirs(season_output_dir, exist_ok=True)
        vrt_files = glob.glob(os.path.join(season_vrt_dir, "*.vrt"))

        for vrt in vrt_files:
            band_name = os.path.splitext(os.path.basename(vrt))[0]
            output_tif = os.path.join(season_output_dir, f"{band_name}_cropped.tif")

            if os.path.exists(output_tif) and not override:
                print(f"[SKIP] {output_tif} already exists.")
                continue

            print(f"[WARP] Cropping {band_name} for {season} (site {os.path.basename(als_path)})...")

            cmd = [
                "gdalwarp",
                "-te", str(bounds.left), str(bounds.bottom), str(bounds.right), str(bounds.top),
                "-te_srs", crs.to_string(),
                "-of", "GTiff",
                "-overwrite",
                vrt,
                output_tif
            ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] gdalwarp failed for {band_name} ({season})")
                print(e.stderr.decode())

    print(f"\n✅ All seasonal VRTs processed and cropped.")

def stack_site_vrts_to_multiband(S2_cropped_folder, override=False):
    """
    Stacks all cropped VRT bands for each season into a single multiband TIFF (all channels per band).

    Args:
        S2_cropped_folder (str): Path to the folder containing season subfolders with cropped TIFFs.
        override (bool): If True, overwrite existing stacked file.
    Returns:
        dict: Mapping season -> stacked multiband TIFF path.
    """
    season_folders = ["spring", "summer", "autumn"]
    stacked_outputs = {}

    for season in season_folders:
        season_folder = os.path.join(S2_cropped_folder, season)
        if not os.path.isdir(season_folder):
            print(f"[SKIP] Season folder not found: {season_folder}")
            continue

        band_tifs = sorted(glob.glob(os.path.join(season_folder, "*_cropped.tif")))
        if not band_tifs:
            print(f"[SKIP] No cropped band TIFFs found in {season_folder}")
            continue

        stacked_output = os.path.join(season_folder, "S2_stacked.tif")

        if os.path.exists(stacked_output) and not override:
            print(f"✅ Loaded existing stacked file: {stacked_output}")
            stacked_outputs[season] = stacked_output
            continue

        arrays = []
        band_names = []
        channel_names = None
        for path in band_tifs:
            with rasterio.open(path) as src:
                arr = src.read()  # shape: (channels, h, w)
                arrays.append(arr)
                band_names.append(os.path.basename(path).split('_')[0])
                if channel_names is None:
                    channel_names = src.descriptions

        arrays = np.stack(arrays, axis=0)  # shape: (bands, channels, h, w)

        bands, channels, height, width = arrays.shape
        meta = rasterio.open(band_tifs[0]).meta.copy()
        meta.update(count=bands * channels)
        with rasterio.open(stacked_output, 'w', **meta) as dst:
            idx = 1
            for b in range(bands):
                for c in range(channels):
                    dst.write(arrays[b, c, :, :], idx)
                    desc = f"{band_names[b]}_{channel_names[c]}"
                    dst.set_band_description(idx, desc)
                    idx += 1
        print(f"✅ Stacked {bands} bands with {channels} channels each into {stacked_output}")
        stacked_outputs[season] = stacked_output

    return stacked_outputs

## --------ALS------------

# ALS RESAMPLING:
# Block-wise percentile clipping, this is the way to go.
# This function resamples ALS to S2 resolution, clipping upper values at a given percentile.
def resample_ALS_to_S2(als_path, s2_path, percentile=98):
    # Open S2 stack to get shape, transform, and CRS
    with rasterio.open(s2_path) as s2_src:
        s2_shape = (s2_src.height, s2_src.width)
        s2_transform = s2_src.transform
        s2_crs = s2_src.crs

    # Open ALS raster
    with rasterio.open(als_path) as als_src:
        als_data = als_src.read(1).astype(np.float32)
        # Calculate the x-th percentile and clip upper values
        als_data[als_data <= 0] = np.nan  # Set negative values to NaN
        # Moving window (20x20) percentile clipping
        # Block-wise processing: process ALS in 20x20 blocks, clip each block at the local percentile
        block_size = 20
        h, w = als_data.shape
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = als_data[i:i+block_size, j:j+block_size]
            if np.all(np.isnan(block)):
                continue
            perc = np.nanpercentile(block, percentile)
            als_data[i:i+block_size, j:j+block_size] = np.clip(block, None, perc)

        als_transform = als_src.transform 
        als_crs = als_src.crs

        # Prepare output array
        als_resampled = np.full(s2_shape, np.nan, dtype=np.float32)

        # Use rasterio's reproject with max aggregation to downsample
        reproject(
            source=als_data,
            destination=als_resampled,
            src_transform=als_transform,
            src_crs=als_crs,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
            resampling=Resampling.max
        )
        als_resampled[als_resampled <= 0] = np.nan  # Set negative values to NaN
    print(f"✅ Resampled ALS to S2 shape: {als_resampled.shape}, S2 shape: {s2_shape}; percentile: {percentile}th")
    return als_resampled

# write the resampled ALS to a new file
def write_resampled_als(als_data, s2_path, output_path):
    """
    Write the resampled ALS data to a new TIFF file.
    
    Args:
        als_data (np.ndarray): Resampled ALS data.
        s2_path (str): Path to the S2 stack for metadata.
        output_path (str): Path to save the resampled ALS TIFF.
    """
    with rasterio.open(s2_path) as s2_src:
        profile = s2_src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan,
            description="CHM"
        )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(als_data, 1)

    print(f"✅ Resampled ALS saved to: {output_path}")
    return output_path

## ------- AUX Layers ------- 


def transform_and_crop_to_als(input_path, als_tif_path, s2_path, OUTPUT_FOLDER):
    # Get target CRS, transform, and shape from S2
    with rasterio.open(s2_path) as s2_src:
        target_crs = s2_src.crs
        target_transform = s2_src.transform
        target_height = s2_src.height
        target_width = s2_src.width
        target_meta = s2_src.meta.copy()
        target_meta.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'interleave': 'band',
            'count': 1,
            'nodata': 0
        })

    # Get the bounds of the ALS raster (for cropping)
    with rasterio.open(als_tif_path) as als_src:
        als_bounds = als_src.bounds

    # Read and reproject/crop the raster
    with rasterio.open(input_path) as src:
        data = src.read(1)
        data_reprojected = np.zeros((target_height, target_width), dtype=np.uint8)
        reproject(
            source=data,
            destination=data_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    # Mask out everything outside the ALS bounds
    with rasterio.open(s2_path) as s2_src:
        s2_window = rasterio.windows.from_bounds(*als_bounds, transform=target_transform)
        s2_window = s2_window.round_offsets().round_lengths()
        cropped_data = data_reprojected[
            int(s2_window.row_off):int(s2_window.row_off + s2_window.height),
            int(s2_window.col_off):int(s2_window.col_off + s2_window.width)
        ]
        cropped_meta = target_meta.copy()
        cropped_meta.update({
            'height': cropped_data.shape[0],
            'width': cropped_data.shape[1],
            'transform': rasterio.windows.transform(s2_window, target_transform)
        })

    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(input_path)[: -4] + "_P.tif")
    #output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(input_path))

    with rasterio.open(output_path, 'w', **cropped_meta) as dst:
        dst.write(cropped_data, 1)

    return cropped_data, cropped_meta, output_path


def resample_and_crop_dem_to_als(dem30_path, als_ref_path, output_folder, resampling_method=Resampling.bilinear):
    """
    Resample DEM from 30m to 10m resolution and crop to ALS reference raster.
    Output filename will be similar to input, but with 'DEM10' instead of 'DEM30'.
    """
    # Generate output filename
    dem30_filename = os.path.basename(dem30_path)
    dem10_filename = dem30_filename.replace('DEM30', 'DEM10')
    output_path = os.path.join(output_folder, dem10_filename)

    # Open ALS reference to get target shape, transform, crs
    with rasterio.open(als_ref_path) as als_src:
        target_crs = als_src.crs
        target_transform = als_src.transform
        target_height = als_src.height
        target_width = als_src.width
        target_meta = als_src.meta.copy()
        target_meta.update({
            'driver': 'GTiff',
            'dtype': 'float32',
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'interleave': 'band',
            'count': 1,
            'nodata': 0
        })

    # Open DEM30 and reproject/resample to ALS grid
    with rasterio.open(dem30_path) as dem_src:
        dem_data = dem_src.read(1)
        dem10 = np.zeros((target_height, target_width), dtype=np.float32)
        reproject(
            source=dem_data,
            destination=dem10,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling_method
        )

    # Write output
    with rasterio.open(output_path, 'w', **target_meta) as dst:
        dst.write(dem10, 1)

    return dem10, target_meta, output_path


## ALL NORMALIZATION FUNCTIONS

def normalize_and_save_s2_geotiff(input_path, namespace, attr_name, min_percentile=0.1, max_percentile=99.9,override=False):
    """
    Reads a GeoTIFF, normalizes each band to [0,1] based on percentiles, saves with '_norm' suffix,
    and updates the namespace with the new filename.

    Parameters:
    - input_path: str, path to the original GeoTIFF
    - namespace: SimpleNamespace, where the filename should be updated
    - attr_name: str, attribute name in the namespace to update
    - min_percentile: float, lower percentile for normalization
    - max_percentile: float, upper percentile for normalization
    """

    # The original S2 data is int16, which can represent integer values from -32768 to 32767.
    # When normalizing to [0, 1], you convert the data to floating point.
    # float16 has less precision than float32 (about 3 decimal digits vs 7), but for normalized reflectance data,
    # float16 is usually sufficient for most ML and remote sensing workflows.
    # You will lose some precision (not information about the original int16 values, but about the normalized value),
    # but for visualization and most ML tasks, this is acceptable.
    # If you need to preserve more precision, use float32 or scale to uint16 (0-65535).
    # For most use cases, float16 is a good compromise between file size and precision.

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_norm{ext}"

    if os.path.exists(output_path) and not override:
        # Update namespace and print message
        if not hasattr(namespace, 'S2_norm'):
            namespace.S2_norm = SimpleNamespace()
        setattr(namespace.S2_norm, attr_name, output_path)
        print(f"✅ Normalized file already exists: {output_path}. Namespace updated.")
        return

    with rasterio.open(input_path) as src:
        s2_data = src.read().astype(np.float32)
        meta = src.meta.copy()
        band_count = s2_data.shape[0]
        descriptions = src.descriptions if hasattr(src, 'descriptions') else None

    # Set negative values to nan
    #s2_data = s2_data.astype(np.float32)
    s2_data[s2_data < 0] = np.nan

    normalized_s2 = np.zeros_like(s2_data, dtype=np.float32)
    for i in range(band_count):
        band_data = s2_data[i]
        band_min = np.nanpercentile(band_data, min_percentile)
        band_max = np.nanpercentile(band_data, max_percentile)
        band_data = np.clip(band_data, band_min, band_max)
        if band_max - band_min > 0:
            normalized_s2[i] = (band_data - band_min) / (band_max - band_min)
        else:
            normalized_s2[i] = np.nan

    # Prepare output path
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_norm{ext}"

    # Save normalized data
    meta.update({'dtype': 'float32', 'compress': 'lzw'})
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(normalized_s2)
        if descriptions:
            dst.descriptions = descriptions

    # Update namespace: create S2_norm if not present, then set attribute (e.g., spring)
    if not hasattr(namespace, 'S2_norm'):
        namespace.S2_norm = SimpleNamespace()
    setattr(namespace.S2_norm, attr_name, output_path)
    #print(f"✅ Normalized S2 data saved to {output_path} and namespace.S2_norm.{attr_name} updated.")

# now lets do ALS normalization 

def compute_chm_norm_params(SITE1, SITE2, SITE3):

    fmask1 = bsc.read_tif_as_array(SITE1.FMASK, verbose=False)
    fmask2 = bsc.read_tif_as_array(SITE2.FMASK, verbose=False)
    fmask3 = bsc.read_tif_as_array(SITE3.FMASK, verbose=False)
    als1 = bsc.read_tif_as_array(SITE1.CHM, verbose=False)
    als2 = bsc.read_tif_as_array(SITE2.CHM, verbose=False)
    als3 = bsc.read_tif_as_array(SITE3.CHM, verbose=False)

    # Ensure fmasks are integer type for comparison
    fmask1_bin = (fmask1.astype(int) == 1)
    fmask2_bin = (fmask2.astype(int) == 1)
    fmask3_bin = (fmask3.astype(int) == 1)

    # Mask out NaNs in ALS arrays and select only forest pixels
    als1_valid = als1[~np.isnan(als1) & fmask1_bin]
    als2_valid = als2[~np.isnan(als2) & fmask2_bin]
    als3_valid = als3[~np.isnan(als3) & fmask3_bin]

    # Calculate mu and std for each combination of ALS valid arrays
    combos = {
        "001": [als1_valid],
        "010": [als2_valid],
        "100": [als3_valid],
        "011": [als1_valid, als2_valid],
        "101": [als1_valid, als3_valid],
        "110": [als2_valid, als3_valid],
        "111": [als1_valid, als2_valid, als3_valid]
    }

    # Use attribute names that are valid for SimpleNamespace (prefix with underscore)
    norm_params = {}
    for key, arrs in combos.items():
        attr_key = f"_{key}"
        lengths = [len(a) for a in arrs]
        total = sum(lengths)
        weights = [l / total for l in lengths]
        concat = np.concatenate(arrs)
        mu = np.average([np.mean(a) for a in arrs], weights=weights)
        std = np.average([np.std(a) for a in arrs], weights=weights)
        norm_params[attr_key] = {"mu": float(mu), "std": float(std), "n": int(total)}

    # Option 1: Store as JSON file
    # with open("chm_norm_params.json", "w") as f:
    #     json.dump(norm_params, f, indent=2)

    # Option 2: Store in a SimpleNamespace
    NORMPARAMS = SimpleNamespace()
    NORMPARAMS.chm = SimpleNamespace(**norm_params)
    NORMPARAMS.info = SimpleNamespace(
        description="CHM normalization parameters",
        encoding="001: FirstSite only, 010: Second Site, 100: Third Site",
        version="1.0"
    )
    return NORMPARAMS

def normalize_chm(SITE1,SITE2,SITE3,NORMPARAMS,jointnorm = True):

    # Read ALS data
    als1 = bsc.read_tif_as_array(SITE1.CHM, verbose=False)
    als2 = bsc.read_tif_as_array(SITE2.CHM, verbose=False)
    als3 = bsc.read_tif_as_array(SITE3.CHM, verbose=False)
    if jointnorm:
        # Combine all valid ALS data for joint normalization
        als1_norm = (als1 - NORMPARAMS.chm._111["mu"]) / NORMPARAMS.chm._111["std"]
        als2_norm = (als2 - NORMPARAMS.chm._111["mu"]) / NORMPARAMS.chm._111["std"]
        als3_norm = (als3 - NORMPARAMS.chm._111["mu"]) / NORMPARAMS.chm._111["std"]

    else:
        # Apply normalization individually
        als1_norm = (als1 - NORMPARAMS.chm._001["mu"]) / NORMPARAMS.chm._001["std"]
        als2_norm = (als2 - NORMPARAMS.chm._010["mu"]) / NORMPARAMS.chm._010["std"]
        als3_norm = (als3 - NORMPARAMS.chm._100["mu"]) / NORMPARAMS.chm._100["std"]
        als2_norm = (als2 - NORMPARAMS.chm["010"].mu) / NORMPARAMS.chm["010"].std
        als3_norm = (als3 - NORMPARAMS.chm["100"].mu) / NORMPARAMS.chm["100"].std

    # Save normalized data using original metadata
    for als_norm, site in zip([als1_norm, als2_norm, als3_norm], [SITE1, SITE2, SITE3]):
        with rasterio.open(site.CHM) as src:
            meta = src.meta.copy()
            meta.update(dtype='float32', nodata=np.nan)
            out_path = site.CHM.replace(".tif", "_norm111.tif")
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(als_norm.astype(np.float32), 1)
                dst.descriptions = [f"nCHM_global"]
            site.CHM_norm = out_path
            site.CHM_norm_params = {
                "mu": NORMPARAMS.chm._111["mu"],
                "std": NORMPARAMS.chm._111["std"]
            }
