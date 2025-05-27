# TreeHeads
Tree-Height-Estimation-Across-Deep-Systems

## Architecture - Big Picture
<p align="center" style="background: white;">
    <img src="images/03_ARCHITECTURE_SKETCH.png" alt="Arch" style="background: white; padding: 10px;">
</p>


## Folder Structure

The following folder structure is needed for organizing the project:

```
TreeHeads/
├── data/
│   ├── 01_input/          # Original, unprocessed data (not tracked by git)
    │   ├── S2_Summer_Median/          # S2 Data
    │   ├── ALS_GT01_2024.tif          # ALS 1 (Ebrach)
    │   ├── ALS_GT02_2024.tif          # ALS 2 (Waldbrunn)
│   ├── 02_processed/     # Cleaned and processed data (not tracked by git)
│   └── 03_training/      # stacked training np images (not tracked by git)
├── images/            # Project images and visualizations
├── processing/         # Jupyter notebooks for processing pipeline
├── training/         # Jupyter notebooks for training pipeline
├── utils/               # frequently used functions 
├── models/            # Saved models and checkpoints (not tracked by git)
└── README.md
```

**Note:**  
The `data/` and `models/` directories are intentionally excluded from version control (e.g., via `.gitignore`) to avoid pushing large or sensitive datasets and model files to the repository. Only code, documentation, and lightweight assets (such as images) should be committed to git.


## Sentinel 2 images with LiDAR Reference Data CHM Overlay
S2 with ALS overlay: 

![CHM_Site2](images/CHM_ALS2.png)