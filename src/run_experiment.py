from src.config_loader import get_config, load_yaml
from src.data_loader import build_patched_dataset

def main():
    # Load configurations
    sites, cfg = get_config("baseline")

    # Build dataset from all sites
    X, Y = build_patched_dataset(cfg, sites, patch_size=32)

    print("Total patches:", len(X))
    print("One patch input shape:", X[0].shape)
    print("One patch outputs:", Y[0].shape)

if __name__ == "__main__":
    main()