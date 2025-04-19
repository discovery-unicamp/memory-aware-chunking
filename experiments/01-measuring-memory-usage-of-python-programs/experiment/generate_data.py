"""
Script for generating synthetic seismic data. Reads environment variables to
determine the range of inlines, xlines, and samples for dataset generation,
and uses `builders.build_seismic_data` to produce the files.

Environment Variables (with defaults):
  - OUTPUT_DIR: The output directory to place generated data (default: "./out/inputs")
  - DATASET_INLINES: The number of inlines in the dataset (default: 100)
  - DATASET_XLINES: The number of xlines in the dataset (default: 100)
  - DATASET_SAMPLES: The number of samples in the dataset (default: 100)
"""

import os

from common import builders

# ------------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/inputs")
DATASET_INLINES = int(os.getenv("DATASET_INLINES", 100))
DATASET_XLINES = int(os.getenv("DATASET_XLINES", 100))
DATASET_SAMPLES = int(os.getenv("DATASET_SAMPLES", 100))


def main():
    """
    Main function that:
      1. Prints the runtime configuration (env vars).
      2. Uses `builders.build_seismic_data()` to generate the seismic data
    """
    print("Generating data...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  DATASET_INLINES: {DATASET_INLINES}")
    print(f"  DATASET_XLINES: {DATASET_XLINES}")
    print(f"  DATASET_SAMPLES: {DATASET_SAMPLES}")
    print()

    print(
        f"Generating dataset (inlines={DATASET_INLINES}, xlines={DATASET_XLINES}, samples={DATASET_SAMPLES})"
    )
    builders.build_seismic_data(
        inlines=DATASET_INLINES,
        xlines=DATASET_XLINES,
        samples=DATASET_SAMPLES,
        output_dir=OUTPUT_DIR,
    )

    print("Finished generating synthetic seismic dataset.")


if __name__ == "__main__":
    main()
