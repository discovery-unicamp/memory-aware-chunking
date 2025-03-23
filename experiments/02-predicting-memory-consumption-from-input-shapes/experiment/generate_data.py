"""
Script for generating synthetic seismic data. Reads environment variables to
determine the range of inlines, xlines, and samples for dataset generation,
and uses `builders.build_seismic_data` to produce the files.

Environment Variables (with defaults):
  - OUTPUT_DIR: The output directory to place generated data (default: "./out/inputs")
  - INITIAL_SIZE: The starting size for inlines/xlines/samples (default: 100)
  - FINAL_SIZE: The final size for inlines/xlines/samples (default: 600)
  - STEP_SIZE: The increment used when generating size combinations (default: 100)
"""

import itertools
import os

from common import builders

# ------------------------------------------------------------------------------
# Global Configuration
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/inputs")
INITIAL_SIZE = int(os.getenv("INITIAL_SIZE", "100"))
FINAL_SIZE = int(os.getenv("FINAL_SIZE", "600"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "100"))


def main():
    """
    Main function that:
      1. Prints the runtime configuration (env vars).
      2. Builds a list of dataset combinations (inlines, xlines, samples),
         spanning from INITIAL_SIZE to FINAL_SIZE in increments of STEP_SIZE.
      3. Uses `builders.build_seismic_data()` for each combination
         to generate the synthetic seismic datasets.
    """
    print("Generating data...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  INITIAL_SIZE: {INITIAL_SIZE}")
    print(f"  FINAL_SIZE: {FINAL_SIZE}")
    print(f"  STEP_SIZE: {STEP_SIZE}")
    print()

    # Generate size combinations (e.g., (100,100,100), (100,100,200), ...)
    dataset_shapes = range(INITIAL_SIZE, FINAL_SIZE + 1, STEP_SIZE)
    dataset_combinations = list(itertools.product(dataset_shapes, repeat=3))

    print(f"Generated {len(dataset_combinations)} dataset combinations.")

    # For each combination, call builders.build_seismic_data
    for inlines, xlines, samples in dataset_combinations:
        print(
            f"Generating dataset (inlines={inlines}, xlines={xlines}, samples={samples})"
        )
        builders.build_seismic_data(
            inlines=inlines,
            xlines=xlines,
            samples=samples,
            output_dir=OUTPUT_DIR,
        )

    print("Finished generating synthetic seismic datasets.")


if __name__ == "__main__":
    main()
