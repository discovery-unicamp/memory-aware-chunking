import itertools
import os

from common import builders

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/inputs")
INITIAL_SIZE = int(os.getenv("INITIAL_SIZE", "100"))
FINAL_SIZE = int(os.getenv("FINAL_SIZE", "600"))
STEP_SIZE = int(os.getenv("STEP_SIZE", "100"))


def main():
    print("Generating data...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  INITIAL_SIZE: {INITIAL_SIZE}")
    print(f"  FINAL_SIZE: {FINAL_SIZE}")
    print(f"  STEP_SIZE: {STEP_SIZE}")
    print()

    dataset_shapes = list(range(INITIAL_SIZE, FINAL_SIZE + 1, STEP_SIZE))
    dataset_combinations = list(itertools.product(dataset_shapes, repeat=3))

    print(f"Generated {len(dataset_combinations)} dataset combinations")

    for inlines, xlines, samples in dataset_combinations:
        print(
            f"Generating dataset with inlines={inlines}, xlines={xlines}, samples={samples}"
        )
        builders.build_seismic_data(
            inlines=inlines,
            xlines=xlines,
            samples=samples,
            output_dir=OUTPUT_DIR,
        )

    print("Finished generating synthetic seismic datasets")


if __name__ == "__main__":
    main()
