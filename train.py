import os
from utils import (
    create_grids,
    assign_trees_to_grids,
    split_df_by_col,
    read_grid_file,
    OUTPUT_DIR,
)


def prepare_data():
    tif_file = "mask2.tif"
    _, num_grid = create_grids(tif_file, OUTPUT_DIR)
    joined = assign_trees_to_grids(os.path.join(OUTPUT_DIR, "grids.shp"))
    grid_id_to_trees = split_df_by_col(joined, "grid_id")
    total = 0
    grids_df = read_grid_file()
    print(grids_df)


def main():
    prepare_data()


if __name__ == "__main__":
    main()
