import rasterio
import numpy as np
from shapely.geometry import box, Polygon
import geopandas as gpd
import os

POINT_PATH = "points/bounding_boxes_point.shp"
BBOX_PATH = "bounding_boxes/bounding_boxes_houchuli.shp"
GRID_SIZE = 20
OUTPUT_DIR = "grids"
TREE_PATH = "processed_trees.shp"


def read_data():
    trees = gpd.read_file(POINT_PATH)
    bboxes = gpd.read_file(BBOX_PATH)
    bboxes = bboxes.drop(columns=["geometry"])

    trees = trees.merge(bboxes, how="inner", left_index=True, right_index=True)
    trees["crown"] = trees.apply(
        lambda row: Polygon(
            [
                (row["x1"], row["y2"]),
                (row["x2"], row["y2"]),
                (row["x2"], row["y1"]),
                (row["x1"], row["y1"]),
            ]
        ).area,
        axis=1,
    )

    for i in range(trees.shape[0]):
        assert trees.iloc[i].geometry.x >= trees.iloc[i].x1
        assert trees.iloc[i].geometry.x <= trees.iloc[i].x2
        assert trees.iloc[i].geometry.y >= trees.iloc[i].y2
        assert trees.iloc[i].geometry.y <= trees.iloc[i].y1
        assert trees.iloc[i].crown >= 0

    select_cols = [
        "max_chm",
        "geometry",
        "xiongjing",
        "crown",
    ]
    return trees[select_cols].rename(
        columns={"max_chm": "height", "xiongjing": "diameter"}
    )


def create_grids(tif_path, output_dir):
    with rasterio.open(tif_path) as src:
        left, bottom = src.bounds.left, src.bounds.bottom
        right, top = src.bounds.right, src.bounds.top
        crs = src.crs

        x_coords = np.arange(left, right, GRID_SIZE)
        y_coords = np.arange(bottom, top, GRID_SIZE)

        grids = []
        for x in x_coords:
            for y in y_coords:
                grid_box = box(x, y, x + GRID_SIZE, y + GRID_SIZE)
                grids.append(grid_box)

        gdf = gpd.GeoDataFrame({"geometry": grids}, crs=crs)
        gdf["grid_id"] = [f"grid_{i}" for i in range(len(gdf))]

        os.makedirs(output_dir, exist_ok=True)
        grid_path = os.path.join(output_dir, "grids.shp")
        gdf.to_file(grid_path)
        return grid_path, len(gdf)


def assign_trees_to_grids(grid_path):
    trees = read_data()
    grids = gpd.read_file(grid_path)
    joined = gpd.sjoin(trees, grids, how="inner", predicate="within")
    joined = joined.drop(columns=["index_right"])

    os.makedirs(os.path.join(OUTPUT_DIR, "trees"), exist_ok=True)

    for grid_id in joined["grid_id"].unique():
        grid_trees = joined[joined["grid_id"] == grid_id]
        output_path = os.path.join(OUTPUT_DIR, "trees", f"{grid_id}.shp")
        grid_trees.to_file(output_path)

    return joined


def read_grid_file():
    return gpd.read_file(os.path.join(OUTPUT_DIR, "grids.shp"))


def split_df_by_col(df, col_name):
    unique_vals = df[col_name].unique()
    return {val: df[df[col_name] == val].copy() for val in unique_vals}


def main():
    tif_file = "mask2.tif"
    create_grids(tif_file, OUTPUT_DIR)
    assign_trees_to_grids(os.path.join(OUTPUT_DIR, "grids.shp"))


if __name__ == "__main__":
    main()
