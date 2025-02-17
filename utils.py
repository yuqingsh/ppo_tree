import rasterio
import numpy as np
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
import os
import math
from tqdm import tqdm
import shutil

POINT_PATH = "points/bounding_boxes_point.shp"
BBOX_PATH = "bounding_boxes/bounding_boxes_houchuli.shp"
GRID_SIZE = 20
OUTPUT_DIR = "grids"
TREE_PATH = "processed_trees.shp"


class ForestManager:
    def __init__(self, raster_path):
        self.trees, self.bboxes = self.read_data()
        self.trees["is_cut"] = False
        self.calculate_crown_width()
        self.update_angle_index()

        with rasterio.open(raster_path) as src:
            self.crs = src.crs
            self.bounds = src.bounds

        temp_raster_path = os.path.join("temp", "temp_raster.tif")

        if not os.path.exists("temp"):
            os.makedirs("temp")

        shutil.copy(raster_path, temp_raster_path)
        self.raster_path = temp_raster_path

        self.update_canopy_closure()

    def read_data(self):
        trees = gpd.read_file(POINT_PATH)
        bboxes = gpd.read_file(BBOX_PATH)

        return (
            trees,
            bboxes,
        )

    def update_angle_index(self):
        """
        @TODO: angle index could be nan
        """
        n_angles = 3
        n_neighbors = 4

        def _dot_product(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]

        def _angle(v1, v2):
            dot = _dot_product(v1, v2)

            v1_mag = math.sqrt(_dot_product(v1, v1))
            v2_mag = math.sqrt(_dot_product(v2, v2))
            if v1_mag < 1e-6 or v2_mag < 1e-6:
                return 0.0
            cos_theta = dot / (v1_mag * v2_mag)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)
            cross = np.cross(v1, v2)
            if cross < 0:
                theta = 2 * math.pi - theta
            return theta

        coordinates_dict = {}
        for row in self.trees.itertuples():
            if not row.is_cut:
                coordinates_dict[row.Index] = (row.geometry.x, row.geometry.y)
        coordinates = list(coordinates_dict.values())
        points = np.array(coordinates)

        if len(points) < n_neighbors:
            n_neighbors = len(points)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
        distances, indices = nbrs.kneighbors(points)

        count_list = []
        for i in range(len(points)):
            p = points[i]
            nearest_indices = indices[i]

            vectors = points[nearest_indices] - p
            angles = []
            for j in range(len(nearest_indices)):
                ang = _angle(vectors[0], vectors[j])
                angles.append((ang, nearest_indices[j]))

            angles.sort()
            selected_indices = [nearest_indices[0]] + [
                idx for ang, idx in angles[1 : n_angles + 1]
            ]

            count = 0
            num_pairs = len(selected_indices)
            for k in range(num_pairs):
                current_idx = selected_indices[k]
                next_idx = selected_indices[(k + 1) % num_pairs]

                v1 = points[current_idx] - p
                v2 = points[next_idx] - p

                ang = _angle(v1, v2)
                if ang > math.radians(279):
                    count = 10
                    break
                elif ang >= math.radians(81):
                    count += 1

            count_list.append(count)

        count_np = np.array(count_list)
        count_np = np.select(
            [
                (count_np == 3),
                (count_np == 2),
                (count_np == 1),
                (count_np == 10),
                (count_np == 4),
            ],
            [0.25, 0.5, 0.75, 1.0, 0.0],
            default=count_np,
        )

        angle_dict = {
            idx: count for idx, count in zip(coordinates_dict.keys(), count_np)
        }
        self.trees["angle_index"] = self.trees.index.map(angle_dict)

    def update_canopy_closure(self):
        with rasterio.open(self.raster_path) as src:
            valid_pixels = src.read(1) != src.nodata
            total_valid_pixels = valid_pixels.sum()
            counts = {val: (src.read(1) == val).sum() for val in range(1, 5)}
            yubidu = sum(counts[val] for val in range(1, 4)) / sum(
                counts[val] for val in range(1, 5)
            )
            self.yubidu = yubidu

    def calculate_crown_width(self):
        crown_width = []
        for row in self.bboxes.itertuples():
            bbox = row.geometry
            left, bottom, right, top = bbox.bounds
            crown_width.append((right - left) * (top - bottom))
        self.trees["guanfu"] = crown_width

    def harvest_tree(self, tree_id):
        self.trees.loc[tree_id, "is_cut"] = True
        bbox_to_harvest = self.bboxes.iloc[tree_id]["geometry"]
        with rasterio.open(self.raster_path, "r+") as src:
            window = src.window(*bbox_to_harvest.bounds)
            data = src.read(1, window=window)
            mask = (data == 1) & (data != src.nodata)
            data[mask] = 4
            src.write(data, 1, window=window)
        self.update_angle_index()
        self.update_canopy_closure()

    def get_sum_angle_index(self):
        running_total = 0
        for row in self.trees.itertuples():
            if not row.is_cut:
                running_total += abs(row.angle_index - 0.5)
        return running_total

    def get_stats(self):
        return {
            "max_chm_min": self.trees["max_chm"].min(),
            "max_chm_max": self.trees["max_chm"].max(),
            "xiongjing_min": self.trees["xiongjing"].min(),
            "xiongjing_max": self.trees["xiongjing"].max(),
            "guanfu_min": self.trees["guanfu"].min(),
            "guanfu_max": self.trees["guanfu"].max(),
        }


def main():
    forest_manager = ForestManager("classification.tif")
    print(forest_manager.trees.head())
    print(forest_manager.yubidu)
    forest_manager.harvest_tree(1)
    print(forest_manager.trees.head())
    print(forest_manager.yubidu)


if __name__ == "__main__":
    main()
