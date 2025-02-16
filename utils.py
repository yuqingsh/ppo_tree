import rasterio
import numpy as np
from shapely.geometry import box, Polygon
from rasterio.mask import mask
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
        # self.update_angle_index()

        with rasterio.open(raster_path) as src:
            self.crs = src.crs

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
        def distance(p1, p2):
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        def dot_product(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]

        def angle(v1, v2):
            dot = dot_product(v1, v2)
            v1_mag = math.sqrt(dot_product(v1, v1))
            v2_mag = math.sqrt(dot_product(v2, v2))
            if v1_mag < 1e-6 or v2_mag < 1e-6:
                return 0
            cos_theta = dot / (v1_mag * v2_mag)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = math.acos(cos_theta)
            cross = np.cross(v1, v2)
            if cross < 0:
                theta = 2 * math.pi - theta
            return theta

        def count_angle(points_dict):
            points = list(points_dict.values())
            count_list = []
            for i, p in tqdm(enumerate(points), total=len(points)):
                distances = [
                    (distance(p, other), j) for j, other in enumerate(points) if j != i
                ]
                distances.sort()
                nearest_point_index = [i for dis, i in distances[:4]]
                angles = []
                for j in range(len(nearest_point_index) - 1):
                    v1 = (
                        points[nearest_point_index[0]][0] - p[0],
                        points[nearest_point_index[0]][1] - p[1],
                    )
                    v2 = (
                        points[nearest_point_index[j + 1]][0] - p[0],
                        points[nearest_point_index[j + 1]][1] - p[1],
                    )
                    angles.append((angle(v1, v2), nearest_point_index[j + 1]))
                angles.sort()
                angles_index = [i for ang, i in angles[:3]]
                angles_index.insert(0, nearest_point_index[0])
                count = 0
                for k in range(len(angles_index)):
                    if k <= len(nearest_point_index) - 2:
                        v1 = (
                            points[angles_index[k]][0] - p[0],
                            points[angles_index[k]][1] - p[1],
                        )
                        v2 = (
                            points[angles_index[k + 1]][0] - p[0],
                            points[angles_index[k + 1]][1] - p[1],
                        )
                    elif k == len(nearest_point_index) - 1:
                        v1 = (
                            points[angles_index[k]][0] - p[0],
                            points[angles_index[k]][1] - p[1],
                        )
                        v2 = (
                            points[angles_index[0]][0] - p[0],
                            points[angles_index[0]][1] - p[1],
                        )
                    if angle(v1, v2) > math.radians(279):
                        count = 10
                        break
                    elif angle(v1, v2) >= math.radians(81):
                        count += 1
                count_list.append(count)
            count_np = np.array(count_list)
            count_np = np.where(count_np == 3, 0.25, count_np)
            count_np = np.where(count_np == 2, 0.5, count_np)
            count_np = np.where(count_np == 1, 0.75, count_np)
            count_np = np.where(count_np == 10, 1, count_np)
            count_np = np.where(count_np == 4, 0, count_np)
            points_dict = {
                key: value for key, value in zip(points_dict.keys(), count_np)
            }
            return points_dict

        coordinates_dict = {}
        for row in self.trees.itertuples():
            if not row.is_cut:
                coordinates_dict[row.Index] = (row.geometry.x, row.geometry.y)
        points_dict = count_angle(coordinates_dict)
        self.trees["angle_index"] = self.trees.index.map(points_dict)

    def update_canopy_closure(self):
        with rasterio.open(self.raster_path) as src:
            valid_pixels = src.read(1) != src.nodata
            total_valid_pixels = valid_pixels.sum()
            counts = {val: (src.read(1) == val).sum() for val in range(1, 5)}
            yubidu = sum(counts[val] for val in range(1, 4)) / total_valid_pixels
            self.yubidu = yubidu

    def harvest_tree(self, tree_id):
        self.trees.loc[tree_id, "is_cut"] = True
        bbox_to_harvest = self.bboxes.iloc[tree_id]["geometry"]
        with rasterio.open(self.raster_path, "r+") as src:
            window = src.window(*bbox_to_harvest.bounds)
            data = src.read(1, window=window)
            mask = (data == 1) & (data != src.nodata)
            data[mask] = 4
            src.write(data, 1, window=window)
        # self.update_angle_index()
        self.update_canopy_closure()


def main():
    forest_manager = ForestManager("classification.tif")
    print(forest_manager.trees.head())
    print(forest_manager.yubidu)
    forest_manager.harvest_tree(1)
    print(forest_manager.trees.head())
    print(forest_manager.yubidu)


if __name__ == "__main__":
    main()
