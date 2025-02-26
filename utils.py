import rasterio
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
import geopandas as gpd
import os
import math
import shutil
import time

BLOCK_X = 0
BLOCK_Y = 1
DIR_PATH = "grids/block_" + str(BLOCK_X) + "_" + str(BLOCK_Y) + "/"
POINT_PATH = DIR_PATH + "points_" + str(BLOCK_X) + "_" + str(BLOCK_Y) + ".shp"
BBOX_PATH = DIR_PATH + "boxes_" + str(BLOCK_X) + "_" + str(BLOCK_Y) + ".shp"
TIF_PATH = DIR_PATH + "raster_" + str(BLOCK_X) + "_" + str(BLOCK_Y) + ".tif"


class ForestManager:
    def __init__(self):
        raster_path = TIF_PATH
        self.trees, self.bboxes = self.read_data()

        temp_raster_path = os.path.join("temp", "temp_raster.tif")

        if not os.path.exists("temp"):
            os.makedirs("temp")

        shutil.copy(raster_path, temp_raster_path)
        self.raster_path = temp_raster_path

        self.read_tif_to_array()
        self.trees["is_cut"] = False
        self.calculate_crown_width()
        self.update_compete_index()

        with rasterio.open(raster_path) as src:
            self.crs = src.crs
            self.bounds = src.bounds
            self.resolution = src.res[0]

        self.update_canopy_closure()

    def read_data(self):
        trees = gpd.read_file(POINT_PATH)
        bboxes = gpd.read_file(BBOX_PATH)

        return (
            trees,
            bboxes,
        )

    def update_compete_index(self):
        n_neighbors = 5

        coordinates_dict = {}
        xiongjings_dict = {}
        for row in self.trees.itertuples():
            if not row.is_cut:
                coordinates_dict[row.Index] = (row.geometry.x, row.geometry.y)
                xiongjings_dict[row.Index] = row.xiongjing
        coordinates = list(coordinates_dict.values())
        points = np.array(coordinates)
        xiongjings = np.array(list(xiongjings_dict.values()))

        if len(points) < n_neighbors:
            n_neighbors = len(points)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
        distances, indices = nbrs.kneighbors(points)

        cis = []
        for i in range(len(points)):
            nearest_indices = indices[i][1:]
            nearest_distances = distances[i][1:]

            if nearest_distances.all() == 0:
                cis.append(0)
                continue

            ci = 0
            for j in range(len(nearest_indices)):
                ci += xiongjings[nearest_indices[j]] / nearest_distances[j]
            ci /= xiongjings[i]
            cis.append(ci)
        cis_dict = {key: value for key, value in zip(coordinates_dict.keys(), cis)}
        self.trees["compete_index"] = self.trees.index.map(cis_dict)

    def get_compete_index(self, tree_id):
        return self.trees.loc[tree_id, "compete_index"]

    def update_canopy_closure(self):
        counts = {val: (self.classification == val).sum() for val in range(1, 5)}
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

    def _get_box_indices(self, tree_id):
        bbox = self.bboxes.iloc[tree_id]["geometry"]
        left, bottom, right, top = bbox.bounds
        left_index = max(0, int((left - self.bounds.left) / self.resolution))
        right_index = min(
            self.classification.shape[1] - 1,
            int((right - self.bounds.left) / self.resolution),
        )
        bottom_index = max(0, int((bottom - self.bounds.bottom) / self.resolution))
        top_index = min(
            self.classification.shape[0] - 1,
            int((top - self.bounds.bottom) / self.resolution),
        )
        return left_index, right_index, bottom_index, top_index

    def harvest_tree(self, tree_id):
        self.trees.loc[tree_id, "is_cut"] = True
        bbox_indices = self._get_box_indices(tree_id)
        self.classification[
            bbox_indices[2] : bbox_indices[3], bbox_indices[0] : bbox_indices[1]
        ] = 4
        self.update_compete_index()
        self.update_canopy_closure()

    def get_stats(self):
        return {
            "max_chm_min": self.trees["max_chm"].min(),
            "max_chm_max": self.trees["max_chm"].max(),
            "xiongjing_min": self.trees["xiongjing"].min(),
            "xiongjing_max": self.trees["xiongjing"].max(),
            "guanfu_min": self.trees["guanfu"].min(),
            "guanfu_max": self.trees["guanfu"].max(),
        }

    def read_tif_to_array(self):
        with rasterio.open(self.raster_path) as src:
            self.classification = src.read(1)


def main():
    fm1 = ForestManager("classification.tif")
    fm1.get_area(0)


if __name__ == "__main__":
    main()
