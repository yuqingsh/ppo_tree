import time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import math
import rasterio
from shapely.geometry import Point, Polygon
import os
import logging
import shutil

# 设置日志记录的配置
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ForestManagement:
    def __init__(
        self,
        grid_path,
        points_path,
        boxes_path,
        raster_path,
        resolution=0.0238,
        crs_epsg=None,
    ):
        if crs_epsg:
            self.crs_epsg = crs_epsg
        else:
            with rasterio.open(raster_path) as src:
                print(src.meta)
                crs = src.crs
                self.crs_epsg = crs
        temp_raster_path = os.path.join("temp", "temp_raster.tif")
        if not os.path.exists(temp_raster_path):
            shutil.copy(raster_path, temp_raster_path)

        self.gdf_grid = gpd.read_file(grid_path, crs_epsg=crs_epsg)
        self.gdf_points, self.gdf_boxes = self.point_2_shp(points_path, boxes_path)
        self.raster_path = temp_raster_path
        self.resolution = resolution  # 空间分辨率

    def point_2_shp(self, points_path, boxes_path):
        crs_epsg = self.crs_epsg
        with open(points_path, "r") as filetxt:
            data_point = []
            data_box = []
            ID = []
            arealist = []
            for i, line in enumerate(filetxt):
                x, y = map(float, line.strip().split(","))  # 解析每行的坐标
                point = Point(x, y)
                data_point.append(point)
                ID.append(i)
        with open(boxes_path, "r") as filetxt:
            for i, line in enumerate(filetxt):
                x1, y1, x2, y2, _ = map(float, line.strip().split(","))
                poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                area = abs(y2 - y1) * abs(x2 - x1)
                data_box.append(poly)
                arealist.append(area)

        gdf_box = gpd.GeoDataFrame(geometry=data_box, crs=crs_epsg)
        gdf_box = gdf_box.assign(ID=ID)
        gdf_points = gpd.GeoDataFrame(geometry=data_point, crs=crs_epsg)
        gdf_points = gdf_points.assign(ID=ID)
        return gdf_points, gdf_box

    def remove_sparse_grids(self):
        joined = gpd.sjoin(self.gdf_grid, self.gdf_points, how="left", op="contains")
        point_counts = joined.groupby("ID_left").size()
        grid_to_remove = point_counts[point_counts <= 4].index
        self.gdf_grid = self.gdf_grid[~self.gdf_grid["ID"].isin(grid_to_remove)]
        rows_to_remove = joined[joined["ID_left"].isin(grid_to_remove)]
        ids_right_to_remove = rows_to_remove["index_right"].dropna().unique()
        self.gdf_points = self.gdf_points[
            ~self.gdf_points["ID"].isin(ids_right_to_remove)
        ]
        self.gdf_boxes = self.gdf_boxes[~self.gdf_boxes["ID"].isin(ids_right_to_remove)]

    def calculate_angle_index(self):
        def distance(p1, p2):
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        def dot_product(v1, v2):
            return v1[0] * v2[0] + v1[1] * v2[1]

        def angle(v1, v2):
            dot = dot_product(v1, v2)
            v1_mag = math.sqrt(dot_product(v1, v1))
            v2_mag = math.sqrt(dot_product(v2, v2))
            cos_theta = dot / (v1_mag * v2_mag)
            theta = math.acos(cos_theta)

            cross = np.cross(v1, v2)
            if cross < 0:
                theta = 2 * math.pi - theta
            return theta

        def count_angle(points):
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
            return count_np

        coordinates_list = [(point.x, point.y) for point in self.gdf_points.geometry]
        angle_index = count_angle(coordinates_list)
        self.gdf_points["angle"] = angle_index

    def calculate_canopy_closure(self):
        with rasterio.open(self.raster_path) as src:
            for i, row in self.gdf_grid.iterrows():
                geom = row["geometry"]
                out_image, out_transform = mask(
                    src, [geom], crop=True, all_touched=True
                )
                valid_pixels = out_image != src.nodata
                total_valid_pixels = valid_pixels.sum()
                if total_valid_pixels == 0:
                    continue
                counts = {val: (out_image == val).sum() for val in range(1, 5)}
                yubidu = sum(counts[val] for val in range(1, 4)) / total_valid_pixels
                self.gdf_grid.loc[i, "yubidu"] = yubidu

    def calculate_canopy_width(self):
        with rasterio.open(self.raster_path) as src:
            for i, row in self.gdf_boxes.iterrows():
                geom = row["geometry"]
                out_image, out_transform = mask(
                    src, [geom], crop=True, all_touched=True
                )
                guanfu = (out_image == 1).sum() * self.resolution * self.resolution
                self.gdf_boxes.loc[i, "guanfu"] = guanfu

    def assign_grid_ids_to_points_and_boxes(self):
        join = gpd.sjoin(
            self.gdf_points, self.gdf_grid[["ID", "geometry"]], how="left", op="within"
        )

        gridid = join["ID_right"]
        self.gdf_points["gridid"] = gridid
        self.gdf_boxes["gridid"] = gridid

    def calculate_number(self):
        joined = gpd.sjoin(self.gdf_grid, self.gdf_points, how="left", op="contains")
        point_counts = joined.groupby("ID_left").size()

        self.gdf_grid["tree_num"] = self.gdf_grid["ID"].map(point_counts)

    def save_or_load(self, method, *args, **kwargs):
        """检查文件存在则加载，否则执行方法并保存结果"""

        filenames = {
            attr: f"temp/{method.__name__}_{attr}.shp"
            for attr in ["gdf_points", "gdf_boxes", "gdf_grid"]
        }

        # 检查所有文件是否存在
        all_files_exist = all(os.path.exists(fname) for fname in filenames.values())

        if not all_files_exist:
            # 至少有一个文件不存在，执行方法
            method(*args, **kwargs)
            # 保存所有相关属性到文件
            for attr, filename in filenames.items():
                if hasattr(self, attr):
                    data = getattr(self, attr)
                    if data is not None:
                        data.to_file(filename)
                        logging.info(f"保存 {filename}")
        else:
            # 所有文件都存在，直接加载
            for attr, filename in filenames.items():
                logging.info(f"加载 {filename}")
                setattr(self, attr, gpd.read_file(filename))

    def run(self):
        self.save_or_load(self.remove_sparse_grids)
        self.save_or_load(self.calculate_angle_index)
        self.save_or_load(self.calculate_canopy_closure)
        self.save_or_load(self.calculate_canopy_width)
        self.save_or_load(self.assign_grid_ids_to_points_and_boxes)
        self.save_or_load(self.calculate_number)


class ForestHarvesting(ForestManagement):
    def __init__(
        self,
        grid_path,
        points_path,
        boxes_path,
        raster_path,
        resolution=0.0238,
        crs_epsg=None,
        ignore_angle=False,
    ):
        super(ForestHarvesting, self).__init__(
            grid_path,
            points_path,
            boxes_path,
            raster_path,
            resolution,
            crs_epsg=crs_epsg,
        )
        super(ForestHarvesting, self).run()
        self.ignore_angle = ignore_angle

    def calculate_tree_harvest_weight(self, ignore_angle=False):
        # 归一化角尺度、郁闭度和冠幅，数量
        normalized_angle = abs(self.gdf_points["angle"] - 0.5) / 0.5
        normalized_canopy_width = 1 - (
            (self.gdf_boxes["guanfu"] - self.gdf_boxes["guanfu"].min())
            / self.gdf_boxes["guanfu"].max()
        )
        self.gdf_grid["n_yubidu"] = (
            self.gdf_grid["yubidu"] - self.gdf_grid["yubidu"].min()
        ) / (self.gdf_grid["yubidu"].max() - self.gdf_grid["yubidu"].min())
        self.gdf_grid["n_num"] = (
            self.gdf_grid["tree_num"] - self.gdf_grid["tree_num"].min()
        ) / self.gdf_grid["tree_num"].max()

        # 计算每棵树的砍伐权重
        for i, row in self.gdf_points.iterrows():
            grid_id = row["gridid"]
            yubidu = self.gdf_grid.loc[self.gdf_grid["ID"] == grid_id, "yubidu"].iloc[0]
            num = self.gdf_grid.loc[self.gdf_grid["ID"] == grid_id, "tree_num"].iloc[0]

            n_yubidu = self.gdf_grid.loc[
                self.gdf_grid["ID"] == grid_id, "n_yubidu"
            ].iloc[0]
            n_num = self.gdf_grid.loc[self.gdf_grid["ID"] == grid_id, "n_num"].iloc[0]

            angle_weight = normalized_angle[i]
            canopy_closure_weight = n_yubidu if yubidu >= 0.7 else 0
            num_weight = n_num if num >= 15 else 0
            canopy_width_weight = normalized_canopy_width[i]

            if not ignore_angle:
                tree_weight = (0.5 * angle_weight + 0.5 * canopy_width_weight) * (
                    0.5 * canopy_closure_weight + 0.5 * num_weight
                )
            else:
                tree_weight = canopy_width_weight * (
                    0.5 * canopy_closure_weight + 0.5 * num_weight
                )
            self.gdf_points.loc[i, "harvest"] = tree_weight
        self.gdf_points.sort_values("harvest", ascending=False, inplace=True)

    def update_raster_and_remove_tree(self, tree_id):
        # 找到对应的gdf_boxes项
        box_to_harvest = self.gdf_boxes.loc[
            self.gdf_boxes["ID"] == tree_id, "geometry"
        ].iloc[0]
        # 读取栅格数据并更新
        with rasterio.open(self.raster_path, "r+") as src:
            out_image, out_transform = mask(
                src, [box_to_harvest], crop=True, all_touched=True
            )
            out_image[out_image == 1] = 0  # 将像素值为1的部分替换为0
            # 将更新后的图像写回文件
            src.write(out_image)

        # 移除gdf_points和gdf_boxes中对应的树木
        self.gdf_points = self.gdf_points[self.gdf_points["ID"] != tree_id]
        self.gdf_boxes = self.gdf_boxes[self.gdf_boxes["ID"] != tree_id]

    def perform_harvesting(self):
        i = 0
        t0 = time.time()
        while True:
            # 确保每个格网中的点数量不低于1500
            # 更新树木的砍伐权重
            self.calculate_tree_harvest_weight(self.ignore_angle)

            # 找到权重最高的树进行砍伐
            tree_to_harvest = self.gdf_points.iloc[0]

            if self.gdf_points.iloc[0]["harvest"] == 0:
                if not self.ignore_angle:
                    self.ignore_angle = True
                else:
                    break  # 所有树的砍伐权重都为0

            tree_id = tree_to_harvest["ID"]

            # 执行砍伐逻辑...
            self.update_raster_and_remove_tree(tree_id)
            logging.info(f"正在砍伐第{i+1}棵树木，ID: {tree_id}")
            # 重新计算整体的郁闭度
            self.calculate_canopy_closure()
            # 重新计算整体的数量
            self.calculate_number()

            i = i + 1

        t1 = time.time()
        print(f"砍了{i}棵树截止,花费时间为{t1-t0}")

    def run(self):
        self.save_or_load(self.perform_harvesting)


if __name__ == "__main__":
    fm = ForestHarvesting(
        grid_path="data/yuwang10m.shp",
        boxes_path="data/geo_box.txt",
        points_path="data/point.txt",
        raster_path="data/svm5tif.tif",
    )
    fm.run()
