import os
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box

# 参数设置
input_tif = "classification.tif"
POINT_PATH = "points/bounding_boxes_point.shp"
BBOX_PATH = "bounding_boxes/bounding_boxes_houchuli.shp"
output_dir = "grids"


# 创建主输出目录
os.makedirs(output_dir, exist_ok=True)

with rasterio.open(input_tif) as src:
    profile = src.profile
    height, width = src.shape
    points = gpd.read_file(POINT_PATH).to_crs(src.crs)  # 坐标系转换[10]()
    boxes = gpd.read_file(BBOX_PATH).to_crs(src.crs)
    block_size = height // 4

    for x, i in enumerate(range(0, height, block_size)):  # 遍历栅格数据[1]()
        for y, j in enumerate(range(0, width, block_size)):
            window = Window(j, i, block_size, block_size)
            block_folder = os.path.join(output_dir, f"block_{x}_{y}")
            os.makedirs(block_folder, exist_ok=True)  # 创建块专属目录[7]()

            # 处理栅格数据
            block_transform = src.window_transform(window)
            block_data = src.read(window=window)
            block_profile = profile.copy()
            block_profile.update(
                {
                    "height": window.height,
                    "width": window.width,
                    "transform": block_transform,
                }
            )
            # 保存栅格文件[2]()
            raster_path = os.path.join(block_folder, f"raster_{x}_{y}.tif")
            with rasterio.open(raster_path, "w", **block_profile) as dst:
                dst.write(block_data)

            # 处理矢量数据
            left, bottom = block_transform * (0, block_size)
            right, top = block_transform * (block_size, 0)
            block_bbox = box(left, bottom, right, top)
            block_points = points[points.within(block_bbox)]
            block_boxes = boxes.loc[block_points.index]

            # 保存点数据[5]()
            if not block_points.empty:
                vector_path = os.path.join(block_folder, f"points_{x}_{y}.shp")
                block_points.to_file(vector_path)

            # 保存边界框数据[6]()
            if not block_boxes.empty:
                vector_path = os.path.join(block_folder, f"boxes_{x}_{y}.shp")
                block_boxes.to_file(vector_path)
