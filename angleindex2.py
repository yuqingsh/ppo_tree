import math
from tqdm import tqdm

# 计算两点之间的距离
import numpy as np


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# 计算两个向量的点积
def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


# 计算两个向量之间的夹角（弧度）
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
    for i, p in tqdm(enumerate(points)):
        # 取最近的点list=>(distance,point_index)
        distances = [
            (distance(p, other), j) for j, other in enumerate(points) if j != i
        ]
        distances.sort()
        # 取出距离最近的点索引
        nearst_point_index = [i for dis, i in distances[:4]]
        angles = []
        for j in range(len(nearst_point_index) - 1):
            v1 = (
                points[nearst_point_index[0]][0] - p[0],
                points[nearst_point_index[0]][1] - p[1],
            )
            v2 = (
                points[nearst_point_index[j + 1]][0] - p[0],
                points[nearst_point_index[j + 1]][1] - p[1],
            )
            angles.append((angle(v1, v2), nearst_point_index[j + 1]))

        angles.sort()
        angles_index = [i for ang, i in angles[:3]]
        angles_index.insert(0, nearst_point_index[0])
        count = 0
        for k in range(len(angles_index)):
            if k <= len(nearst_point_index) - 2:
                v1 = (
                    points[angles_index[k]][0] - p[0],
                    points[angles_index[k]][1] - p[1],
                )
                v2 = (
                    points[angles_index[k + 1]][0] - p[0],
                    points[angles_index[k + 1]][1] - p[1],
                )
            elif k == len(nearst_point_index) - 1:
                v1 = (
                    points[angles_index[k]][0] - p[0],
                    points[angles_index[k]][1] - p[1],
                )
                v2 = (
                    points[angles_index[0]][0] - p[0],
                    points[angles_index[0]][1] - p[1],
                )
            if angle(v1, v2) > math.radians(279):
                # print('大于288')
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


if __name__ == "__main__":
    import geopandas as gpd

    # 加载shp文件
    gdf = gpd.read_file(r"points/bounding_boxes_point.shp")
    # 初始化一个列表来存储坐标
    coordinates_list = []

    # 遍历每个点要素
    for point in gdf.geometry:
        # 获取点的x和y坐标
        x, y = point.x, point.y
        # 将坐标添加到列表中
        coordinates_list.append((x, y))
    data = np.array(coordinates_list)

    # txt_file = r'F:\biyelunwen\shamu\result\predict\box\0.75\nms\point.txt'
    # # 示例点集合
    # data = np.loadtxt(txt_file, delimiter=',')
    angle = count_angle(data)
    result = np.column_stack((data, angle))
    np.savetxt("../data/angleindex4.txt", result, fmt="%f", delimiter=",")

    # # 计算角度大于72度的数量
    # result = count_angles(points)
    # print("大于72度的角度数量：", result)
