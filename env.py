import gymnasium as gym
import rasterio
from rasterio.mask import mask
import shutil
import os
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os
import logging
import math
import numpy as np
from tqdm import tqdm
from utils import ForestManager

MAX_TREE_CUT = 1500
GRID_SIZE = 20
