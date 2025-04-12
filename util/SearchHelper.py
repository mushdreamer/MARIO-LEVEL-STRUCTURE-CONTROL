import random
import math
import pandas as pd
import numpy as np
import toml
from numpy.linalg import eig
import json

with open('GANTrain/index2str.json') as f:
  index2str = json.load(f)
  
def get_char(x):
    return index2str[str(x)] 

def to_level(number_level):
    result = []
    number_level=eval(number_level)
    for x in number_level:
        #print(x)
        result.append(''.join(get_char(y) for y in x)+'\n')
    result= ''.join(result)
    return result

def make_record_frame(points):
    x, y, f = zip(*points)
    d = {'x':x, 'y':y, 'f':f}
    return pd.DataFrame(d)

def is_structure_valid(number_level):
    array = np.array(eval(number_level))  # 16x56 整数数组
    h, w = array.shape

    MAX_JUMP_HEIGHT = 4
    MAX_JUMP_WIDTH = 5

    # 起点检测：列 0-2 至少要有落脚点
    start_col = array[:, 0:3]
    if np.all(start_col == 0):
        print("Invalid: Start area has no ground")
        return False

    # 遍历每一列，检查墙高
    for col in range(w):
        column = array[:, col]
        top = next((i for i, v in enumerate(column) if v != 0), h)
        wall_height = h - top
        if wall_height > MAX_JUMP_HEIGHT + 4:  # 冗余容差
            print(f"Invalid: Wall too high at column {col}, height: {wall_height}")
            return False

    # 检查平台跨度：如果有连续 0 超过 6 则为无解 gap
    for row in range(h - 1, -1, -1):
        gap_length = 0
        for col in range(w):
            if array[row][col] == 0:
                gap_length += 1
                if gap_length > MAX_JUMP_WIDTH:
                    print(f"Invalid: Gap too wide at row {row}, width: {gap_length}")
                    return False
            else:
                gap_length = 0

    # 检查终点旗帜是否可达
    flag_pos = np.where(array == 7)
    if len(flag_pos[0]) == 0:
        print("Invalid: No goal flag found.")
        return False

    # 你还可以加其他判定（如敌人密度）

    return True

def compute_structure_score(number_level):
    score = 1.0
    array = np.array(eval(number_level))

    # 起点安全性
    if np.all(array[:, 0:3] == 0):
        score -= 0.3

    # 检查墙高分布
    for col in range(array.shape[1]):
        column = array[:, col]
        top = next((i for i, v in enumerate(column) if v != 0), 16)
        wall_height = 16 - top
        if wall_height > 10:
            score -= 0.1

    # 检查是否存在终点
    if not np.any(array == 7):
        score -= 0.3

    # 限制 score 在 [0, 1]
    score = max(0.0, min(1.0, score))
    return score

class Individual:

    def __init__(self):
        pass

    def read_mario_features(self):
        pass

class DecompMatrix:
    def __init__(self, dimension):
        self.C = np.eye(dimension, dtype=np.float_) 
        self.eigenbasis = np.eye(dimension, dtype=np.float_)
        self.eigenvalues = np.ones((dimension,), dtype=np.float_)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=np.float_)

    def update_eigensystem(self):
        for i in range(len(self.C)):
            for j in range(i):
                self.C[i,j] = self.C[j,i]
        
        self.eigenvalues, self.eigenbasis = eig(self.C) 
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenbasis = np.real(self.eigenbasis)
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)

        for i in range(len(self.C)):
            for j in range(i+1):
                self.invsqrt[i,j] = self.invsqrt[j,i] = sum(
                        self.eigenbasis[i,k] * self.eigenbasis[j,k]
                        / self.eigenvalues[k] ** 0.5 for k in range(len(self.C))
                    )

class FeatureMap:

   def __init__(self, max_individuals, feature_ranges, resolutions):
      self.max_individuals = max_individuals
      self.feature_ranges = feature_ranges
      self.resolutions = resolutions

      self.elite_map = {}
      self.elite_indices = []

      self.num_individuals_added = 0

   def get_feature_index(self, feature_id, feature):
      feature_range = self.feature_ranges[feature_id]
      if feature-1e-9 <= feature_range[0]:
         return 0
      if feature_range[1] <= feature + 1e-9:
         return self.resolutions[feature_id]-1

      gap = feature_range[1] - feature_range[0]
      pos = feature - feature_range[0]
      index = int((self.resolutions[feature_id] * pos + 1e-9) / gap)
      return index

   def get_index(self, cur):
      return tuple(self.get_feature_index(i, f) for i, f in enumerate(cur.features))

   def add_to_map(self, to_add):
      index = self.get_index(to_add)

      replaced_elite = False
      if index not in self.elite_map:
         self.elite_indices.append(index)
         self.elite_map[index] = to_add
         replaced_elite = True
         to_add.delta = (1, to_add.fitness)
      elif self.elite_map[index].fitness < to_add.fitness:
         to_add.delta = (0, to_add.fitness-self.elite_map[index].fitness)
         self.elite_map[index] = to_add
         replaced_elite = True

      return replaced_elite

   def add(self, to_add):
      self.num_individuals_added += 1
      replaced_elite = self.add_to_map(to_add)
      return replaced_elite

   def get_random_elite(self):
      pos = random.randint(0, len(self.elite_indices)-1)
      index = self.elite_indices[pos]
      return self.elite_map[index]

