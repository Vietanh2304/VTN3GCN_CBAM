import math
import numpy as np
from numpy import random
from einops import einsum
import torch
class Rotate(object):
    def __init__(self, range_angle, num_frames, num_nodes, point):
        self.range_angle = range_angle
        self.num_frames = num_frames
        self.num_nodes = num_nodes
        self.point = point
    def __call__(self, data, label):
        # data, label = sample
        data = data.double()
        angle = math.radians(random.uniform((-1)*self.range_angle, self.range_angle))
        rotation_matrix = torch.Tensor([[math.cos(angle), (-1)*math.sin(angle)], 
                                    [math.sin(angle), math.cos(angle)]])
        # print(type(rotation_matrix))
        ox, oy = self.point
        data[0, :, :] -= ox
        data[1, :, :] -= oy
        
        result = einsum(rotation_matrix.double(), data, "a b, b c d e -> a c d e") + 0.5

        return result, label


class Left(object):
    def __init__(self, width):
        self.width = width
    def __call__(self, data, label):
        idx = find_frames(data)
        p = random.random()
        if p > 0.5:
            data[0, :idx, :] -= self.width
        return data, label
    
class Right(object):
    def __init__(self, width):
        self.width = width
    def __call__(self, data, label):
        idx = find_frames(data)
        p = random.random()
        if p > 0.5:
            data[0, :idx, :] += self.width
        return data, label
    
class GaussianNoise(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    def __call__(self, data, label):
        # C, T, V, 1
        print(data.size())
        noise = torch.randn(size = data.size())
        data = data + noise
        return data, label

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, data, label):
        for t in self.transforms:
            data, label = t(data, label)
        return data, label
    
def find_frames(data):
    for i in range(data.shape[1]):
        if(data[:, i, :][0][0] == 0):
            # print(i)
            return i
    return data.shape[1]