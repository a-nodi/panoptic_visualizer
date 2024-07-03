import numpy as np
import torch

class ScenePreprocessor:    
    
    @staticmethod
    def get_coordinate_format(dimension, voxel_size):
        
        x_ = torch.linspace(0., dimension[0] - 0, dimension[0])
        y_ = torch.linspace(-128., dimension[1] - 128, dimension[1])
        z_ = torch.linspace(-20., dimension[2] - 20, dimension[2])
        
        x, y, z = torch.meshgrid(x_ * voxel_size, y_ * voxel_size, z_ * voxel_size)
        h = torch.ones_like(x)
        x, y, z, h = x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3), h.unsqueeze(3)
        
        coord_format = [x, y, z, h]
    
        return coord_format
    
    @staticmethod
    def accumulate_voxel():

    
    
    