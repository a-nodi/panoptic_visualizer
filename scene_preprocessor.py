import numpy as np
import torch
import MinkowskiEngine as ME
from utils import convert_quaternion_to_rotation_matrix

NUMBER_OF_NUSCENES_LABEL_TYPES = 32


class ScenePreprocessor:    

    @staticmethod
    def filter_out_scene(coords, labels):
        is_valid = (labels != 0)  # Filter out invalid labels
        valid_coords, valid_labels = coords[is_valid], labels[is_valid]  # Filter out invalid coords
    
        return valid_coords, valid_labels
    
    @staticmethod
    def one_hot_encode(label):
        return torch.nn.functional.one_hot(label.long(), NUMBER_OF_NUSCENES_LABEL_TYPES).float()
    
    @staticmethod
    def accumulate_voxel(coords, labels, voxel_size):  # sequence_of_labels
        concatenated_coords = torch.cat(coords, dim=0)
        concatenated_labels = torch.cat(labels, dim=0)

        quantized_scene = ME.SparseTensor(
            features=concatenated_labels,
            coordinates=ME.utils.batched_coordinates([concatenated_coords / voxel_size]),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM
        )
        
        accumulated_coord, accumulated_label = quantized_scene.decomposed_coordinates_and_features
        accumulated_coord = accumulated_coord[0] * voxel_size
        accumulated_label = accumulated_label[0].argmax(dim=-1)
        
        accumulated_coord, accumulated_label = accumulated_coord.numpy(force=True), accumulated_label.numpy(force=True)  # Convert torch tensor to numpy array
        
        accumulated_coord = np.round(accumulated_coord / voxel_size) * voxel_size  # Quantize the coordinates
        
        return accumulated_coord, accumulated_label
    
    @staticmethod
    def transform_to_fpv_pose(extrinsic, height):
        extrinsic[:3, :3] = np.array(  # OpenGL convention: flip y and z axis
            [[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]
        ) @ np.array(  # camera pose correction
            [[0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]]
        ) @ extrinsic[:3, :3]
        
        # extrinsic[0, 3], extrinsic[1, 3], extrinsic[2, 3] = extrinsic[0, 3], extrinsic[1, 3], extrinsic[2, 3]  # Reset axis            
        
        return extrinsic
    
    @staticmethod
    def transform_to_bev_pose(extrinsic, height):
        # Set view to bird eye view
        extrinsic[:3, :3] = np.array(
            [[1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]]
        )
        
        # extrinsic[1, 3] = -extrinsic[1, 3]
        extrinsic[2, 3] = height  # Set height
    
        return extrinsic
    
    @staticmethod
    def get_tracking_ids(coords, tracking_boxes):
        tracking_ids = np.full((coords.shape[0],), -1, dtype=np.int32)
        int_mapping = {}
        reverse_mapping = {}

        _tracking_boxes = []
        for tracking_boxes_for_one_frame in tracking_boxes:
            _tracking_boxes.extend(tracking_boxes_for_one_frame)
        
        # Annotate coords with tracking ids
        for tracking_box in _tracking_boxes:
            
            translation = tracking_box.translation
            rotation = convert_quaternion_to_rotation_matrix(tracking_box.rotation)
            size = tracking_box.size
            tracking_id = tracking_box.tracking_id
            
            if tracking_id not in int_mapping.keys():
                int_mapping[tracking_id] = len(int_mapping)
            
            translated_coords = coords[:, :3] - translation
            rotated_coords = translated_coords @ rotation.T
            is_x = (-size[0] / 2 <= rotated_coords[:, 0]) & (rotated_coords[:, 0] <= size[0] / 2)
            is_y = (-size[1] / 2 <= rotated_coords[:, 1]) & (rotated_coords[:, 1] <= size[1] / 2)
            is_z = (-size[2] / 2 <= rotated_coords[:, 2]) & (rotated_coords[:, 2] <= size[2] / 2)
            is_inside = is_x & is_y & is_z
            
            tracking_ids[is_inside] = int_mapping[tracking_id]
            reverse_mapping[int_mapping[tracking_id]] = tracking_id
        
        tracking_ids = [reverse_mapping[i] if i != -1 else "" for i in tracking_ids ]
        
        for _ in tracking_ids:
            if _ != "":
                print(_)
        return tracking_ids
    
    @staticmethod
    def create_trajectory(coord, label, extrinsic, intrinsic, tracking_ids, visualization_type, height):
        if visualization_type == "FPV":
            transform_pose = ScenePreprocessor.transform_to_fpv_pose
            
        elif visualization_type == "BEV":
            transform_pose = ScenePreprocessor.transform_to_bev_pose
        
        else:
            raise ValueError(f"Invalid visualization type: {visualization_type}")
        
        extrinsic = transform_pose(extrinsic, height)
        
        trajectory = {
            "coord": coord,
            "label": label,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "tracking_ids": tracking_ids,
        }
        
        return trajectory
