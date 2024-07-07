import json
import numpy as np
import os.path as osp
from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
from nuscenes.utils.data_classes import LidarSegPointCloud
from utils import convert_quaternion_to_rotation_matrix

class Loader:
    # TODO: Make it work when is_gt is False
    def __init__(self, configs, verbose=False):
        self.nsample_per_frame = configs["nsample_per_frame"]
        self.verbose = verbose
        
        if self.verbose:
            print("Initializing Loader...")
        
        # Initialize NuScenes dataset
        self.nusc = NuScenes(
            version=configs["version"], 
            dataroot=configs["dataroot"], 
            verbose=self.verbose,
            map_resolution=configs["map_resolution"]
        )
        
        # Load Tracking Config
        with open(configs["tracking_config_path"], "r") as f:
            self.tracking_configs = TrackingConfig.deserialize(json.load(f))
        
        # Load tracking gt data
        if configs["is_gt"]:  # Load GT data
            self.tracking_boxes = load_gt(
                nusc=self.nusc,
                eval_split=configs["eval_split"],
                box_cls=TrackingBox,
                verbose=self.verbose
            )
        
        else:  # Load tracking prediction data
            self.tracking_boxes, self.meta = load_prediction(
                result_path=configs["result_path"],
                max_boxes_per_sample=configs["max_boxes_per_sample"],
                box_cls=TrackingBox,
                verbose=self.verbose
            )
        
        # Get sample metadata
        self.sample_tokens = [sample_token for sample_token in self.tracking_boxes.boxes.keys()]
        self.samples = [self.nusc.get('sample', sample_token) for sample_token in self.sample_tokens]
        
        # Get sensor tokens
        self.lidar_tokens = [sample['data']['LIDAR_TOP'] for sample in self.samples]
        self.camera_tokens = [sample['data']['CAM_FRONT'] for sample in self.samples]    
        
        # Get sensor data
        self.list_of_lidar_data = [self.nusc.get('sample_data', lidar_token) for lidar_token in self.lidar_tokens]
        self.list_of_camera_data = [self.nusc.get('sample_data', camera_token) for camera_token in self.camera_tokens]
        
        # Get point cloud and label paths
        self.list_of_pcd_path = [osp.join(self.nusc.dataroot, self.nusc.get('sample_data', lidar_token)["filename"]) for lidar_token in self.lidar_tokens]
        self.list_of_label_path = [osp.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_token)["filename"]) for lidar_token in self.lidar_tokens]
        
        # Get camera pose and intrinsic
        self.list_of_camera_calibrated_sensor = [self.nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token']) for camera_data in self.list_of_camera_data]
        self.list_of_camera_intrinsic = [np.array(camera_calibrated_sensor['camera_intrinsic']) for camera_calibrated_sensor in self.list_of_camera_calibrated_sensor]
        self.list_of_camera_rotation = [camera_calibrated_sensor['rotation'] for camera_calibrated_sensor in self.list_of_camera_calibrated_sensor]
        self.list_of_camera_translation = [camera_calibrated_sensor['translation'] for camera_calibrated_sensor in self.list_of_camera_calibrated_sensor]
        self.list_of_camera_extrinsic = [Loader.make_extrinsic_matrix(rotation, translation) for rotation, translation in zip(self.list_of_camera_rotation, self.list_of_camera_translation)]
        
        # Get lidar pose
        self.list_of_lidar_calibrated_sensor = [self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token']) for lidar_data in self.list_of_lidar_data]
        self.list_of_lidar_rotation = [lidar_calibrated_sensor['rotation'] for lidar_calibrated_sensor in self.list_of_lidar_calibrated_sensor]
        self.list_of_lidar_translation = [lidar_calibrated_sensor['translation'] for lidar_calibrated_sensor in self.list_of_lidar_calibrated_sensor]
        self.list_of_lidar_extrinsic = [Loader.make_extrinsic_matrix(rotation, translation) for rotation, translation in zip(self.list_of_lidar_rotation, self.list_of_lidar_translation)]

        if self.verbose:
            print("Loader initialized.")
    
    @staticmethod
    def make_extrinsic_matrix(rotation, translation):
        """
        Make extrinsic matrix from rotation and translation.
        Args:
            rotation: 3x3 rotation matrix.
            translation: 3x1 translation vector.
        Returns:
            4x4 extrinsic matrix.
        """
        rotation = convert_quaternion_to_rotation_matrix(rotation)        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        
        return extrinsic
    
    @staticmethod
    def load_lidar(pcd_path, label_path):
        return LidarSegPointCloud(pcd_path, label_path)

    def __len__(self):
            return len(self.sample_tokens) - self.config["nsample_per_frame"] + 1
    
    def __getitem__(self, idx):
        list_of_pcd = []
        list_of_label = []
        list_of_camera_intrinsic = []
        list_of_camera_extrinsic = []
        list_of_lidar_extrinsic = []
        list_of_tracking_boxes = []
        
        if idx >= len(self.sample_tokens):
            raise IndexError("Index out of range")
        
        for i in range(self.nsample_per_frame):
            
            if idx + i >= len(self.sample_tokens):
                break
            
            pcd_path = self.list_of_pcd_path[idx + i]
            label_path = self.list_of_label_path[idx + i]
            
            lidar_seg_pcd = Loader.load_lidar(pcd_path, label_path)
            
            list_of_pcd.append(lidar_seg_pcd.points)
            list_of_label.append(lidar_seg_pcd.labels)
            
            list_of_camera_intrinsic.append(self.list_of_camera_intrinsic[idx + i])
            list_of_camera_extrinsic.append(self.list_of_camera_extrinsic[idx + i])
            list_of_lidar_extrinsic.append(self.list_of_lidar_extrinsic[idx + i])
            list_of_tracking_boxes.append(self.tracking_boxes[self.sample_tokens[idx + i]])
            
        return {
            "pcd": list_of_pcd,
            "labels": list_of_label,
            "camera_intrinsic": list_of_camera_intrinsic,
            "camera_extrinsic": list_of_camera_extrinsic,
            "lidar_extrinsic": list_of_lidar_extrinsic,
            "tracking_boxes": list_of_tracking_boxes
        }

