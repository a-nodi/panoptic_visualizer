import numpy as np
import yaml
import torch
import argparse
from tqdm import tqdm
from loader import Loader
from utils import unzip
from scene_preprocessor import ScenePreprocessor
from visualizer import Visualizer


def main(loader_configs, preprocess_configs, visualizer_configs, verbose=False):
    
    if verbose:
        print("Loader configs:")
        for key, value in loader_configs.items():
            print(f"{key}: {value}")

        print("\nPreprocess configs:")
        for key, value in preprocess_configs.items():
            print(f"{key}: {value}")
            
        print("\nVisualizer configs:")
        for key, value in visualizer_configs.items():
            print(f"{key}: {value}")
    
    # Initialize loader
    loader = Loader(configs=loader_configs, verbose=verbose)
    
    # Initialize visualizer
    visualizer = Visualizer(configs=visualizer_configs, verbose=verbose)

    voxel_size = preprocess_configs["voxel_size"]

    trajectory = []
    with tqdm(enumerate(loader), total=loader_configs["nframe"], disable=not verbose) as pbar:
        pbar.set_description("Preprocessing scenes")
        for i, scene_data in pbar:
            if i >= loader_configs["nframe"]:
                break
            
            coords, labels = [torch.tensor(coord)[:, :3] for coord in scene_data["pcd"]], [torch.tensor(label) for label in scene_data["labels"]]
            coords, labels = unzip([ScenePreprocessor.filter_out_scene(coord, label) for coord, label in zip(coords, labels)])
            labels = [ScenePreprocessor.one_hot_encode(label) for label in labels]
            accumulated_coords, accumulated_labels = ScenePreprocessor.accumulate_voxel(coords, labels, voxel_size)
            accumulated_coords = accumulated_coords.astype(np.float64)
            intrinsic = scene_data["camera_intrinsic"][0]
            # intrinsic = visualizer.get_custom_instrinsic(visualizer_configs["resolution"])
            camera_poses = scene_data["camera_extrinsic"]
            
            trajectory_element = ScenePreprocessor.create_trajectory(
                coord=accumulated_coords,
                label=accumulated_labels,
                extrinsic=camera_poses[0],
                intrinsic=intrinsic,
                visualization_type=visualizer_configs["camera_view"],
                height=visualizer_configs["BEV_height"]
            )

            trajectory.append(trajectory_element)
    
    wraped_up_scenes = visualizer.wrapup_scenes(trajectory, voxel_size)
    visualizer.visualize(wraped_up_scenes, voxel_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize panoptic segmentation results")
    parser.add_argument("--verbose", type=bool, default=True, help="Whether to print debug information")
    parser.add_argument("--loader_config_path", type=str, default="./panoptic_visualizer/loader_configs.yaml", help="Path to the loader config file")
    parser.add_argument("--preprocess_config_path", type=str, default="./panoptic_visualizer/preprocess_configs.yaml", help="Path to the preprocess config file")
    parser.add_argument("--visualizer_config_path", type=str, default="./panoptic_visualizer/visualizer_configs.yaml", help="Path to the visualizer config file")
    args = parser.parse_args()
    
    verbose = args.verbose
    loader_configs = yaml.load(open(args.loader_config_path, "r"), Loader=yaml.FullLoader)
    preprocess_configs = yaml.load(open(args.preprocess_config_path, "r"), Loader=yaml.FullLoader)
    visualizer_configs = yaml.load(open(args.visualizer_config_path, "r"), Loader=yaml.FullLoader)
    
    main(
        loader_configs=loader_configs,
        preprocess_configs=preprocess_configs,
        visualizer_configs=visualizer_configs,
        verbose=verbose
    )
