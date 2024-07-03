import yaml
import argparse
from loader import Loader
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
    visualizer = Visualizer(configs=visualizer_configs)

    coord_format = ScenePreprocessor.get_coordinate_format(
        dimension=preprocess_configs["dimension"],
        voxel_size=preprocess_configs["voxel_size"]
    )

    
    
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
