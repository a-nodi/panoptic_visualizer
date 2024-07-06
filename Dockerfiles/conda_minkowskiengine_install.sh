conda init bash
source activate open3d
conda install gxx=11.4.0
conda install pybind11
conda install -y openblas-devel -c anaconda
cd MinkowskiEngine
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6+PTX;8.9;9.0
CC=g++ python3 setup.py install --force_cuda --cuda_home=/usr/local/cuda --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas