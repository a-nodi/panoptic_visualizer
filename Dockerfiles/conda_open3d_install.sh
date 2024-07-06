conda init bash
conda create -n open3d -y python=3.11
source activate open3d
cd ~/Open3D-0.18.0/build
cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF -DPYTHON3_ROOT=/root/miniconda3/envs/open3d/bin/python3 -DBUILD_CUDA_MODULE=ON ..
make -j$(nproc)
make install-pip-package
conda install -c conda-forge -y gcc=12.1.0    