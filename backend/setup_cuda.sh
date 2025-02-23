#!/bin/bash

set -e  

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting CUDA Setup...${NC}"


if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run with sudo${NC}"
    exit 1
fi

check_nvidia_driver() {
    echo -e "\n${YELLOW}Checking NVIDIA Driver...${NC}"
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA driver is installed and functioning${NC}"
        nvidia-smi
    else
        echo -e "${RED}NVIDIA driver not found${NC}"
        echo -e "${YELLOW}Installing NVIDIA driver...${NC}"
        ubuntu-drivers devices
        ubuntu-drivers autoinstall
        echo -e "${GREEN}Driver installation complete. Please reboot your system.${NC}"
        exit 0
    fi
}

install_cuda_toolkit() {
    echo -e "\n${YELLOW}Installing CUDA Toolkit...${NC}"
    
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    apt-get update

    apt-get -y install cuda-12-1
    
    echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
    
    echo -e "${GREEN}CUDA Toolkit installation complete${NC}"
}

install_cudnn() {
    echo -e "\n${YELLOW}Installing cuDNN...${NC}"
    
    apt-get install -y zlib1g
    
    apt-get install -y libcudnn8=8.9.2.*-1+cuda12.1
    apt-get install -y libcudnn8-dev=8.9.2.*-1+cuda12.1
    
    echo -e "${GREEN}cuDNN installation complete${NC}"
}

install_tensorrt() {
    echo -e "\n${YELLOW}Installing TensorRT...${NC}"
    
    apt-get install -y libnvinfer8=8.6.1.6-1+cuda12.0
    apt-get install -y libnvinfer-dev=8.6.1.6-1+cuda12.0
    apt-get install -y libnvinfer-plugin8=8.6.1.6-1+cuda12.0
    
    echo -e "${GREEN}TensorRT installation complete${NC}"
}

setup_python_env() {
    echo -e "\n${YELLOW}Setting up Python environment...${NC}"
    
    apt-get install -y python3-pip
    pip3 install --upgrade pip
    
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    cd /path/to/project  
    pip3 install -r requirements.txt
    
    echo -e "${GREEN}Python environment setup complete${NC}"
}

verify_installation() {
    echo -e "\n${YELLOW}Verifying installation...${NC}"
    
    python3 ml/cuda_init.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}CUDA setup verified successfully${NC}"
    else
        echo -e "${RED}CUDA setup verification failed${NC}"
        exit 1
    fi
}

echo -e "\n${YELLOW}Starting installation process...${NC}"

check_nvidia_driver

install_cuda_toolkit

install_cudnn

install_tensorrt

setup_python_env

verify_installation

echo -e "\n${GREEN}CUDA setup completed successfully!${NC}"
echo -e "${YELLOW}Please reboot your system to ensure all changes take effect.${NC}"

cat > test_cuda.py << EOL
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
EOL

echo -e "\n${YELLOW}You can test CUDA functionality by running:${NC}"
echo "python3 test_cuda.py"