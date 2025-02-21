#!/bin/bash

# CUDA Setup Script for ML Pipeline
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting CUDA Setup...${NC}"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run with sudo${NC}"
    exit 1
fi

# Function to check NVIDIA driver
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

# Function to install CUDA toolkit
install_cuda_toolkit() {
    echo -e "\n${YELLOW}Installing CUDA Toolkit...${NC}"
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    apt-get update

    # Install CUDA 12.1
    apt-get -y install cuda-12-1
    
    # Add CUDA paths to environment
    echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
    
    echo -e "${GREEN}CUDA Toolkit installation complete${NC}"
}

# Function to install cuDNN
install_cudnn() {
    echo -e "\n${YELLOW}Installing cuDNN...${NC}"
    
    # Add NVIDIA repository for cuDNN
    apt-get install -y zlib1g
    
    # Install cuDNN 8.9 for CUDA 12.1
    apt-get install -y libcudnn8=8.9.2.*-1+cuda12.1
    apt-get install -y libcudnn8-dev=8.9.2.*-1+cuda12.1
    
    echo -e "${GREEN}cuDNN installation complete${NC}"
}

# Function to install TensorRT
install_tensorrt() {
    echo -e "\n${YELLOW}Installing TensorRT...${NC}"
    
    # Install TensorRT 8.6
    apt-get install -y libnvinfer8=8.6.1.6-1+cuda12.0
    apt-get install -y libnvinfer-dev=8.6.1.6-1+cuda12.0
    apt-get install -y libnvinfer-plugin8=8.6.1.6-1+cuda12.0
    
    echo -e "${GREEN}TensorRT installation complete${NC}"
}

# Function to setup Python environment
setup_python_env() {
    echo -e "\n${YELLOW}Setting up Python environment...${NC}"
    
    # Install Python dependencies
    apt-get install -y python3-pip
    pip3 install --upgrade pip
    
    # Install PyTorch with CUDA support
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other requirements
    cd /path/to/project  # Replace with actual project path
    pip3 install -r requirements.txt
    
    echo -e "${GREEN}Python environment setup complete${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${YELLOW}Verifying installation...${NC}"
    
    # Run CUDA initialization script
    python3 ml/cuda_init.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}CUDA setup verified successfully${NC}"
    else
        echo -e "${RED}CUDA setup verification failed${NC}"
        exit 1
    fi
}

# Main installation process
echo -e "\n${YELLOW}Starting installation process...${NC}"

# 1. Check and install NVIDIA driver
check_nvidia_driver

# 2. Install CUDA toolkit
install_cuda_toolkit

# 3. Install cuDNN
install_cudnn

# 4. Install TensorRT
install_tensorrt

# 5. Setup Python environment
setup_python_env

# 6. Verify installation
verify_installation

echo -e "\n${GREEN}CUDA setup completed successfully!${NC}"
echo -e "${YELLOW}Please reboot your system to ensure all changes take effect.${NC}"

# Create a test script to verify CUDA functionality
cat > test_cuda.py << EOL
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
EOL

echo -e "\n${YELLOW}You can test CUDA functionality by running:${NC}"
echo "python3 test_cuda.py"