import torch
import tensorrt as trt
import subprocess
import sys
import os
from typing import Dict, Optional, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_nvidia_smi() -> Dict[str, str]:
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected:\n%s", result.stdout)
            return {'status': 'success', 'output': result.stdout}
        else:
            logger.error("nvidia-smi command failed")
            return {'status': 'error', 'output': result.stderr}
    except FileNotFoundError:
        logger.error("nvidia-smi not found. Please install NVIDIA drivers.")
        return {'status': 'error', 'output': 'nvidia-smi not found'}

def validate_cuda_installation() -> Dict[str, bool]:
    validation = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_initialized': False,
        'tensorrt_available': False,
        'fp16_supported': False,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if validation['cuda_available']:
        try:
            torch.cuda.init()
            validation['cuda_initialized'] = True
            
            try:
                trt.Logger()
                validation['tensorrt_available'] = True
            except Exception as e:
                logger.warning(f"TensorRT not available: {e}")
            
            if torch.cuda.get_device_capability()[0] >= 7:
                validation['fp16_supported'] = True
            
            logger.info("CUDA validation results: %s", validation)
        except Exception as e:
            logger.error(f"Error during CUDA validation: {e}")
    else:
        logger.error("CUDA is not available")
    
    return validation

def setup_cuda_cache(cache_dir: Optional[str] = None) -> str:
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.cache/torch/cuda')

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ['CUDA_CACHE_PATH'] = str(cache_dir)

    logger.info(f"CUDA cache directory set to: {cache_dir}")
    return str(cache_dir)

def optimize_cuda_settings():
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.cuda.empty_cache()

    logger.info("CUDA settings optimized")

def get_cuda_info() -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {'status': 'CUDA not available'}
    
    info = {
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
        'device_name': torch.cuda.get_device_name(0),
        'memory_allocated': f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
        'memory_cached': f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
        'max_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    }
    
    return info

def initialize_cuda_environment(
    cache_dir: Optional[str] = None,
    optimize: bool = True
) -> Dict[str, Any]:
    results = {
        'nvidia_smi': check_nvidia_smi(),
        'validation': validate_cuda_installation(),
        'cuda_info': get_cuda_info()
    }
    
    if results['validation']['cuda_available']:
        results['cache_dir'] = setup_cuda_cache(cache_dir)
        
        if optimize:
            optimize_cuda_settings()
            results['optimized'] = True
    
    return results

def main():
    logger.info("Initializing CUDA environment...")
    results = initialize_cuda_environment()
    
    if results['validation']['cuda_available']:
        logger.info("CUDA initialization successful!")
        logger.info("\nCUDA Information:")
        for key, value in results['cuda_info'].items():
            logger.info(f"{key}: {value}")
    else:
        logger.error("CUDA initialization failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()