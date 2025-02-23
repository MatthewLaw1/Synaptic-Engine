"""CUDA and TensorRT setup for optimized model inference."""

import torch
import torch.cuda
import tensorrt as trt
import torch2trt
from typing import Optional, Dict, Any, Union
import logging
import os
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class CUDAManager:
    """Manages CUDA setup and optimization."""
    
    def __init__(
        self,
        device_id: int = 0,
        enable_tensorrt: bool = True,
        fp16_mode: bool = True,
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 32
    ):
        self.device_id = device_id
        self.enable_tensorrt = enable_tensorrt
        self.fp16_mode = fp16_mode
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        
        # Initialize CUDA
        self._setup_cuda()
        
        if enable_tensorrt:
            self._setup_tensorrt()
    
    def _setup_cuda(self):
        """Initialize CUDA device and settings."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Set device
        torch.cuda.set_device(self.device_id)
        self.device = torch.device(f'cuda:{self.device_id}')
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Log CUDA info
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(self.device_id)}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability(self.device_id)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(self.device_id) / 1e9:.2f} GB")
        
        # Set optimal CUDA settings
        torch.cuda.empty_cache()
        if self.fp16_mode:
            # Enable automatic mixed precision
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _setup_tensorrt(self):
        """Initialize TensorRT."""
        logger.info("Initializing TensorRT")
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Create TensorRT builder and config
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Set config options
        self.config.max_workspace_size = self.max_workspace_size
        if self.fp16_mode:
            self.config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
    def optimize_model(
        self,
        model: torch.nn.Module,
        input_shapes: Dict[str, tuple],
        cache_dir: Optional[str] = None,
        force_rebuild: bool = False
    ) -> torch.nn.Module:
        """
        Optimize model using TensorRT.
        
        Args:
            model: PyTorch model to optimize
            input_shapes: Dictionary of input names and their shapes
            cache_dir: Directory to cache optimized models
            force_rebuild: Whether to force rebuild cached models
            
        Returns:
            Optimized model
        """
        if not self.enable_tensorrt:
            return model
        
        model = model.to(self.device)
        model.eval()
        
        # Create cache path if provided
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            model_name = model.__class__.__name__
            cache_path = Path(cache_dir) / f"{model_name}_tensorrt.pth"
        
        # Load cached model if available
        if cache_path and cache_path.exists() and not force_rebuild:
            logger.info(f"Loading cached TensorRT model from {cache_path}")
            return torch.load(cache_path)
        
        # Create sample inputs
        sample_inputs = {
            name: torch.randn(shape).to(self.device)
            for name, shape in input_shapes.items()
        }
        
        # Convert to TensorRT
        logger.info("Converting model to TensorRT")
        trt_model = torch2trt.torch2trt(
            model,
            [sample_inputs],
            fp16_mode=self.fp16_mode,
            max_workspace_size=self.max_workspace_size,
            max_batch_size=self.max_batch_size
        )
        
        # Save optimized model
        if cache_path:
            logger.info(f"Saving optimized model to {cache_path}")
            torch.save(trt_model, cache_path)
        
        return trt_model
    
    @torch.cuda.amp.autocast()
    def inference(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform optimized inference.
        
        Args:
            model: Model to use for inference
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        # Convert numpy arrays to tensors
        tensor_inputs = {
            name: torch.from_numpy(input_data).to(self.device)
                if isinstance(input_data, np.ndarray)
                else input_data.to(self.device)
            for name, input_data in inputs.items()
        }
        
        with torch.no_grad():
            outputs = model(**tensor_inputs)
            
            # Convert outputs to CPU if needed
            if isinstance(outputs, dict):
                return {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs.items()
                }
            else:
                return outputs.cpu()
    
    def profile_model(
        self,
        model: torch.nn.Module,
        sample_inputs: Dict[str, torch.Tensor],
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Profile model performance.
        
        Args:
            model: Model to profile
            sample_inputs: Sample inputs for profiling
            num_warmup: Number of warmup runs
            num_runs: Number of profiling runs
            
        Returns:
            Dictionary containing profiling metrics
        """
        model = model.to(self.device)
        model.eval()
        
        # Move inputs to device
        device_inputs = {
            name: tensor.to(self.device)
            for name, tensor in sample_inputs.items()
        }
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(**device_inputs)
        
        # Profile
        logger.info("Profiling...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                _ = model(**device_inputs)
                end_event.record()
                
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'memory_mb': torch.cuda.max_memory_allocated(self.device) / 1e6
        }
        
        logger.info(f"Profiling results: {stats}")
        return stats
    
    def cleanup(self):
        """Clean up CUDA resources."""
        torch.cuda.empty_cache()
        if hasattr(self, 'builder'):
            del self.builder
        if hasattr(self, 'config'):
            del self.config

def create_cuda_manager(**kwargs) -> CUDAManager:
    """Factory function to create CUDA manager."""
    return CUDAManager(**kwargs)