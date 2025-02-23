from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import math

@dataclass
class VLLMConfig:
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    kv_cache_size: int = 1024
    tensor_parallel_size: int = 1
    prefill_chunk_size: int = 512
    max_num_batched_tokens: int = 8192

class KVCache:
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.cache = {}
        self.cache_lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Tuple[Tensor, Tensor]]:
        with self.cache_lock:
            return self.cache.get(key)
            
    def set(self, key: str, k: Tensor, v: Tensor):
        with self.cache_lock:
            if len(self.cache) * 8 * (k.element_size() + v.element_size()) > self.config.kv_cache_size * 1024 * 1024:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (k, v)

class VLLMEngine:
    
    def __init__(
        self,
        model: nn.Module,
        config: VLLMConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.config = config
        self.device = device
        self.kv_cache = KVCache(config)
        
        if config.tensor_parallel_size > 1:
            self.setup_tensor_parallelism()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.request_queue = Queue()
        self.response_queues: Dict[str, Queue] = {}
        
        self.worker_thread = threading.Thread(target=self._process_batch_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def setup_tensor_parallelism(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for tensor parallelism")
            
        num_gpus = torch.cuda.device_count()
        if num_gpus < self.config.tensor_parallel_size:
            raise RuntimeError(f"Requested {self.config.tensor_parallel_size} GPUs but only {num_gpus} available")
            
        devices = [f'cuda:{i}' for i in range(self.config.tensor_parallel_size)]
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=devices,
            output_device=devices[-1]
        )

    def _process_batch_queue(self):
        while True:
            batch = []
            batch_size = 0
            
            while batch_size < self.config.max_num_batched_tokens:
                try:
                    request = self.request_queue.get(timeout=0.1)
                    batch.append(request)
                    batch_size += len(request[1])
                except:
                    break
                    
            if not batch:
                continue
                
            eeg_features = []
            bio_features = []
            request_ids = []
            
            for req_id, eeg, bio in batch:
                eeg_features.append(eeg)
                bio_features.append(bio)
                request_ids.append(req_id)
                
            max_len = max(len(f) for f in eeg_features)
            eeg_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f) for f in eeg_features],
                batch_first=True
            )
            bio_padded = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f) for f in bio_features],
                batch_first=True
            )
            
            with torch.no_grad():
                results = self.model(
                    eeg_padded.to(self.device),
                    bio_padded.to(self.device)
                )
                
            for i, req_id in enumerate(request_ids):
                self.response_queues[req_id].put(
                    {k: v[i] for k, v in results.items()}
                )
                del self.response_queues[req_id]
    
    def prefill(self, eeg_features: np.ndarray, bio_features: np.ndarray) -> str:
        cache_key = f"{hash(eeg_features.tobytes())}-{hash(bio_features.tobytes())}"
        
        num_chunks = math.ceil(len(eeg_features) / self.config.prefill_chunk_size)
        
        for i in range(num_chunks):
            start_idx = i * self.config.prefill_chunk_size
            end_idx = min((i + 1) * self.config.prefill_chunk_size, len(eeg_features))
            
            eeg_chunk = torch.tensor(eeg_features[start_idx:end_idx]).to(self.device)
            bio_chunk = torch.tensor(bio_features[start_idx:end_idx]).to(self.device)
            
            with torch.no_grad():
                k, v = self.model.pipeline.generate_kv_cache(eeg_chunk, bio_chunk)
                self.kv_cache.set(f"{cache_key}-{i}", k, v)
        
        return cache_key
    
    async def predict_async(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        cache_key: Optional[str] = None,
        return_explanations: bool = False
    ) -> Dict:
        request_id = f"{threading.get_ident()}-{hash(eeg_features.tobytes())}"
        response_queue = Queue()
        self.response_queues[request_id] = response_queue
        
        self.request_queue.put((request_id, eeg_features, bio_features))
        
        results = response_queue.get()
        
        predictions = {
            'classifications': results['final']['classifications'],
            'confidence_scores': results['final']['confidence']
        }
        
        if return_explanations:
            predictions['explanations'] = self.model.pipeline.explain_pipeline(results)
            predictions['reduction_stats'] = self.model.pipeline.get_reduction_stats(results)
        
        return predictions
    
    def predict(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        return_explanations: bool = False
    ) -> Dict:
        import asyncio
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.predict_async(
                eeg_features,
                bio_features,
                return_explanations=return_explanations
            )
        )