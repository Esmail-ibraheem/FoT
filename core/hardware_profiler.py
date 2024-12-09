"""
Hardware Profiler for Optimus-Megatron
Detects and profiles hardware resources for optimal parallelization strategy.
"""

import torch
import torch.cuda as cuda
from typing import Dict, List, Optional, Tuple
import numpy as np
import psutil
import os
import subprocess
from dataclasses import dataclass
import time

# Megatron-specific imports
from megatron import get_args
from megatron.model.distributed import DistributedDataParallel
from megatron.mpu import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank
)

@dataclass
class GPUInfo:
    index: int
    total_memory: int  # in bytes
    compute_capability: tuple
    name: str
    nvlink_connected_gpus: List[int]
    pcie_bandwidth: float  # GB/s

@dataclass
class SystemInfo:
    cpu_count: int
    total_memory: int  # in bytes
    gpu_count: int
    gpu_infos: List[GPUInfo]
    network_bandwidth: float  # GB/s
    interconnect_type: str  # 'NVLink' or 'PCIe'

class HardwareProfiler:
    def __init__(self):
        self.system_info = self._profile_system()
        self._memory_bandwidth_cache = {}
        self._compute_throughput_cache = {}

    def _profile_system(self) -> SystemInfo:
        """Profile the complete system hardware configuration."""
        gpu_infos = []
        gpu_count = torch.cuda.device_count()

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            nvlink_gpus = self._detect_nvlink_connections(i)
            pcie_bandwidth = self._measure_pcie_bandwidth(i)
            
            gpu_infos.append(GPUInfo(
                index=i,
                total_memory=props.total_memory,
                compute_capability=(props.major, props.minor),
                name=props.name,
                nvlink_connected_gpus=nvlink_gpus,
                pcie_bandwidth=pcie_bandwidth
            ))

        return SystemInfo(
            cpu_count=psutil.cpu_count(logical=True),
            total_memory=psutil.virtual_memory().total,
            gpu_count=gpu_count,
            gpu_infos=gpu_infos,
            network_bandwidth=self._measure_network_bandwidth(),
            interconnect_type=self._determine_interconnect_type(gpu_infos)
        )

    def _detect_nvlink_connections(self, gpu_index: int) -> List[int]:
        """Detect NVLink connections for a given GPU."""
        # This is a placeholder implementation
        # In practice, this would use NVIDIA Management Library (NVML) to detect actual NVLink connections
        connected_gpus = []
        return connected_gpus

    def _measure_pcie_bandwidth(self, gpu_index: int) -> float:
        """Measure PCIe bandwidth for a given GPU."""
        # Implement PCIe bandwidth measurement using peer-to-peer memory transfers
        # This is a placeholder implementation
        return 16.0  # GB/s typical for PCIe 3.0 x16

    def _measure_network_bandwidth(self) -> float:
        """Measure available network bandwidth between nodes."""
        # This is a placeholder implementation
        # In practice, this would perform actual network bandwidth measurements
        return 100.0  # GB/s typical for modern interconnects

    def _determine_interconnect_type(self, gpu_infos: List[GPUInfo]) -> str:
        """Determine the primary interconnect type being used."""
        has_nvlink = any(len(gpu.nvlink_connected_gpus) > 0 for gpu in gpu_infos)
        return 'NVLink' if has_nvlink else 'PCIe'

    def get_memory_bandwidth(self, gpu_index: int) -> float:
        """Measure memory bandwidth for a specific GPU."""
        if gpu_index not in self._memory_bandwidth_cache:
            self._memory_bandwidth_cache[gpu_index] = self._measure_memory_bandwidth(gpu_index)
        return self._memory_bandwidth_cache[gpu_index]

    def _measure_memory_bandwidth(self, gpu_index: int) -> float:
        """Perform actual memory bandwidth measurement."""
        # Implement memory bandwidth measurement using CUDA events and large memory transfers
        # This is a placeholder implementation
        return 900.0  # GB/s typical for modern GPUs

    def get_compute_throughput(self, gpu_index: int) -> float:
        """Measure compute throughput for a specific GPU."""
        if gpu_index not in self._compute_throughput_cache:
            self._compute_throughput_cache[gpu_index] = self._measure_compute_throughput(gpu_index)
        return self._compute_throughput_cache[gpu_index]

    def _measure_compute_throughput(self, gpu_index: int) -> float:
        """Perform actual compute throughput measurement."""
        # Implement compute throughput measurement using matrix multiplications
        # This is a placeholder implementation
        return 19.5  # TFLOPS typical for modern GPUs

    def get_optimal_device_mapping(self, num_gpus_needed: int) -> List[int]:
        """Determine optimal GPU mapping based on topology and bandwidth."""
        available_gpus = list(range(self.system_info.gpu_count))
        
        # Sort GPUs by their interconnect bandwidth and compute capability
        gpu_scores = []
        for gpu in self.system_info.gpu_infos:
            score = (
                len(gpu.nvlink_connected_gpus) * 1000 +  # Prioritize NVLink connections
                gpu.pcie_bandwidth +
                self.get_compute_throughput(gpu.index)
            )
            gpu_scores.append((gpu.index, score))
        
        gpu_scores.sort(key=lambda x: x[1], reverse=True)
        return [gpu[0] for gpu in gpu_scores[:num_gpus_needed]]

    def get_hardware_recommendation(self) -> Dict:
        """Generate hardware-aware recommendations for parallelism strategies."""
        gpu_memory = min(gpu.total_memory for gpu in self.system_info.gpu_infos)
        has_nvlink = self.system_info.interconnect_type == 'NVLink'
        
        return {
            'tensor_parallel_size': 2 if has_nvlink else 1,
            'pipeline_parallel_size': 2 if self.system_info.gpu_count >= 4 else 1,
            'data_parallel_size': max(1, self.system_info.gpu_count // 4),
            'optimal_batch_size': self._calculate_optimal_batch_size(gpu_memory),
            'activation_checkpointing': gpu_memory < 32 * (1024**3),  # Use if less than 32GB
            'recommended_precision': 'fp16' if gpu_memory < 40 * (1024**3) else 'fp32'
        }

    def _calculate_optimal_batch_size(self, gpu_memory: int) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        # This is a simplified calculation
        # In practice, this would consider model size, activation memory, and other factors
        base_batch_size = 32
        memory_factor = gpu_memory / (32 * (1024**3))  # Normalize to 32GB
        return max(1, int(base_batch_size * memory_factor))

    def print_system_info(self):
        """Print detailed system information."""
        print(f"System Configuration:")
        print(f"CPU Cores: {self.system_info.cpu_count}")
        print(f"Total System Memory: {self.system_info.total_memory / (1024**3):.2f} GB")
        print(f"GPU Count: {self.system_info.gpu_count}")
        print(f"Interconnect Type: {self.system_info.interconnect_type}")
        print(f"Network Bandwidth: {self.system_info.network_bandwidth:.2f} GB/s")
        print("\nGPU Details:")
        for gpu in self.system_info.gpu_infos:
            print(f"\nGPU {gpu.index}:")
            print(f"  Name: {gpu.name}")
            print(f"  Memory: {gpu.total_memory / (1024**3):.2f} GB")
            print(f"  Compute Capability: {gpu.compute_capability}")
            print(f"  NVLink Connected GPUs: {gpu.nvlink_connected_gpus}")
            print(f"  PCIe Bandwidth: {gpu.pcie_bandwidth:.2f} GB/s")

    def detect_gpu_capabilities(self) -> Dict:
        """Detect GPU capabilities and configurations."""
        if not torch.cuda.is_available():
            return {'error': 'No CUDA-capable GPUs detected'}
        
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info[f'gpu_{i}'] = {
                'name': props.name,
                'compute_capability': f'{props.major}.{props.minor}',
                'total_memory': props.total_memory,
                'memory_bandwidth': self._measure_memory_bandwidth(i),
                'tensor_cores': props.major >= 7,
                'multi_processor_count': props.multi_processor_count,
                'max_threads_per_block': props.max_threads_per_block,
                'max_shared_memory_per_block': props.max_shared_memory_per_block,
                'tensor_parallel_rank': get_tensor_model_parallel_rank(),
                'data_parallel_rank': get_data_parallel_rank(),
                'pipeline_parallel_rank': get_pipeline_model_parallel_rank()
            }
        
        return gpu_info

    def detect_network_topology(self) -> Dict:
        """Detect network interconnect topology and capabilities."""
        args = get_args()
        
        topology = {
            'tensor_parallel': {
                'world_size': get_tensor_model_parallel_world_size(),
                'group': str(get_tensor_model_parallel_group()),
                'rank': get_tensor_model_parallel_rank()
            },
            'data_parallel': {
                'world_size': get_data_parallel_world_size(),
                'group': str(get_data_parallel_group()),
                'rank': get_data_parallel_rank()
            },
            'pipeline_parallel': {
                'world_size': get_pipeline_model_parallel_world_size(),
                'group': str(get_pipeline_model_parallel_group()),
                'rank': get_pipeline_model_parallel_rank()
            }
        }
        
        # Measure network bandwidth between GPUs
        if torch.cuda.device_count() > 1:
            topology['inter_gpu_bandwidth'] = self._measure_gpu_interconnect()
        
        # Detect NCCL/GLOO backend and version
        topology['distributed_backend'] = {
            'nccl_version': torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else None,
            'gloo_available': hasattr(torch.distributed, 'GlooBackend'),
            'current_backend': torch.distributed.get_backend() 
                if torch.distributed.is_initialized() else None
        }
        
        return topology

    def _measure_gpu_interconnect(self, size_mb: int = 100) -> Dict:
        """Measure GPU interconnect bandwidth."""
        if not torch.cuda.is_available():
            return {}
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            return {}
        
        bandwidth_matrix = {}
        tensor_size = size_mb * 1024 * 1024 // 4  # Convert to number of float32
        
        for src in range(num_gpus):
            bandwidth_matrix[src] = {}
            for dst in range(num_gpus):
                if src != dst:
                    # Skip if GPUs are not in the same process group
                    if not self._are_gpus_in_same_group(src, dst):
                        continue
                    
                    with torch.cuda.device(src):
                        send_tensor = torch.randn(tensor_size, dtype=torch.float32, device=f'cuda:{src}')
                    
                    with torch.cuda.device(dst):
                        recv_tensor = torch.empty(tensor_size, dtype=torch.float32, device=f'cuda:{dst}')
                    
                    # Warmup
                    for _ in range(5):
                        recv_tensor.copy_(send_tensor)
                    
                    torch.cuda.synchronize(src)
                    torch.cuda.synchronize(dst)
                    
                    # Measure bandwidth
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    for _ in range(10):
                        recv_tensor.copy_(send_tensor)
                    end.record()
                    
                    torch.cuda.synchronize(src)
                    torch.cuda.synchronize(dst)
                    
                    elapsed_time = start.elapsed_time(end) / 1000  # Convert to seconds
                    bandwidth = (size_mb * 10) / elapsed_time  # MB/s
                    
                    bandwidth_matrix[src][dst] = bandwidth
        
        return bandwidth_matrix

    def _are_gpus_in_same_group(self, gpu1: int, gpu2: int) -> bool:
        """Check if two GPUs are in the same process group."""
        # Check tensor parallel group
        if (get_tensor_model_parallel_rank(gpu1) is not None and 
            get_tensor_model_parallel_rank(gpu2) is not None):
            return True
        
        # Check data parallel group
        if (get_data_parallel_rank(gpu1) is not None and 
            get_data_parallel_rank(gpu2) is not None):
            return True
        
        # Check pipeline parallel group
        if (get_pipeline_model_parallel_rank(gpu1) is not None and 
            get_pipeline_model_parallel_rank(gpu2) is not None):
            return True
        
        return False
