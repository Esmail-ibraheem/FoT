"""
Hardware Profiler for Optimus-Megatron
Analyzes hardware resources and capabilities for optimal parallelization strategy.
"""

import torch
import torch.cuda as cuda
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
import psutil
import logging
import json
from dataclasses import dataclass
import time
from enum import Enum
import platform
from pathlib import Path

class InterconnectType(Enum):
    """Types of GPU interconnects."""
    NVLINK = "nvlink"
    PCIE = "pcie"
    INFINITY_FABRIC = "infinity_fabric"
    OTHER = "other"

@dataclass
class GPUInfo:
    """Detailed information about a single GPU."""
    index: int
    total_memory: int  # in bytes
    free_memory: int  # in bytes
    compute_capability: tuple
    name: str
    nvlink_connected_gpus: List[int]
    pcie_bandwidth: float  # GB/s
    memory_bandwidth: float  # GB/s
    compute_capability_features: Dict[str, bool]
    temperature: float  # in Celsius
    power_usage: float  # in Watts
    power_limit: float  # in Watts
    utilization: float  # percentage

@dataclass
class SystemInfo:
    """Comprehensive system hardware information."""
    cpu_count: int
    cpu_physical_cores: int
    cpu_frequency: float  # GHz
    total_memory: int  # in bytes
    free_memory: int  # in bytes
    gpu_count: int
    gpu_infos: List[GPUInfo]
    network_bandwidth: float  # GB/s
    interconnect_type: InterconnectType
    numa_nodes: int
    cpu_architecture: str
    os_info: Dict[str, str]
    memory_speed: float  # MHz
    swap_space: int  # bytes

class HardwareProfiler:
    """Analyzes and profiles hardware resources for optimal model parallelization."""
    
    def __init__(self):
        """Initialize the hardware profiler with logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.system_info = self._profile_system()
        self._memory_bandwidth_cache = {}
        self._compute_throughput_cache = {}
        self._topology_graph = self._build_topology_graph()
        self._parallel_groups = self._initialize_parallel_groups() if dist.is_initialized() else None

    def _initialize_parallel_groups(self):
        """Initialize parallel process groups using PyTorch distributed."""
        if not dist.is_initialized():
            return None
            
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Create default process group for data parallelism
        data_parallel_group = dist.new_group(ranks=list(range(world_size)))
        
        # For tensor parallelism, we'll create groups based on available GPUs
        gpu_count = torch.cuda.device_count()
        tensor_parallel_size = min(gpu_count, world_size)
        tensor_parallel_groups = []
        
        for i in range(world_size // tensor_parallel_size):
            ranks = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))
            group = dist.new_group(ranks=ranks)
            tensor_parallel_groups.append(group)
        
        return {
            'data_parallel': data_parallel_group,
            'tensor_parallel': tensor_parallel_groups[rank // tensor_parallel_size] if tensor_parallel_groups else None,
            'world_size': world_size,
            'rank': rank,
            'tensor_parallel_size': tensor_parallel_size
        }

    def _profile_system(self) -> SystemInfo:
        """Profile the complete system hardware configuration."""
        try:
            gpu_infos = []
            gpu_count = torch.cuda.device_count()

            for i in range(gpu_count):
                gpu_infos.append(self._profile_gpu(i))

            cpu_info = self._get_cpu_info()
            memory_info = self._get_memory_info()
            network_info = self._get_network_info()

            return SystemInfo(
                cpu_count=cpu_info["cpu_count"],
                cpu_physical_cores=cpu_info["physical_cores"],
                cpu_frequency=cpu_info["frequency"],
                total_memory=memory_info["total"],
                free_memory=memory_info["free"],
                gpu_count=gpu_count,
                gpu_infos=gpu_infos,
                network_bandwidth=network_info["bandwidth"],
                interconnect_type=self._determine_interconnect_type(gpu_infos),
                numa_nodes=self._get_numa_nodes(),
                cpu_architecture=cpu_info["architecture"],
                os_info=self._get_os_info(),
                memory_speed=memory_info["speed"],
                swap_space=memory_info["swap"]
            )
        except Exception as e:
            self.logger.error(f"Error during system profiling: {str(e)}")
            raise

    def _profile_gpu(self, gpu_index: int) -> GPUInfo:
        """Profile a specific GPU device comprehensively."""
        try:
            props = torch.cuda.get_device_properties(gpu_index)
            nvlink_gpus = self._detect_nvlink_connections(gpu_index)
            pcie_bandwidth = self._measure_pcie_bandwidth(gpu_index)
            memory_info = self._get_gpu_memory_info(gpu_index)
            
            return GPUInfo(
                index=gpu_index,
                total_memory=props.total_memory,
                free_memory=memory_info["free"],
                compute_capability=(props.major, props.minor),
                name=props.name,
                nvlink_connected_gpus=nvlink_gpus,
                pcie_bandwidth=pcie_bandwidth,
                memory_bandwidth=self._measure_memory_bandwidth(gpu_index),
                compute_capability_features=self._get_compute_features(props),
                temperature=self._get_gpu_temperature(gpu_index),
                power_usage=self._get_gpu_power_usage(gpu_index),
                power_limit=self._get_gpu_power_limit(gpu_index),
                utilization=self._get_gpu_utilization(gpu_index)
            )
        except Exception as e:
            self.logger.error(f"Error profiling GPU {gpu_index}: {str(e)}")
            raise

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

    def _measure_memory_bandwidth(self, gpu_index: int) -> float:
        """Measure memory bandwidth for a specific GPU."""
        # Implement memory bandwidth measurement using CUDA events and large memory transfers
        # This is a placeholder implementation
        return 900.0  # GB/s typical for modern GPUs

    def _get_compute_features(self, props) -> Dict[str, bool]:
        """Get compute features for a given GPU."""
        # This is a placeholder implementation
        # In practice, this would use NVIDIA Management Library (NVML) to detect actual compute features
        features = {
            "double_precision": True,
            "single_precision": True,
            "half_precision": True,
            "tensor_cores": True
        }
        return features

    def _get_gpu_temperature(self, gpu_index: int) -> float:
        """Get temperature for a given GPU."""
        # This is a placeholder implementation
        # In practice, this would use NVIDIA Management Library (NVML) to detect actual GPU temperature
        return 50.0  # Celsius

    def _get_gpu_power_usage(self, gpu_index: int) -> float:
        """Get power usage for a given GPU."""
        # This is a placeholder implementation
        # In practice, this would use NVIDIA Management Library (NVML) to detect actual GPU power usage
        return 250.0  # Watts

    def _get_gpu_power_limit(self, gpu_index: int) -> float:
        """Get power limit for a given GPU."""
        # This is a placeholder implementation
        # In practice, this would use NVIDIA Management Library (NVML) to detect actual GPU power limit
        return 300.0  # Watts

    def _get_gpu_utilization(self, gpu_index: int) -> float:
        """Get utilization for a given GPU."""
        # This is a placeholder implementation
        # In practice, this would use NVIDIA Management Library (NVML) to detect actual GPU utilization
        return 50.0  # percentage

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        cpu_info = {
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "frequency": psutil.cpu_freq().current / 1000,  # GHz
            "architecture": platform.machine()
        }
        return cpu_info

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        memory_info = {
            "total": psutil.virtual_memory().total,
            "free": psutil.virtual_memory().available,
            "speed": 3200  # MHz, typical for DDR4
        }
        return memory_info

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        network_info = {
            "bandwidth": 100.0  # GB/s, typical for modern interconnects
        }
        return network_info

    def _get_numa_nodes(self) -> int:
        """Get number of NUMA nodes."""
        return len(psutil.cpu_count(logical=False))

    def _get_os_info(self) -> Dict[str, str]:
        """Get OS information."""
        os_info = {
            "name": platform.system(),
            "version": platform.release(),
            "architecture": platform.machine()
        }
        return os_info

    def _determine_interconnect_type(self, gpu_infos: List[GPUInfo]) -> InterconnectType:
        """Determine the primary interconnect type being used."""
        has_nvlink = any(len(gpu.nvlink_connected_gpus) > 0 for gpu in gpu_infos)
        if has_nvlink:
            return InterconnectType.NVLINK
        else:
            return InterconnectType.PCIE

    def _build_topology_graph(self) -> Dict[str, Any]:
        """Build a graph representing the system topology."""
        # This is a placeholder implementation
        # In practice, this would use a graph library to build a graph representing the system topology
        topology_graph = {
            "nodes": [],
            "edges": []
        }
        return topology_graph

    def get_parallel_group_info(self) -> Dict[str, Any]:
        """Get information about parallel process groups."""
        if not self._parallel_groups:
            return {}
        
        return {
            'tensor_parallel': {
                'world_size': self._parallel_groups['world_size'],
                'rank': self._parallel_groups['rank']
            },
            'data_parallel': {
                'world_size': self._parallel_groups['world_size'],
                'rank': self._parallel_groups['rank']
            },
            'pipeline_parallel': {
                'world_size': self._parallel_groups['world_size'],
                'rank': self._parallel_groups['rank']
            }
        }

    def get_hardware_characteristics(self) -> Dict[str, Any]:
        """Get hardware characteristics for strategy selection."""
        return {
            'gpu_count': self.system_info.gpu_count,
            'gpu_memory': [gpu.total_memory for gpu in self.system_info.gpu_infos],
            'gpu_compute_capability': [gpu.compute_capability for gpu in self.system_info.gpu_infos],
            'interconnect': self.system_info.interconnect_type.value,
            'network_bandwidth': self.system_info.network_bandwidth,
            'memory_per_gpu': self.system_info.gpu_infos[0].total_memory if self.system_info.gpu_infos else 0,
            'nvlink_groups': [gpu.nvlink_connected_gpus for gpu in self.system_info.gpu_infos],
            'parallel_groups': self.get_parallel_group_info()
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current hardware metrics for monitoring."""
        metrics = {}
        try:
            metrics['gpu_memory_used'] = {
                i: self._get_gpu_memory_info(i)['used'] 
                for i in range(self.system_info.gpu_count)
            }
            metrics['gpu_utilization'] = {
                i: self._get_gpu_utilization(i) 
                for i in range(self.system_info.gpu_count)
            }
            metrics['temperature'] = {
                i: self._get_gpu_temperature(i) 
                for i in range(self.system_info.gpu_count)
            }
            metrics['power_usage'] = {
                i: self._get_gpu_power_usage(i) 
                for i in range(self.system_info.gpu_count)
            }
            
            if dist.is_initialized():
                metrics['communication_overhead'] = self._measure_communication_overhead()
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {str(e)}")
            return {}

    def _measure_communication_overhead(self) -> float:
        """Measure communication overhead between devices."""
        if not dist.is_initialized():
            return 0.0
        
        try:
            # Measure all-reduce time for a small tensor
            tensor = torch.randn(1024, 1024, device='cuda')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            dist.all_reduce(tensor)
            end.record()
            
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1000  # Convert to seconds
        except Exception as e:
            self.logger.error(f"Error measuring communication overhead: {str(e)}")
            return 0.0

    def export_profile(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export hardware profile to a JSON file and return as dictionary."""
        profile_data = {
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": self.system_info.cpu_count,
                "cpu_physical_cores": self.system_info.cpu_physical_cores,
                "cpu_frequency": self.system_info.cpu_frequency,
                "total_memory": self.system_info.total_memory,
                "free_memory": self.system_info.free_memory,
                "gpu_count": self.system_info.gpu_count,
                "network_bandwidth": self.system_info.network_bandwidth,
                "interconnect_type": self.system_info.interconnect_type.value,
                "numa_nodes": self.system_info.numa_nodes,
                "cpu_architecture": self.system_info.cpu_architecture,
                "os_info": self.system_info.os_info,
                "memory_speed": self.system_info.memory_speed,
                "swap_space": self.system_info.swap_space
            },
            "gpus": [self._gpu_info_to_dict(gpu) for gpu in self.system_info.gpu_infos],
            "topology": self._topology_to_dict(),
            "performance_metrics": self._get_performance_metrics()
        }

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                self.logger.info(f"Hardware profile exported to {filepath}")
            except Exception as e:
                self.logger.error(f"Error exporting profile to {filepath}: {str(e)}")
                raise

        return profile_data

    def _gpu_info_to_dict(self, gpu: GPUInfo) -> Dict[str, Any]:
        """Convert GPUInfo to a dictionary."""
        gpu_dict = {
            "index": gpu.index,
            "total_memory": gpu.total_memory,
            "free_memory": gpu.free_memory,
            "compute_capability": gpu.compute_capability,
            "name": gpu.name,
            "nvlink_connected_gpus": gpu.nvlink_connected_gpus,
            "pcie_bandwidth": gpu.pcie_bandwidth,
            "memory_bandwidth": gpu.memory_bandwidth,
            "compute_capability_features": gpu.compute_capability_features,
            "temperature": gpu.temperature,
            "power_usage": gpu.power_usage,
            "power_limit": gpu.power_limit,
            "utilization": gpu.utilization
        }
        return gpu_dict

    def _topology_to_dict(self) -> Dict[str, Any]:
        """Convert topology graph to a dictionary."""
        # This is a placeholder implementation
        # In practice, this would use a graph library to convert the topology graph to a dictionary
        topology_dict = {
            "nodes": [],
            "edges": []
        }
        return topology_dict

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        # This is a placeholder implementation
        # In practice, this would use a benchmarking library to get actual performance metrics
        performance_metrics = {
            "memory_bandwidth": 900.0,  # GB/s
            "compute_throughput": 19.5  # TFLOPS
        }
        return performance_metrics
