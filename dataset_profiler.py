"""
Dataset Profiler for Dynamic Distributed Training
Analyzes dataset characteristics and optimizes data loading strategies
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from dataclasses import dataclass
import time
import psutil
import threading
from queue import Queue
import logging
import json
from enum import Enum
import os
import io
import math
from collections import defaultdict

class DataType(Enum):
    """Types of data in the dataset."""
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    MIXED = "mixed"
    OTHER = "other"

@dataclass
class SampleStats:
    """Statistics for a single sample."""
    size_bytes: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    preprocessing_time: float  # milliseconds
    loading_time: float  # milliseconds

@dataclass
class DatasetStats:
    """Overall dataset statistics."""
    total_samples: int
    total_size_bytes: int
    avg_sample_size: float
    sample_size_std: float
    data_types: List[DataType]
    shape_distribution: Dict[Tuple[int, ...], int]
    dtype_distribution: Dict[torch.dtype, int]
    estimated_memory_usage: float
    estimated_preprocessing_time: float
    sample_loading_distribution: Dict[str, float]

@dataclass
class DataLoaderProfile:
    """DataLoader configuration profile."""
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    estimated_throughput: float
    memory_usage: float
    cpu_usage: float
    io_wait_time: float

class IOProfiler:
    """Profiles I/O operations and disk performance."""
    
    def __init__(self, 
                 monitoring_interval: float = 0.1,
                 history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        self.io_stats_history = []
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        self.last_disk_io = psutil.disk_io_counters()
        self.last_measure_time = time.time()
        
    def start_monitoring(self):
        """Start I/O monitoring."""
        if self.monitoring_thread is not None:
            return
            
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop I/O monitoring."""
        if self.monitoring_thread is None:
            return
            
        self.stop_monitoring.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                stats = self._collect_io_stats()
                self.io_stats_history.append(stats)
                
                if len(self.io_stats_history) > self.history_size:
                    self.io_stats_history.pop(0)
                    
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Error in I/O monitoring: {e}")
                
    def _collect_io_stats(self) -> Dict[str, float]:
        """Collect I/O statistics."""
        current_time = time.time()
        time_delta = current_time - self.last_measure_time
        
        disk_io = psutil.disk_io_counters()
        read_speed = (disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
        write_speed = (disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta
        
        self.last_disk_io = disk_io
        self.last_measure_time = current_time
        
        return {
            'read_speed': read_speed,
            'write_speed': write_speed,
            'read_count': disk_io.read_count,
            'write_count': disk_io.write_count,
            'read_time': disk_io.read_time,
            'write_time': disk_io.write_time
        }
        
    def get_io_stats(self) -> Dict[str, float]:
        """Get current I/O statistics."""
        if not self.io_stats_history:
            return {}
        return self.io_stats_history[-1]
        
    def get_average_throughput(self) -> Tuple[float, float]:
        """Get average read/write throughput."""
        if not self.io_stats_history:
            return (0.0, 0.0)
            
        avg_read = np.mean([stats['read_speed'] for stats in self.io_stats_history])
        avg_write = np.mean([stats['write_speed'] for stats in self.io_stats_history])
        return (avg_read, avg_write)

class DatasetProfiler:
    """Profiles dataset characteristics and optimizes data loading."""
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size_range: Tuple[int, int] = (1, 512),
                 num_workers_range: Tuple[int, int] = (0, 8),
                 sample_size: int = 1000,
                 profiling_device: str = 'cuda'):
        self.dataset = dataset
        self.batch_size_range = batch_size_range
        self.num_workers_range = num_workers_range
        self.sample_size = min(sample_size, len(dataset))
        self.profiling_device = profiling_device
        
        self.io_profiler = IOProfiler()
        self.sample_stats: List[SampleStats] = []
        self.dataloader_profiles: List[DataLoaderProfile] = []
        
        self.logger = logging.getLogger(__name__)
        
    def profile_dataset(self) -> DatasetStats:
        """Profile the dataset characteristics."""
        self.io_profiler.start_monitoring()
        self._collect_sample_stats()
        self.io_profiler.stop_monitoring()
        
        return DatasetStats(
            total_samples=len(self.dataset),
            total_size_bytes=self._calculate_total_size(),
            avg_sample_size=np.mean([stat.size_bytes for stat in self.sample_stats]),
            sample_size_std=np.std([stat.size_bytes for stat in self.sample_stats]),
            data_types=self._identify_data_types(),
            shape_distribution=self._get_shape_distribution(),
            dtype_distribution=self._get_dtype_distribution(),
            estimated_memory_usage=self._estimate_memory_usage(),
            estimated_preprocessing_time=self._estimate_preprocessing_time(),
            sample_loading_distribution=self._get_loading_distribution()
        )
        
    def optimize_dataloader(self) -> DataLoaderProfile:
        """Find optimal DataLoader configuration."""
        self._profile_dataloader_configs()
        return self._select_optimal_config()
        
    def _collect_sample_stats(self):
        """Collect statistics for a sample of the dataset."""
        indices = np.random.choice(len(self.dataset), self.sample_size, replace=False)
        
        for idx in indices:
            start_time = time.time()
            sample = self.dataset[idx]
            loading_time = (time.time() - start_time) * 1000  # Convert to ms
            
            start_time = time.time()
            if isinstance(sample, torch.Tensor):
                processed_sample = sample.to(self.profiling_device)
            elif isinstance(sample, (tuple, list)):
                processed_sample = tuple(
                    x.to(self.profiling_device) if isinstance(x, torch.Tensor) else x
                    for x in sample
                )
            else:
                processed_sample = sample
            preprocessing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            self.sample_stats.append(SampleStats(
                size_bytes=self._get_sample_size(sample),
                shape=self._get_sample_shape(sample),
                dtype=self._get_sample_dtype(sample),
                preprocessing_time=preprocessing_time,
                loading_time=loading_time
            ))
            
    def _get_sample_size(self, sample: Any) -> int:
        """Get size of a sample in bytes."""
        if isinstance(sample, torch.Tensor):
            return sample.element_size() * sample.numel()
        elif isinstance(sample, (tuple, list)):
            return sum(self._get_sample_size(x) for x in sample)
        elif isinstance(sample, np.ndarray):
            return sample.nbytes
        elif isinstance(sample, (str, bytes)):
            return len(sample)
        else:
            return 0
            
    def _get_sample_shape(self, sample: Any) -> Tuple[int, ...]:
        """Get shape of a sample."""
        if isinstance(sample, torch.Tensor):
            return tuple(sample.shape)
        elif isinstance(sample, np.ndarray):
            return tuple(sample.shape)
        elif isinstance(sample, (tuple, list)):
            return tuple(self._get_sample_shape(x) for x in sample)
        else:
            return tuple()
            
    def _get_sample_dtype(self, sample: Any) -> torch.dtype:
        """Get dtype of a sample."""
        if isinstance(sample, torch.Tensor):
            return sample.dtype
        elif isinstance(sample, np.ndarray):
            return torch.from_numpy(sample).dtype
        elif isinstance(sample, (tuple, list)):
            dtypes = [self._get_sample_dtype(x) for x in sample]
            return dtypes[0] if dtypes else torch.float32
        else:
            return torch.float32
            
    def _calculate_total_size(self) -> int:
        """Calculate total dataset size in bytes."""
        avg_size = np.mean([stat.size_bytes for stat in self.sample_stats])
        return int(avg_size * len(self.dataset))
        
    def _identify_data_types(self) -> List[DataType]:
        """Identify types of data in the dataset."""
        data_types = set()
        
        for stat in self.sample_stats:
            if len(stat.shape) == 4:  # Typical image/video shape (B, C, H, W)
                data_types.add(DataType.IMAGE)
            elif len(stat.shape) == 2:  # Typical text/tabular shape
                data_types.add(DataType.TEXT if stat.dtype == torch.long else DataType.TABULAR)
            elif len(stat.shape) == 3:  # Typical audio shape
                data_types.add(DataType.AUDIO)
                
        return list(data_types) if data_types else [DataType.OTHER]
        
    def _get_shape_distribution(self) -> Dict[Tuple[int, ...], int]:
        """Get distribution of sample shapes."""
        distribution = defaultdict(int)
        for stat in self.sample_stats:
            distribution[stat.shape] += 1
        return dict(distribution)
        
    def _get_dtype_distribution(self) -> Dict[torch.dtype, int]:
        """Get distribution of data types."""
        distribution = defaultdict(int)
        for stat in self.sample_stats:
            distribution[stat.dtype] += 1
        return dict(distribution)
        
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for the dataset."""
        avg_size = np.mean([stat.size_bytes for stat in self.sample_stats])
        return avg_size * self.batch_size_range[1]  # Estimate for largest batch size
        
    def _estimate_preprocessing_time(self) -> float:
        """Estimate preprocessing time per batch."""
        return np.mean([stat.preprocessing_time for stat in self.sample_stats])
        
    def _get_loading_distribution(self) -> Dict[str, float]:
        """Get distribution of sample loading times."""
        loading_times = [stat.loading_time for stat in self.sample_stats]
        return {
            'mean': np.mean(loading_times),
            'std': np.std(loading_times),
            'min': np.min(loading_times),
            'max': np.max(loading_times),
            'median': np.median(loading_times)
        }
        
    def _profile_dataloader_configs(self):
        """Profile different DataLoader configurations."""
        batch_sizes = np.logspace(
            np.log2(self.batch_size_range[0]),
            np.log2(self.batch_size_range[1]),
            num=5,
            base=2,
            dtype=int
        )
        
        num_workers_options = range(
            self.num_workers_range[0],
            self.num_workers_range[1] + 1
        )
        
        for batch_size in batch_sizes:
            for num_workers in num_workers_options:
                profile = self._profile_dataloader_config(batch_size, num_workers)
                self.dataloader_profiles.append(profile)
                
    def _profile_dataloader_config(self, batch_size: int, num_workers: int) -> DataLoaderProfile:
        """Profile a specific DataLoader configuration."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Measure throughput
        start_time = time.time()
        cpu_usage_start = psutil.cpu_percent(interval=None)
        
        num_batches = min(10, len(dataloader))
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.profiling_device)
            elif isinstance(batch, (tuple, list)):
                batch = tuple(x.to(self.profiling_device) if isinstance(x, torch.Tensor) else x
                            for x in batch)
                
        end_time = time.time()
        cpu_usage_end = psutil.cpu_percent(interval=None)
        
        throughput = num_batches * batch_size / (end_time - start_time)
        cpu_usage = (cpu_usage_end - cpu_usage_start) / 2  # Average CPU usage
        
        io_stats = self.io_profiler.get_io_stats()
        memory_info = psutil.Process().memory_info()
        
        return DataLoaderProfile(
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else 1,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False,
            estimated_throughput=throughput,
            memory_usage=memory_info.rss / 1024**2,  # MB
            cpu_usage=cpu_usage,
            io_wait_time=io_stats.get('read_time', 0) / 1000  # Convert to seconds
        )
        
    def _select_optimal_config(self) -> DataLoaderProfile:
        """Select the optimal DataLoader configuration."""
        if not self.dataloader_profiles:
            raise RuntimeError("No DataLoader profiles available")
            
        # Score each configuration based on multiple metrics
        scores = []
        for profile in self.dataloader_profiles:
            # Normalize metrics
            throughput_score = profile.estimated_throughput / max(p.estimated_throughput for p in self.dataloader_profiles)
            memory_score = 1 - (profile.memory_usage / max(p.memory_usage for p in self.dataloader_profiles))
            cpu_score = 1 - (profile.cpu_usage / max(p.cpu_usage for p in self.dataloader_profiles))
            io_score = 1 - (profile.io_wait_time / max(p.io_wait_time for p in self.dataloader_profiles))
            
            # Weighted sum of scores
            total_score = (
                0.4 * throughput_score +
                0.2 * memory_score +
                0.2 * cpu_score +
                0.2 * io_score
            )
            scores.append((total_score, profile))
            
        # Return profile with highest score
        return max(scores, key=lambda x: x[0])[1]
        
    def export_profile(self, filepath: str):
        """Export profiling results to a file."""
        profile_data = {
            'dataset_stats': {
                'total_samples': len(self.dataset),
                'total_size_bytes': self._calculate_total_size(),
                'data_types': [dt.value for dt in self._identify_data_types()],
                'shape_distribution': {str(k): v for k, v in self._get_shape_distribution().items()},
                'dtype_distribution': {str(k): v for k, v in self._get_dtype_distribution().items()},
                'loading_distribution': self._get_loading_distribution()
            },
            'dataloader_profiles': [
                {
                    'batch_size': profile.batch_size,
                    'num_workers': profile.num_workers,
                    'prefetch_factor': profile.prefetch_factor,
                    'pin_memory': profile.pin_memory,
                    'persistent_workers': profile.persistent_workers,
                    'estimated_throughput': profile.estimated_throughput,
                    'memory_usage': profile.memory_usage,
                    'cpu_usage': profile.cpu_usage,
                    'io_wait_time': profile.io_wait_time
                }
                for profile in self.dataloader_profiles
            ],
            'optimal_config': {
                'batch_size': self._select_optimal_config().batch_size,
                'num_workers': self._select_optimal_config().num_workers,
                'prefetch_factor': self._select_optimal_config().prefetch_factor,
                'pin_memory': self._select_optimal_config().pin_memory,
                'persistent_workers': self._select_optimal_config().persistent_workers
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
