"""
Dataset Profiler for Optimus-Megatron
Analyzes dataset characteristics for optimal batch size and data parallelism configuration.
"""

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import psutil
import time

# Megatron-specific imports
from megatron import get_args
from megatron import get_timers
from megatron.data.dataset_utils import get_train_valid_test_split_
from megatron.mpu import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_world_size
)

@dataclass
class DatasetStats:
    total_samples: int
    sample_shape: tuple
    memory_per_sample: int
    estimated_disk_throughput: float
    estimated_preprocessing_time: float
    sample_distribution: Dict[str, float]

@dataclass
class DataLoaderProfile:
    optimal_batch_size: int
    num_workers: int
    prefetch_factor: Optional[int]
    pin_memory: bool
    drop_last: bool
    estimated_throughput: float

class DatasetProfiler:
    def __init__(self):
        self.profile_cache = {}
        self.throughput_measurements = {}

    def profile_dataset(self, dataset: Dataset, 
                       sample_size: int = 1000) -> Tuple[DatasetStats, DataLoaderProfile]:
        """Profile the dataset characteristics and recommend optimal DataLoader configuration."""
        if id(dataset) in self.profile_cache:
            return self.profile_cache[id(dataset)]

        # Profile basic dataset statistics
        dataset_stats = self._analyze_dataset(dataset, sample_size)
        
        # Profile DataLoader configurations
        dataloader_profile = self._optimize_dataloader(dataset, dataset_stats)
        
        result = (dataset_stats, dataloader_profile)
        self.profile_cache[id(dataset)] = result
        return result

    def _analyze_dataset(self, dataset: Dataset, sample_size: int) -> DatasetStats:
        """Analyze dataset characteristics."""
        # Get sample shape and memory requirements
        sample_indices = np.random.choice(len(dataset), 
                                        min(sample_size, len(dataset)), 
                                        replace=False)
        
        # Measure memory per sample
        sample = dataset[sample_indices[0]]
        if isinstance(sample, torch.Tensor):
            sample_shape = tuple(sample.shape)
            memory_per_sample = sample.element_size() * sample.nelement()
        elif isinstance(sample, tuple):
            sample_shape = tuple(s.shape if isinstance(s, torch.Tensor) else None 
                               for s in sample)
            memory_per_sample = sum(s.element_size() * s.nelement() 
                                  if isinstance(s, torch.Tensor) else 0 
                                  for s in sample)
        else:
            raise ValueError(f"Unsupported sample type: {type(sample)}")

        # Measure disk throughput
        disk_throughput = self._measure_disk_throughput(dataset, sample_indices)
        
        # Measure preprocessing time
        preprocessing_time = self._measure_preprocessing_time(dataset, sample_indices)
        
        # Analyze sample distribution
        sample_distribution = self._analyze_sample_distribution(dataset, sample_indices)

        return DatasetStats(
            total_samples=len(dataset),
            sample_shape=sample_shape,
            memory_per_sample=memory_per_sample,
            estimated_disk_throughput=disk_throughput,
            estimated_preprocessing_time=preprocessing_time,
            sample_distribution=sample_distribution
        )

    def _measure_disk_throughput(self, dataset: Dataset, sample_indices: np.ndarray) -> float:
        """Measure dataset loading throughput from disk."""
        start_time = time.time()
        for idx in sample_indices[:100]:  # Measure first 100 samples
            _ = dataset[idx]
        end_time = time.time()
        
        total_bytes = len(sample_indices) * dataset[sample_indices[0]].element_size() * \
                     dataset[sample_indices[0]].nelement()
        return total_bytes / (end_time - start_time)  # bytes per second

    def _measure_preprocessing_time(self, dataset: Dataset, 
                                 sample_indices: np.ndarray) -> float:
        """Measure average preprocessing time per sample."""
        preprocessing_times = []
        for idx in sample_indices[:100]:  # Measure first 100 samples
            start_time = time.time()
            _ = dataset[idx]
            preprocessing_times.append(time.time() - start_time)
        return np.mean(preprocessing_times)

    def _analyze_sample_distribution(self, dataset: Dataset, 
                                  sample_indices: np.ndarray) -> Dict[str, float]:
        """Analyze distribution characteristics of the samples."""
        samples = [dataset[idx] for idx in sample_indices[:100]]
        if isinstance(samples[0], torch.Tensor):
            return {
                'mean': float(torch.stack(samples).mean()),
                'std': float(torch.stack(samples).std()),
                'sparsity': float((torch.stack(samples) == 0).float().mean())
            }
        return {}  # Return empty dict for non-tensor samples

    def _optimize_dataloader(self, dataset: Dataset, 
                           dataset_stats: DatasetStats) -> DataLoaderProfile:
        """Determine optimal DataLoader configuration."""
        args = get_args()
        
        # Calculate optimal batch size based on memory constraints and model parallel size
        available_memory = torch.cuda.get_device_properties(0).total_memory
        tensor_parallel_size = get_tensor_model_parallel_world_size()
        memory_per_batch = dataset_stats.memory_per_sample * 2  # Account for gradients
        
        # Adjust memory per batch for tensor parallelism
        memory_per_batch = memory_per_batch // tensor_parallel_size
        
        # Calculate max batch size (use 70% of available memory)
        max_batch_size = int(available_memory * 0.7 / memory_per_batch)
        
        # Adjust for data parallel training
        data_parallel_size = get_data_parallel_world_size()
        local_batch_size = max(1, max_batch_size // data_parallel_size)
        
        # Optimize number of workers based on CPU cores and memory
        cpu_count = psutil.cpu_count(logical=False)
        system_memory = psutil.virtual_memory()
        memory_per_worker = dataset_stats.memory_per_sample * local_batch_size
        max_workers_by_memory = int(system_memory.available * 0.5 / memory_per_worker)
        optimal_workers = min(cpu_count, max_workers_by_memory, 8)  # Cap at 8 workers
        
        # Determine prefetch factor based on preprocessing time
        prefetch_factor = self._calculate_prefetch_factor(dataset_stats)
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            shuffle=True,
            seed=args.seed,
            drop_last=True
        )
        
        # Create batch sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=local_batch_size,
            drop_last=True
        )
        
        # Configure DataLoader with optimal settings
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=optimal_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if optimal_workers > 0 else None,
            persistent_workers=optimal_workers > 0
        )
        
        # Measure throughput with optimal configuration
        throughput = self._measure_dataloader_throughput(dataloader)
        
        return DataLoaderProfile(
            optimal_batch_size=local_batch_size,
            num_workers=optimal_workers,
            prefetch_factor=prefetch_factor if optimal_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            estimated_throughput=throughput
        )

    def _calculate_prefetch_factor(self, dataset_stats: DatasetStats) -> Optional[int]:
        """Calculate optimal prefetch factor based on preprocessing time."""
        if dataset_stats.estimated_preprocessing_time is None:
            return 2  # Default value if preprocessing time unknown
        
        # Calculate how many samples can be preprocessed during one training step
        samples_per_second = 1.0 / dataset_stats.estimated_preprocessing_time
        training_time_per_batch = 0.1
        
        # Calculate prefetch factor (samples to preload)
        prefetch_factor = max(2, int(samples_per_second * training_time_per_batch))
        return min(prefetch_factor, 4)  # Cap at 4 to avoid excessive memory usage

    def _measure_dataloader_throughput(self, dataloader: DataLoader) -> float:
        """Measure throughput of a DataLoader configuration."""
        start_time = time.time()
        num_samples = 0
        
        # Measure throughput for a few batches
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Measure only first 10 batches
                break
            if isinstance(batch, torch.Tensor):
                num_samples += batch.size(0)
            elif isinstance(batch, tuple):
                num_samples += batch[0].size(0)
        
        elapsed_time = time.time() - start_time
        return num_samples / elapsed_time  # samples per second

    def get_data_parallel_recommendation(self, dataset_stats: DatasetStats, 
                                      dataloader_profile: DataLoaderProfile, 
                                      num_gpus: int) -> Dict:
        """Generate recommendations for data parallel training."""
        args = get_args()
        
        # Calculate memory requirements with tensor parallelism consideration
        tensor_parallel_size = get_tensor_model_parallel_world_size()
        total_memory_requirement = (dataset_stats.memory_per_sample * 
                                  dataloader_profile.optimal_batch_size // 
                                  tensor_parallel_size)
        
        # Calculate effective batch sizes
        data_parallel_size = get_data_parallel_world_size()
        global_batch_size = dataloader_profile.optimal_batch_size * data_parallel_size
        
        return {
            'global_batch_size': global_batch_size,
            'micro_batch_size': dataloader_profile.optimal_batch_size,
            'gradient_accumulation_steps': self._calculate_gradient_accumulation(
                dataset_stats, dataloader_profile, num_gpus
            ),
            'shuffle_strategy': self._recommend_shuffle_strategy(dataset_stats),
            'memory_optimization': {
                'pin_memory': dataloader_profile.pin_memory,
                'persistent_workers': dataset_stats.total_samples > 10000,
                'use_shared_memory': total_memory_requirement < psutil.virtual_memory().available
            }
        }

    def _calculate_gradient_accumulation(self, dataset_stats: DatasetStats,
                                      dataloader_profile: DataLoaderProfile,
                                      num_gpus: int) -> int:
        """Calculate optimal number of gradient accumulation steps."""
        target_global_batch = 1024  # Target global batch size
        actual_global_batch = dataloader_profile.optimal_batch_size * num_gpus
        
        if actual_global_batch >= target_global_batch:
            return 1
        
        return max(1, target_global_batch // actual_global_batch)

    def _recommend_shuffle_strategy(self, dataset_stats: DatasetStats) -> Dict:
        """Recommend data shuffling strategy based on dataset characteristics."""
        total_size = dataset_stats.total_samples
        
        if total_size < 10000:
            return {
                'shuffle': True,
                'distributed_shuffle': False,
                'memory_efficient': False
            }
        elif total_size < 1000000:
            return {
                'shuffle': True,
                'distributed_shuffle': True,
                'memory_efficient': False
            }
        else:
            return {
                'shuffle': True,
                'distributed_shuffle': True,
                'memory_efficient': True
            }

    def print_profile_summary(self, dataset_stats: DatasetStats, 
                            dataloader_profile: DataLoaderProfile):
        """Print a summary of the dataset and DataLoader profile."""
        print("\nDataset Profile Summary:")
        print(f"Total Samples: {dataset_stats.total_samples:,}")
        print(f"Sample Shape: {dataset_stats.sample_shape}")
        print(f"Memory per Sample: {dataset_stats.memory_per_sample / 1024:.2f} KB")
        print(f"Estimated Disk Throughput: {dataset_stats.estimated_disk_throughput / 1024 / 1024:.2f} MB/s")
        print(f"Average Preprocessing Time: {dataset_stats.estimated_preprocessing_time * 1000:.2f} ms")
        
        print("\nDataLoader Configuration:")
        print(f"Optimal Batch Size: {dataloader_profile.optimal_batch_size}")
        print(f"Number of Workers: {dataloader_profile.num_workers}")
        print(f"Prefetch Factor: {dataloader_profile.prefetch_factor}")
        print(f"Pin Memory: {dataloader_profile.pin_memory}")
        print(f"Estimated Throughput: {dataloader_profile.estimated_throughput:.2f} samples/s")
