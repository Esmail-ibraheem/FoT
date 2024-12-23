"""
Parallelism Manager for Dynamic Distributed Training
Pure PyTorch implementation for flexible model training parallelization
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import time
import threading
import logging
import psutil
from queue import Queue
from datetime import datetime
import json
import os
from communication_optimizer import CommunicationOptimizer
import subprocess
from collections import deque

from .strategy_selector import (
    DynamicStrategySelector,
    ParallelismConfig,
    ParallelismType,
    MonitoringMetrics
)

class ModelParallelModule(nn.Module):
    """Base class for model parallel modules."""
    
    def __init__(self, module: nn.Module, device_ids: List[int]):
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
    def _split_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split a tensor along its first dimension."""
        if tensor.size(0) < self.num_devices:
            raise ValueError(f"Batch size ({tensor.size(0)}) must be >= number of devices ({self.num_devices})")
        return torch.chunk(tensor, self.num_devices, dim=0)

class DataParallel(ModelParallelModule):
    """Enhanced DataParallel implementation with dynamic batch splitting."""
    
    def __init__(self, module: nn.Module, device_ids: List[int], output_device: Optional[int] = None):
        super().__init__(module, device_ids)
        self.output_device = output_device if output_device is not None else device_ids[0]
        
    def forward(self, *inputs, **kwargs):
        if not inputs and not kwargs:
            raise ValueError("No inputs provided")
            
        inputs = self._prepare_inputs(inputs)
        kwargs = self._prepare_kwargs(kwargs)
        
        replicas = self._replicate_module()
        outputs = self._parallel_forward(replicas, inputs, kwargs)
        return self._gather_outputs(outputs)
        
    def _prepare_inputs(self, inputs: Tuple) -> List[Tuple]:
        """Prepare inputs for parallel processing."""
        prepared = []
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                prepared.append(self._split_tensor(tensor))
            else:
                prepared.append([tensor] * self.num_devices)
        return list(zip(*prepared))
        
    def _prepare_kwargs(self, kwargs: Dict) -> List[Dict]:
        """Prepare kwargs for parallel processing."""
        prepared_kwargs = []
        for i in range(self.num_devices):
            device_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    splits = self._split_tensor(value)
                    device_kwargs[key] = splits[i]
                else:
                    device_kwargs[key] = value
            prepared_kwargs.append(device_kwargs)
        return prepared_kwargs
        
    def _replicate_module(self) -> List[nn.Module]:
        """Create module replicas on different devices."""
        return [self.module.to(torch.device(f'cuda:{device_id}'))
                for device_id in self.device_ids]
                
    def _parallel_forward(self, replicas: List[nn.Module], 
                         inputs: List[Tuple], 
                         kwargs: List[Dict]) -> List[torch.Tensor]:
        """Execute forward pass in parallel."""
        outputs = []
        for i, (module, input, kwarg) in enumerate(zip(replicas, inputs, kwargs)):
            with torch.cuda.device(self.device_ids[i]):
                output = module(*input, **kwarg)
                outputs.append(output)
        return outputs
        
    def _gather_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Gather outputs from all devices."""
        return torch.cat(outputs, dim=0).to(self.output_device)

class TensorParallel(ModelParallelModule):
    """Implementation of tensor parallelism for large models."""
    
    def __init__(self, module: nn.Module, device_ids: List[int]):
        super().__init__(module, device_ids)
        self.param_groups = self._split_parameters()
        
    def _split_parameters(self) -> List[Dict[str, nn.Parameter]]:
        """Split model parameters across devices."""
        param_size = sum(p.numel() for p in self.module.parameters())
        params_per_device = param_size // self.num_devices
        
        current_device = 0
        current_size = 0
        param_groups = [{} for _ in range(self.num_devices)]
        
        for name, param in self.module.named_parameters():
            param_size = param.numel()
            if current_size + param_size > params_per_device and current_device < self.num_devices - 1:
                current_device += 1
                current_size = 0
            
            param_groups[current_device][name] = param.to(f'cuda:{self.device_ids[current_device]}')
            current_size += param_size
            
        return param_groups
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism."""
        # Split input across devices
        splits = self._split_tensor(x)
        outputs = []
        
        # Process each split on its designated device
        for i, (split, params) in enumerate(zip(splits, self.param_groups)):
            with torch.cuda.device(self.device_ids[i]):
                device_output = self._device_forward(split, params)
                outputs.append(device_output)
                
        # Gather and combine results
        return self._combine_outputs(outputs)
        
    def _device_forward(self, x: torch.Tensor, params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Execute forward pass on a single device."""
        # Replace module parameters with device-specific ones
        with torch.no_grad():
            for name, param in params.items():
                module_param = self.module.get_parameter(name)
                module_param.data = param.data
                
        return self.module(x)
        
    def _combine_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Combine outputs from all devices."""
        return torch.cat(outputs, dim=1)  # Combine along feature dimension

class PipelineParallel(ModelParallelModule):
    """Implementation of pipeline parallelism with micro-batching."""
    
    def __init__(self, module: nn.Module, device_ids: List[int], chunks: int = 8):
        super().__init__(module, device_ids)
        self.chunks = chunks
        self.layers_per_device = len(list(module.children())) // self.num_devices
        self.pipeline_modules = self._create_pipeline_modules()
        
    def _create_pipeline_modules(self) -> List[nn.Module]:
        """Split model into pipeline stages."""
        layers = list(self.module.children())
        if len(layers) < self.num_devices:
            raise ValueError(f"Number of layers ({len(layers)}) must be >= number of devices ({self.num_devices})")
            
        pipeline_modules = []
        for i in range(self.num_devices):
            start_idx = i * self.layers_per_device
            end_idx = start_idx + self.layers_per_device
            stage = nn.Sequential(*layers[start_idx:end_idx]).to(f'cuda:{self.device_ids[i]}')
            pipeline_modules.append(stage)
            
        return pipeline_modules
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pipeline parallelism."""
        micro_batches = self._create_micro_batches(x)
        outputs = []
        
        # Pipeline schedule
        for i in range(len(micro_batches)):
            current_output = micro_batches[i]
            
            # Forward pass through pipeline stages
            for stage_idx, stage in enumerate(self.pipeline_modules):
                with torch.cuda.device(self.device_ids[stage_idx]):
                    current_output = stage(current_output.to(f'cuda:{self.device_ids[stage_idx]}'))
                    
            outputs.append(current_output)
            
        return self._combine_micro_batches(outputs)
        
    def _create_micro_batches(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split input into micro-batches."""
        batch_size = x.size(0)
        micro_batch_size = batch_size // self.chunks
        return torch.chunk(x, self.chunks, dim=0)
        
    def _combine_micro_batches(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Combine micro-batch outputs."""
        return torch.cat(outputs, dim=0)

class ParallelismManager:
    """Manages parallelism strategies and transitions between them."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 checkpointing_dir: Optional[str] = None,
                 monitoring_interval: int = 100):
        """Initialize the parallelism manager."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointing_dir = checkpointing_dir
        
        # Initialize strategy selector
        self.strategy_selector = DynamicStrategySelector(
            monitoring_interval=monitoring_interval
        )
        
        # Initialize communication optimizer
        self.comm_optimizer = CommunicationOptimizer(
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            monitoring_interval=monitoring_interval/1000.0,  # Convert to seconds
            enable_fusion=True,
            enable_overlap=True
        )
        
        # State variables
        self.is_initialized = False
        self.current_iteration = 0
        self._strategy_lock = threading.Lock()
        self.transition_in_progress = False
        
        # Initialize logger
        self.logger = logging.getLogger('ParallelismManager')
        
        # Available devices
        self.device_ids = list(range(torch.cuda.device_count()))
        if not self.device_ids:
            raise RuntimeError("No CUDA devices available")
            
        # Initialize distributed environment
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            
        # Save initial configuration
        self.initial_config = self._get_initial_config()
        self.current_parallel_model = None
        self.strategy_selector.current_config = self.initial_config
        
    def _get_initial_config(self) -> ParallelismConfig:
        """Get initial parallelism configuration based on model size and available resources."""
        total_params = sum(p.numel() for p in self.model.parameters())
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        if total_params > 1e9:  # Large model (>1B parameters)
            return ParallelismConfig(
                strategy_type=ParallelismType.PIPELINE,
                pipeline_parallel_size=len(self.device_ids),
                micro_batch_size=32,
                pipeline_chunks=8
            )
        elif total_params > 1e8:  # Medium model (>100M parameters)
            return ParallelismConfig(
                strategy_type=ParallelismType.TENSOR,
                tensor_parallel_size=len(self.device_ids),
                micro_batch_size=64
            )
        else:  # Small model
            return ParallelismConfig(
                strategy_type=ParallelismType.DATA,
                data_parallel_size=len(self.device_ids),
                micro_batch_size=128
            )
            
    def initialize_parallel_model(self):
        """Initialize the parallel model based on current configuration."""
        if self.is_initialized:
            return
            
        config = self.strategy_selector.current_config
        
        # Start communication monitoring
        self.comm_optimizer.start_monitoring()
        
        # Initialize model based on strategy
        if config.strategy_type == ParallelismType.DATA_PARALLEL:
            self.model = DataParallel(self.model, self.device_ids)
        elif config.strategy_type == ParallelismType.TENSOR_PARALLEL:
            self.model = TensorParallel(self.model, self.device_ids)
        elif config.strategy_type == ParallelismType.PIPELINE_PARALLEL:
            self.model = PipelineParallel(self.model, self.device_ids)
            
        self.is_initialized = True

    def train_step(self, batch: torch.Tensor):
        """Execute a training step with the current parallelism strategy."""
        # Optimize communication for gradient synchronization
        if isinstance(self.model, DataParallel):
            for param in self.model.parameters():
                if param.grad is not None:
                    self.comm_optimizer.optimize_all_reduce(param.grad)
                    
        # Forward pass
        output = self.model(batch)
        loss = self._compute_loss(output)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            
        # Update metrics and check for strategy adaptation
        self._update_metrics()
        
        return loss.item()

    def _update_metrics(self):
        """Update monitoring metrics and trigger strategy adaptation if needed."""
        metrics = {
            'throughput': self._calculate_throughput(),
            'gpu_memory': self._get_gpu_memory_usage(),
            'gpu_utilization': self._get_gpu_utilization(),
            'communication_overhead': self._measure_communication_overhead(),
            'pipeline_overhead': self._measure_pipeline_overhead(),
            'load_imbalance': self._measure_load_imbalance(),
            'tensor_parallel_efficiency': self._measure_tensor_parallel_efficiency(),
            'data_parallel_efficiency': self._measure_data_parallel_efficiency(),
            'gradient_sync_time': self._measure_gradient_sync_time()
        }
        
        # Update strategy selector with metrics
        self.strategy_selector.update_metrics(metrics)
        
        # Export communication stats periodically
        if self.current_iteration % 1000 == 0:
            stats_file = os.path.join(self.checkpointing_dir, f'comm_stats_{self.current_iteration}.json')
            self.comm_optimizer.export_stats(stats_file)
        
        self.current_iteration += 1

    def cleanup(self):
        """Clean up resources and stop monitoring."""
        if hasattr(self, 'comm_optimizer'):
            self.comm_optimizer.stop_monitoring()
        
    def save_checkpoint(self):
        """Save model checkpoint with current parallelism configuration."""
        if self.checkpointing_dir is None:
            return
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'iteration': self.current_iteration,
            'parallelism_config': self.strategy_selector.current_config,
            'metrics_history': self.strategy_selector.metrics_history,
            'strategy_changes': self.strategy_selector.strategy_changes
        }
        
        checkpoint_path = f"{self.checkpointing_dir}/checkpoint_{self.current_iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and restore parallelism configuration."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_iteration = checkpoint['iteration']
        self.strategy_selector.current_config = checkpoint['parallelism_config']
        self.strategy_selector.metrics_history = checkpoint['metrics_history']
        self.strategy_selector.strategy_changes = checkpoint['strategy_changes']
        
        self.initialize_parallel_model()

    def validate_configuration(self) -> bool:
        """Validate complete system configuration."""
        try:
            # Check hardware compatibility
            self._validate_device_mapping()
            
            # Check process group configuration
            if dist.is_initialized():
                world_size = dist.get_world_size()
                if world_size != self.strategy.total_parallel_size():
                    raise ValueError(f"Process world size {world_size} doesn't match strategy size {self.strategy.total_parallel_size()}")
            
            # Validate memory requirements
            available_memory = {device: torch.cuda.get_device_properties(device).total_memory
                              for device in range(torch.cuda.device_count())}
            required_memory = self._estimate_memory_requirements()
            
            for device, memory in available_memory.items():
                if required_memory > memory * 0.9:  # 90% threshold
                    raise ValueError(f"Insufficient memory on device {device}")
            
            # Validate communication patterns
            if hasattr(self, 'comm_optimizer'):
                self.comm_optimizer.validate_communication_pattern()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
            
    def _estimate_memory_requirements(self) -> int:
        """Estimate memory requirements for current configuration."""
        if not hasattr(self, 'model'):
            return 0
            
        total_params = sum(p.numel() * p.element_size() 
                          for p in self.model.parameters())
        
        # Account for optimizer states (e.g., Adam has 2 momentum buffers)
        optimizer_memory = total_params * 2 if hasattr(self, 'optimizer') else 0
        
        # Account for activations (rough estimate)
        batch_size = self.batch_size
        if hasattr(self.model, 'config'):
            sequence_length = getattr(self.model.config, 'max_position_embeddings', 512)
            hidden_size = getattr(self.model.config, 'hidden_size', 768)
            activation_memory = batch_size * sequence_length * hidden_size * 4  # float32
        else:
            activation_memory = total_params  # rough estimate
            
        # Add buffer for gradient accumulation if used
        gradient_memory = total_params if self.gradient_accumulation_steps > 1 else 0
        
        return total_params + optimizer_memory + activation_memory + gradient_memory

    def _calculate_throughput(self) -> float:
        """Calculate training throughput in samples/second."""
        if not hasattr(self, 'throughput_history'):
            self.throughput_history = deque(maxlen=100)  # Keep last 100 measurements
            
        # Get timing information
        current_time = time.time()
        if not hasattr(self, 'last_throughput_time'):
            self.last_throughput_time = current_time
            self.last_iteration = self.current_iteration
            return 0.0
            
        # Calculate time and iterations elapsed
        time_elapsed = current_time - self.last_throughput_time
        iterations_elapsed = self.current_iteration - self.last_iteration
        
        if time_elapsed < 1e-6:  # Avoid division by zero
            return 0.0
            
        # Calculate samples processed
        samples_processed = iterations_elapsed * self.batch_size
        if isinstance(self.model, PipelineParallel):
            samples_processed *= self.model.chunks  # Account for micro-batches
            
        # Calculate throughput
        current_throughput = samples_processed / time_elapsed
        
        # Update history
        self.throughput_history.append(current_throughput)
        
        # Update timing information for next calculation
        self.last_throughput_time = current_time
        self.last_iteration = self.current_iteration
        
        # Return moving average for stability
        return sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0.0

    def _get_gpu_memory_usage(self) -> Dict[int, float]:
        """Get GPU memory usage for each device."""
        memory_usage = {}
        for device_id in self.device_ids:
            memory_usage[device_id] = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
        return memory_usage
        
    def _get_gpu_utilization(self) -> Dict[int, float]:
        """Get GPU utilization for each device."""
        utilization = {}
        for device_id in self.device_ids:
            # Get GPU utilization using nvidia-smi
            try:
                result = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu',
                    '--format=csv,noheader,nounits',
                    '-i', str(device_id)
                ])
                utilization[device_id] = float(result.decode().strip())
            except (subprocess.CalledProcessError, ValueError):
                utilization[device_id] = 0.0
        return utilization

    def _measure_communication_overhead(self) -> float:
        """Measure communication overhead."""
        if not hasattr(self, 'last_comm_stats'):
            self.last_comm_stats = []
        
        # Get latest communication statistics
        current_stats = self.comm_optimizer.communication_stats
        
        # Calculate overhead from new stats only
        new_stats = current_stats[len(self.last_comm_stats):]
        self.last_comm_stats = current_stats
        
        if not new_stats:
            return 0.0
            
        # Sum up communication time
        total_comm_time = sum(stat.duration_ms for stat in new_stats)
        # Get total processing time
        total_time = self._measure_batch_processing_time() * 1000  # Convert to ms
        
        return (total_comm_time / total_time) * 100 if total_time > 0 else 0.0

    def _measure_pipeline_overhead(self) -> float:
        """Measure pipeline bubble overhead."""
        if not isinstance(self.model, PipelineParallel):
            return 0.0
            
        # Get pipeline statistics
        num_stages = len(self.model.pipeline_modules)
        micro_batch_size = self.model.chunks
        
        # Calculate theoretical vs actual time
        stage_times = [0.0] * num_stages
        for i, stage in enumerate(self.model.pipeline_modules):
            with torch.cuda.synchronize():
                start = time.time()
                # Run a sample forward pass
                sample_input = torch.randn(micro_batch_size, *self.input_shape[1:]).to(f'cuda:{self.device_ids[i]}')
                stage(sample_input)
                torch.cuda.synchronize()
                stage_times[i] = time.time() - start
        
        max_stage_time = max(stage_times)
        total_time = sum(stage_times)
        
        # Calculate bubble overhead
        theoretical_time = max_stage_time * num_stages
        actual_time = total_time + (micro_batch_size - 1) * max_stage_time
        
        return ((actual_time - theoretical_time) / theoretical_time) * 100

    def _measure_load_imbalance(self) -> float:
        """Measure load imbalance across devices."""
        device_loads = []
        for device_id in self.device_ids:
            # Get GPU compute time
            torch.cuda.synchronize(device_id)
            start = time.time()
            # Run a sample computation
            sample_tensor = torch.randn(1000, 1000).to(device_id)
            torch.matmul(sample_tensor, sample_tensor)
            torch.cuda.synchronize(device_id)
            compute_time = time.time() - start
            device_loads.append(compute_time)
        
        if not device_loads:
            return 0.0
            
        avg_load = sum(device_loads) / len(device_loads)
        max_load = max(device_loads)
        
        return ((max_load - avg_load) / avg_load) * 100 if avg_load > 0 else 0.0

    def _measure_tensor_parallel_efficiency(self) -> float:
        """Measure tensor parallel efficiency."""
        if not isinstance(self.model, TensorParallel):
            return 0.0
            
        # Measure computation vs communication time
        torch.cuda.synchronize()
        start = time.time()
        
        # Run a sample forward pass
        sample_input = torch.randn(32, *self.input_shape[1:]).to(self.device_ids[0])
        self.model(sample_input)
        
        torch.cuda.synchronize()
        total_time = time.time() - start
        
        # Get communication time from optimizer
        comm_time = sum(stat.duration_ms for stat in self.comm_optimizer.communication_stats[-10:]) / 1000
        
        compute_time = total_time - comm_time
        return (compute_time / total_time) * 100 if total_time > 0 else 0.0

    def _measure_data_parallel_efficiency(self) -> float:
        """Measure data parallel efficiency."""
        if not isinstance(self.model, DataParallel):
            return 0.0
            
        # Measure computation vs synchronization time
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Run a sample forward-backward pass
        sample_input = torch.randn(32, *self.input_shape[1:]).to(self.device_ids[0])
        output = self.model(sample_input)
        loss = self._compute_loss(output)
        loss.backward()
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Get gradient synchronization time
        sync_time = self._measure_gradient_sync_time()
        
        compute_time = total_time - sync_time
        return (compute_time / total_time) * 100 if total_time > 0 else 0.0

    def _measure_gradient_sync_time(self) -> float:
        """Measure gradient synchronization time."""
        if not isinstance(self.model, (DataParallel, TensorParallel)):
            return 0.0
            
        sync_times = []
        for param in self.model.parameters():
            if param.grad is not None:
                torch.cuda.synchronize()
                start = time.time()
                
                # Perform gradient synchronization
                dist.all_reduce(param.grad)
                
                torch.cuda.synchronize()
                sync_times.append(time.time() - start)
        
        return sum(sync_times)

    def _measure_batch_processing_time(self) -> float:
        """Measure batch processing time."""
        torch.cuda.synchronize()
        start = time.time()
        
        # Process a sample batch
        sample_input = torch.randn(32, *self.input_shape[1:]).to(self.device_ids[0])
        output = self.model(sample_input)
        loss = self._compute_loss(output)
        loss.backward()
        
        torch.cuda.synchronize()
        return time.time() - start

    def _measure_pipeline_stall_time(self) -> float:
        """Measure pipeline stall time."""
        if not isinstance(self.model, PipelineParallel):
            return 0.0
            
        total_stall_time = 0.0
        num_stages = len(self.model.pipeline_modules)
        
        for i in range(num_stages):
            torch.cuda.synchronize(self.device_ids[i])
            start = time.time()
            
            # Wait for input from previous stage (if not first stage)
            if i > 0:
                self.model._wait_for_input(i)
                
            # Process stage
            sample_input = torch.randn(self.model.chunks, *self.input_shape[1:]).to(self.device_ids[i])
            self.model.pipeline_modules[i](sample_input)
            
            # Wait for output consumption (if not last stage)
            if i < num_stages - 1:
                self.model._wait_for_output_consumption(i)
                
            torch.cuda.synchronize(self.device_ids[i])
            stage_time = time.time() - start
            
            # Stall time is the difference between actual and compute time
            compute_time = self._get_stage_compute_time(i)
            total_stall_time += stage_time - compute_time
            
        return total_stall_time

    def _get_stage_compute_time(self, stage_idx: int) -> float:
        """Helper method to get pure computation time for a pipeline stage."""
        torch.cuda.synchronize(self.device_ids[stage_idx])
        start = time.time()
        
        # Run computation without pipeline overhead
        sample_input = torch.randn(self.model.chunks, *self.input_shape[1:]).to(self.device_ids[stage_idx])
        self.model.pipeline_modules[stage_idx](sample_input)
        
        torch.cuda.synchronize(self.device_ids[stage_idx])
        return time.time() - start

    def _compute_loss(self, output: torch.Tensor) -> torch.Tensor:
        """Compute loss for the current batch."""
        if hasattr(self, 'criterion'):
            return self.criterion(output)
        return output.mean()  # Default loss for testing

    def _get_memory_reserved(self) -> Dict[int, float]:
        """Get reserved memory for each device."""
        memory_reserved = {}
        for device_id in self.device_ids:
            memory_reserved[device_id] = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
        return memory_reserved
