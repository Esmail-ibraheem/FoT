"""
Parallelism Manager for Optimus-Megatron with dynamic strategy adaptation
"""

import torch
import torch.distributed as dist
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

# Megatron-specific imports
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.model.distributed import DistributedDataParallel
from megatron.model.module import param_is_not_shared
from megatron.mpu import (
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_group,
    get_data_parallel_group,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_data_parallel_rank
)

from .strategy_selector import (
    DynamicStrategySelector,
    ParallelismConfig,
    ParallelismType,
    MonitoringMetrics
)

class ParallelismManager:
    """Manages parallelism strategies and transitions between them."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 checkpointing_dir: Optional[str] = None,
                 monitoring_interval: int = 100):
        """Initialize the parallelism manager."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointing_dir = checkpointing_dir
        
        # Get Megatron args
        self.args = get_args()
        self.timers = get_timers()
        
        # Initialize strategy selector
        self.strategy_selector = DynamicStrategySelector(
            monitoring_interval=monitoring_interval
        )
        
        # State variables
        self.is_initialized = False
        self.current_iteration = 0
        self._strategy_lock = threading.Lock()
        self.transition_in_progress = False
        
        # Initialize logger
        self.logger = logging.getLogger('MegatronParallelismManager')
        
        # Save initial configuration
        self.initial_config = ParallelismConfig(
            strategy_type=ParallelismType.DATA,
            data_parallel_size=self.args.data_parallel_size,
            tensor_parallel_size=self.args.tensor_model_parallel_size,
            pipeline_parallel_size=self.args.pipeline_model_parallel_size,
            micro_batch_size=self.args.micro_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            zero_optimization_stage=self.args.zero_optimization_stage,
            activation_checkpointing=self.args.activations_checkpoint_method is not None,
            cpu_offload=self.args.cpu_offload,
            communication_overlap=self.args.overlap_grad_reduce
        )
        
        self.strategy_selector.current_config = self.initial_config

    def _update_process_groups(self, config: ParallelismConfig):
        """Update process groups for new configuration."""
        # Save current model parameters
        old_state_dict = self._gather_model_state()
        
        # Destroy current process groups
        mpu.destroy_model_parallel()
        
        # Update args with new configuration
        self.args.data_parallel_size = config.data_parallel_size
        self.args.tensor_model_parallel_size = config.tensor_parallel_size
        self.args.pipeline_model_parallel_size = config.pipeline_parallel_size
        self.args.micro_batch_size = config.micro_batch_size
        self.args.gradient_accumulation_steps = config.gradient_accumulation_steps
        
        # Reinitialize process groups
        mpu.initialize_model_parallel(
            tensor_model_parallel_size_=config.tensor_parallel_size,
            pipeline_model_parallel_size_=config.pipeline_parallel_size,
            virtual_pipeline_model_parallel_size_=self.args.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=self.args.pipeline_model_parallel_split_rank
        )
        
        # Redistribute model state
        self._redistribute_model_state(old_state_dict)

    def _gather_model_state(self) -> Dict[str, torch.Tensor]:
        """Gather model state from all parallel groups."""
        state_dict = {}
        
        # Remove DDP wrapper if present
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        
        # Gather from tensor parallel group
        for name, param in model.named_parameters():
            if param_is_not_shared(param):
                # Gather across tensor parallel group
                gathered = [torch.zeros_like(param) for _ in range(self.args.tensor_model_parallel_size)]
                torch.distributed.all_gather(
                    gathered,
                    param.data,
                    group=get_tensor_model_parallel_group()
                )
                state_dict[name] = torch.cat(gathered, dim=0)
            else:
                state_dict[name] = param.data.clone()
        
        return state_dict

    def _redistribute_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Redistribute model state according to new configuration."""
        # Remove DDP wrapper if present
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        
        # Redistribute parameters
        for name, param in model.named_parameters():
            if param_is_not_shared(param):
                full_param = state_dict[name]
                
                # Calculate shard size for tensor parallelism
                shard_size = full_param.size(0) // self.args.tensor_model_parallel_size
                start_idx = get_tensor_model_parallel_rank() * shard_size
                end_idx = start_idx + shard_size
                
                # Update parameter
                param.data.copy_(full_param[start_idx:end_idx])
            else:
                param.data.copy_(state_dict[name])
        
        # Rewrap with DDP if needed
        if isinstance(self.model, DistributedDataParallel):
            self.model = DistributedDataParallel(
                model,
                process_group=get_data_parallel_group(),
                accumulate_allreduce_grads_in_fp32=self.args.accumulate_allreduce_grads_in_fp32,
                overlap_grad_reduce=self.args.overlap_grad_reduce,
                use_contiguous_buffers_in_ddp=self.args.use_contiguous_buffers_in_ddp
            )

    def _update_optimizer_state(self, config: ParallelismConfig):
        """Update optimizer state for new configuration."""
        # Get current optimizer state
        optimizer_state = self.optimizer.state_dict()
        
        # Create new optimizer with updated configuration
        self.optimizer = get_megatron_optimizer(
            model=self.model,
            args=self.args
        )
        
        # Update optimizer state
        # Note: This needs careful handling of parameter mapping
        # between old and new configurations
        try:
            self.optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            self.logger.warning(f"Could not restore optimizer state: {str(e)}")
            self.logger.info("Continuing with reinitialized optimizer")

    def _verify_process_groups(self, config: ParallelismConfig) -> bool:
        """Verify process group configuration."""
        try:
            # Verify group sizes
            if get_tensor_model_parallel_group().size() != config.tensor_parallel_size:
                return False
            if get_pipeline_model_parallel_group().size() != config.pipeline_parallel_size:
                return False
            if get_data_parallel_group().size() != config.data_parallel_size:
                return False
            
            # Verify connectivity
            for group in [get_tensor_model_parallel_group(),
                         get_pipeline_model_parallel_group(),
                         get_data_parallel_group()]:
                tensor = torch.ones(1, device=torch.cuda.current_device())
                dist.all_reduce(tensor, group=group)
                if tensor.item() != group.size():
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying process groups: {str(e)}")
            return False

    def _verify_model_distribution(self, config: ParallelismConfig) -> bool:
        """Verify model parameter distribution."""
        try:
            model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            
            # Verify parameter sharding
            for name, param in model.named_parameters():
                if param_is_not_shared(param):
                    # Verify parameter size matches configuration
                    expected_size = param.size(0) * get_tensor_model_parallel_rank()
                    if expected_size % config.tensor_parallel_size != 0:
                        self.logger.error(f"Parameter {name} not properly sharded")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying model distribution: {str(e)}")
            return False

    def _verify_optimizer_state(self, config: ParallelismConfig) -> bool:
        """Verify optimizer state."""
        try:
            # Verify optimizer parameters match model parameters
            model_params = set(self.model.parameters())
            optim_params = set()
            for group in self.optimizer.param_groups:
                optim_params.update(group['params'])
            
            if model_params != optim_params:
                self.logger.error("Optimizer parameters don't match model parameters")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying optimizer state: {str(e)}")
            return False

    def initialize(self):
        """Initialize the parallelism framework."""
        self.logger.info("Initializing Optimus-Megatron dynamic parallelism framework...")
        
        # Initialize parallelism strategy
        initial_strategy = self.strategy_selector.initialize_strategy(
            model_size=sum(p.numel() for p in self.model.parameters()),
            batch_size=self.args.micro_batch_size,
            num_gpus=torch.cuda.device_count(),
            gpu_memory=torch.cuda.get_device_properties(0).total_memory,
            network_bandwidth=psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        )
        
        # Apply initial strategy
        self._apply_parallelism_strategy(initial_strategy)
        
        # Start monitoring if dynamic adaptation is enabled
        if self.strategy_selector.enable_dynamic_adaptation:
            self.strategy_selector.start_monitoring()
        
        self.is_initialized = True
        self.logger.info("Parallelism framework initialization complete")
        
        # Save initial state if checkpointing is enabled
        if self.checkpointing_dir:
            self.save_state()

    def step(self, batch_metrics: Optional[Dict[str, float]] = None):
        """
        Process one training step and update metrics.
        
        Args:
            batch_metrics: Optional dictionary containing batch-specific metrics
        """
        if not self.is_initialized:
            raise RuntimeError("ParallelismManager must be initialized before stepping")
        
        self.current_iteration += 1
        
        # Collect and update metrics
        if batch_metrics:
            metrics = self._collect_step_metrics(batch_metrics)
            self.strategy_selector.update_metrics(metrics)
            
            # Store metrics for analysis
            self._update_performance_metrics(metrics)
            
            # Let the RL agent make decisions
            if (self.strategy_selector.enable_dynamic_adaptation and 
                time.time() - self.last_strategy_change_time > 300):  # 5 minutes minimum between changes
                self._check_and_adapt_strategy()

    def _collect_step_metrics(self, batch_metrics: Dict[str, float]) -> MonitoringMetrics:
        """Collect comprehensive training metrics for the current step."""
        gpu_memory = {}
        gpu_util = {}
        memory_reserved = {}
        
        # Collect GPU metrics
        for i in range(torch.cuda.device_count()):
            gpu_memory[i] = torch.cuda.memory_allocated(i) / 1024**3  # GB
            gpu_util[i] = torch.cuda.utilization(i)
            memory_reserved[i] = torch.cuda.memory_reserved(i) / 1024**3  # GB
            
        # Get communication metrics
        start_time = time.time()
        if dist.is_initialized():
            tensor = torch.randn(1024, device='cuda')
            dist.all_reduce(tensor)
        grad_sync_time = time.time() - start_time
        
        # Get pipeline metrics
        pipeline_stall = self._measure_pipeline_stalls()
        
        return MonitoringMetrics(
            timestamp=time.time(),
            iteration=self.current_iteration,
            throughput=batch_metrics.get('throughput', 0.0),
            gpu_memory_used=gpu_memory,
            gpu_utilization=gpu_util,
            communication_overhead=batch_metrics.get('communication_overhead', 0.0),
            pipeline_bubble_overhead=batch_metrics.get('pipeline_bubble', 0.0),
            load_imbalance=batch_metrics.get('load_imbalance', 0.0),
            tensor_parallel_efficiency=batch_metrics.get('tp_efficiency', 1.0),
            data_parallel_efficiency=batch_metrics.get('dp_efficiency', 1.0),
            gradient_sync_time=grad_sync_time,
            batch_processing_time=batch_metrics.get('batch_time', 0.0),
            pipeline_stall_time=pipeline_stall,
            memory_reserved=memory_reserved
        )
        
    def _measure_pipeline_stalls(self) -> float:
        """Measure pipeline stall time."""
        if not hasattr(self, 'last_forward_time'):
            self.last_forward_time = time.time()
            return 0.0
            
        current_time = time.time()
        stall_time = current_time - self.last_forward_time
        self.last_forward_time = current_time
        
        return max(0.0, stall_time - self.strategy_selector.monitoring_interval)

    def _apply_parallelism_strategy(self, config: ParallelismConfig) -> bool:
        """Apply a new parallelism strategy with proper synchronization and error handling."""
        try:
            with self._strategy_lock:
                self.logger.info(f"Applying new parallelism strategy: {config}")
                
                # Store current state for rollback
                old_config = self.strategy_selector.current_config
                
                # Phase 1: Validate configuration
                if not self._validate_config(config):
                    self.logger.error("Invalid parallelism configuration")
                    return False
                
                # Phase 2: Prepare for transition
                self._prepare_transition()
                
                # Phase 3: Synchronize all processes
                dist.barrier()
                
                try:
                    # Phase 4: Update process groups
                    self._update_process_groups(config)
                    
                    # Phase 5: Redistribute model
                    self._redistribute_model(config)
                    
                    # Phase 6: Update optimizer state
                    self._update_optimizer_state(config)
                    
                    # Phase 7: Verify transition
                    if not self._verify_transition(config):
                        raise RuntimeError("Strategy transition verification failed")
                    
                    # Phase 8: Commit changes
                    self.strategy_selector.current_config = config
                    dist.barrier()
                    
                    self.logger.info("Successfully applied new parallelism strategy")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error during strategy transition: {str(e)}")
                    # Rollback to previous state
                    self._rollback_to_config(old_config)
                    return False
                    
        except Exception as e:
            self.logger.error(f"Critical error in strategy application: {str(e)}")
            return False

    def _validate_config(self, config: ParallelismConfig) -> bool:
        """Validate parallelism configuration."""
        try:
            # Check basic requirements
            total_gpus = (config.data_parallel_size * 
                         config.tensor_parallel_size * 
                         config.pipeline_parallel_size)
            
            if total_gpus > torch.cuda.device_count():
                self.logger.error(f"Configuration requires {total_gpus} GPUs, but only {torch.cuda.device_count()} available")
                return False
            
            # Check power-of-2 requirements
            for size in [config.data_parallel_size, 
                        config.tensor_parallel_size, 
                        config.pipeline_parallel_size]:
                if size & (size - 1) != 0:  # Not power of 2
                    self.logger.error(f"Parallel size {size} is not a power of 2")
                    return False
            
            # Validate micro-batch size
            if config.micro_batch_size <= 0:
                self.logger.error("Micro-batch size must be positive")
                return False
            
            # Check memory requirements
            if not self._check_memory_requirements(config):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return False

    def _check_memory_requirements(self, config: ParallelismConfig) -> bool:
        """Check if the configuration meets memory requirements."""
        try:
            # Get model size and estimated activation memory
            model_size = sum(p.numel() for p in self.model.parameters())
            activation_size = model_size * 2  # Assume 2x model size for activations
            
            # Get available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate memory requirements per GPU
            memory_per_gpu = (model_size / config.tensor_parallel_size + 
                            activation_size / config.pipeline_parallel_size)
            
            # Add optimizer state if not using CPU offload
            if not config.cpu_offload:
                memory_per_gpu += model_size / config.data_parallel_size
            
            # Add buffer for temporary allocations
            memory_per_gpu *= 1.1  # 10% buffer
            
            if memory_per_gpu > gpu_memory:
                self.logger.error(f"Configuration requires {memory_per_gpu/1e9:.2f}GB per GPU, "
                           f"but only {gpu_memory/1e9:.2f}GB available")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking memory requirements: {str(e)}")
            return False

    def _prepare_transition(self):
        """Prepare for strategy transition."""
        # Synchronize outstanding operations
        torch.cuda.synchronize()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Ensure all processes are ready
        dist.barrier()

    def _redistribute_model(self, config: ParallelismConfig):
        """Redistribute model parameters according to new configuration."""
        # Implementation depends on model architecture
        pass

    def _verify_transition(self, config: ParallelismConfig) -> bool:
        """Verify that the transition was successful."""
        try:
            # Verify process groups
            if not self._verify_process_groups(config):
                return False
            
            # Verify model distribution
            if not self._verify_model_distribution(config):
                return False
            
            # Verify optimizer state
            if not self._verify_optimizer_state(config):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying transition: {str(e)}")
            return False

    def _rollback_to_config(self, config: ParallelismConfig):
        """Rollback to previous configuration."""
        self.logger.info("Rolling back to previous configuration")
        try:
            self._update_process_groups(config)
            self._redistribute_model(config)
            self._update_optimizer_state(config)
            self.strategy_selector.current_config = config
            dist.barrier()
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {str(e)}")
            # At this point, manual intervention may be required
            raise RuntimeError("Critical error during rollback")

    def _check_and_adapt_strategy(self):
        """Check if strategy adaptation is needed and perform it if necessary."""
        if self.transition_in_progress:
            return
            
        with self._strategy_lock:
            current_strategy = self.strategy_selector.get_current_strategy()
            if current_strategy != self.strategy_selector.current_config:
                self.transition_in_progress = True
                self._apply_parallelism_strategy(current_strategy)
                self.transition_in_progress = False

    def _update_performance_metrics(self, metrics: MonitoringMetrics):
        """Update stored performance metrics."""
        self.performance_metrics['throughput'].append(metrics.throughput)
        self.performance_metrics['gpu_utilization'].append(np.mean(list(metrics.gpu_utilization.values())))
        self.performance_metrics['memory_utilization'].append(np.mean(list(metrics.gpu_memory_used.values())))
        self.performance_metrics['communication_overhead'].append(metrics.communication_overhead)
        self.performance_metrics['pipeline_efficiency'].append(1.0 - metrics.pipeline_bubble_overhead)
        self.performance_metrics['tensor_parallel_efficiency'].append(metrics.tensor_parallel_efficiency)
        self.performance_metrics['data_parallel_efficiency'].append(metrics.data_parallel_efficiency)

    def save_state(self):
        """Save current parallelism state and metrics."""
        if not self.checkpointing_dir:
            return
            
        self.checkpointing_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current strategy
        strategy_path = self.checkpointing_dir / 'parallelism_strategy.json'
        with open(strategy_path, 'w') as f:
            json.dump(self.strategy_selector.current_config.__dict__, f, indent=2)
        
        # Save metrics history
        metrics_path = self.checkpointing_dir / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save strategy change history
        history_path = self.checkpointing_dir / 'strategy_history.json'
        self.strategy_selector.export_strategy_history(str(history_path))

    def load_state(self):
        """Load saved parallelism state and metrics."""
        if not self.checkpointing_dir:
            return
            
        strategy_path = self.checkpointing_dir / 'parallelism_strategy.json'
        if strategy_path.exists():
            with open(strategy_path, 'r') as f:
                strategy_dict = json.load(f)
                self.strategy_selector.current_config = ParallelismConfig(**strategy_dict)
        
        metrics_path = self.checkpointing_dir / 'performance_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.performance_metrics = json.load(f)

    def get_current_strategy(self) -> ParallelismConfig:
        """Get the current parallelism strategy."""
        return self.strategy_selector.get_current_strategy()

    def get_strategy_history(self) -> List[Tuple[int, ParallelismConfig]]:
        """Get the history of strategy changes."""
        return self.strategy_selector.get_strategy_history()

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """Get collected performance metrics."""
        return self.performance_metrics

    def print_status(self):
        """Print current status of the parallelism framework."""
        current_strategy = self.get_current_strategy()
        
        print("\nOptimus-Megatron Status:")
        print("------------------------")
        print(f"Current Iteration: {self.current_iteration}")
        print("\nActive Parallelism Strategy:")
        print(f"  Strategy Type: {current_strategy.strategy_type.value}")
        print(f"  Data Parallel Size: {current_strategy.data_parallel_size}")
        print(f"  Tensor Parallel Size: {current_strategy.tensor_parallel_size}")
        print(f"  Pipeline Parallel Size: {current_strategy.pipeline_parallel_size}")
        print(f"  Micro Batch Size: {current_strategy.micro_batch_size}")
        print(f"  Gradient Accumulation Steps: {current_strategy.gradient_accumulation_steps}")
        
        print("\nRecent Performance Metrics:")
        for metric, values in self.performance_metrics.items():
            if values:
                recent_avg = sum(values[-5:]) / min(5, len(values))
                print(f"  {metric.replace('_', ' ').title()}: {recent_avg:.2f}")
                
        if self.strategy_selector.strategy_changes:
            print("\nRecent Strategy Changes:")
            for iteration, config in self.strategy_selector.strategy_changes[-3:]:
                print(f"  Iteration {iteration}: {config.strategy_type.value}")

    def cleanup(self):
        """Clean up resources and save final state."""
        if self.strategy_selector.enable_dynamic_adaptation:
            self.strategy_selector.stop_monitoring()
        
        if self.checkpointing_dir:
            self.save_state()
            
        self.logger.info("Parallelism manager cleanup complete")
