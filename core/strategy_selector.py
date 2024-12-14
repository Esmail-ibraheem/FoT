""" 
Strategy Selector for Optimus-Megatron with AI-driven decision making
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from dataclasses import dataclass
import time
from enum import Enum
import threading
import logging
import json
import psutil
from queue import Queue
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Megatron-specific imports
from megatron import get_args
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MegatronStrategySelector')

class Experience(NamedTuple):
    """Experience tuple for reinforcement learning."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool

@dataclass
class MonitoringMetrics:
    """Metrics tracked for dynamic strategy adaptation."""
    timestamp: float
    iteration: int
    throughput: float
    gpu_memory_used: Dict[int, float]
    gpu_utilization: Dict[int, float]
    communication_overhead: float
    pipeline_bubble_overhead: float
    load_imbalance: float
    tensor_parallel_efficiency: float
    data_parallel_efficiency: float
    gradient_sync_time: float
    batch_processing_time: float
    pipeline_stall_time: float
    memory_reserved: Dict[int, float]
    
class RLAgent(nn.Module):
    """Reinforcement Learning agent for parallelism strategy selection."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
        
    def select_action(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.network[-1].out_features)
        
        with torch.no_grad():
            return self.forward(state).argmax().item()
            
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        current_q = self.forward(states).gather(1, actions.unsqueeze(1))
        next_q = self.forward(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def add_experience(self, exp: Experience):
        self.memory.append(exp)

class ParallelismType(Enum):
    """Types of parallelism supported."""
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    SEQUENCE = "sequence"
    EXPERT = "expert"
    ZERO = "zero"

class ModelArchitecture(Enum):
    """Supported model architectures for specialized strategies."""
    TRANSFORMER = "transformer"
    MLP = "mlp"
    CNN = "cnn"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

@dataclass
class ModelCharacteristics:
    """Characteristics of the model that influence parallelism decisions."""
    architecture: ModelArchitecture
    num_layers: int
    hidden_size: int
    num_attention_heads: Optional[int]
    sequence_length: Optional[int]
    vocab_size: Optional[int]
    has_pipeline_stages: bool
    parameter_size: int  # in billions
    activation_size: int  # in GB
    peak_memory: float  # in GB
    compute_intensity: float  # FLOPs/byte

@dataclass
class HardwareCharacteristics:
    """Hardware characteristics that influence parallelism decisions."""
    num_gpus: int
    gpu_memory: int  # in GB
    gpu_flops: float  # in TFLOPS
    network_bandwidth: float  # in GB/s
    network_latency: float  # in microseconds
    nvlink_available: bool
    gpu_topology: Dict[str, Any]

@dataclass
class ParallelismConfig:
    """Parallelism configuration."""
    strategy_type: ParallelismType
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    zero_optimization_stage: int
    activation_checkpointing: bool
    cpu_offload: bool
    communication_overlap: bool

class StrategyMonitor:
    """Monitors training metrics and system resources for strategy adaptation."""
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 metrics_window_size: int = 10,
                 adaptation_threshold: float = 0.15):
        self.monitoring_interval = monitoring_interval
        self.metrics_window_size = metrics_window_size
        self.adaptation_threshold = adaptation_threshold
        
        # Initialize RL agent
        self.state_dim = 9  # Number of metrics we track
        self.action_dim = 5  # Number of possible parallelism configurations
        self.rl_agent = RLAgent(self.state_dim, self.action_dim)
        
        self.metrics_queue = Queue(maxsize=metrics_window_size)
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        self.current_metrics = None
        self.baseline_throughput = None
        self.strategy_change_timestamps = []
        
        # Initialize GPU monitoring
        self.num_gpus = torch.cuda.device_count()
        self.max_throughput = 1000.0  # Example max throughput
        self.max_comm_overhead = 0.5  # Example max communication overhead
        self.max_pipeline_stalls = 0.2  # Example max pipeline stalls
        
        # Performance history
        self.performance_history = []
        
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_thread is not None:
            logger.warning("Monitoring thread already running")
            return
            
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started performance monitoring with AI-driven adaptation")
        
    def _collect_metrics(self) -> MonitoringMetrics:
        """Collect comprehensive performance and resource metrics."""
        # Get GPU metrics
        gpu_memory = {}
        gpu_util = {}
        memory_reserved = {}
        for i in range(self.num_gpus):
            gpu_memory[i] = torch.cuda.memory_allocated(i) / 1024**3  # GB
            gpu_util[i] = torch.cuda.utilization(i)
            memory_reserved[i] = torch.cuda.memory_reserved(i) / 1024**3  # GB
        
        # Measure batch processing time
        start_time = time.time()
        # Your batch processing code here
        batch_time = time.time() - start_time
        
        # Measure gradient sync time
        start_time = time.time()
        if dist.is_initialized():
            tensor = torch.randn(1024, device='cuda')
            dist.all_reduce(tensor)
        grad_sync_time = time.time() - start_time
        
        # Get pipeline stall time
        pipeline_stall = self._measure_pipeline_stalls()
        
        # Other metrics
        comm_overhead = self._measure_communication_overhead()
        tp_efficiency = self._measure_tensor_parallel_efficiency()
        dp_efficiency = self._measure_data_parallel_efficiency()
        pp_bubble = self._measure_pipeline_bubble_overhead()
        load_imbalance = self._calculate_load_imbalance(gpu_util)
        
        return MonitoringMetrics(
            timestamp=time.time(),
            iteration=self._get_current_iteration(),
            throughput=self._measure_throughput(),
            gpu_memory_used=gpu_memory,
            gpu_utilization=gpu_util,
            communication_overhead=comm_overhead,
            pipeline_bubble_overhead=pp_bubble,
            load_imbalance=load_imbalance,
            tensor_parallel_efficiency=tp_efficiency,
            data_parallel_efficiency=dp_efficiency,
            gradient_sync_time=grad_sync_time,
            batch_processing_time=batch_time,
            pipeline_stall_time=pipeline_stall,
            memory_reserved=memory_reserved
        )
        
    def _measure_pipeline_stalls(self) -> float:
        """Measure time spent in pipeline stalls."""
        try:
            if not hasattr(self, 'last_forward_time'):
                self.last_forward_time = time.time()
                return 0.0
                
            current_time = time.time()
            stall_time = current_time - self.last_forward_time
            self.last_forward_time = current_time
            
            return max(0.0, stall_time - self.monitoring_interval)
        except Exception as e:
            logger.warning(f"Error measuring pipeline stalls: {str(e)}")
            return 0.0
            
    def _get_state_representation(self) -> np.ndarray:
        """Convert current metrics to state vector for RL agent."""
        if not self.metrics_queue.qsize():
            return np.zeros(self.state_dim)
        
        recent_metrics = list(self.metrics_queue.queue)
        
        # Compute normalized metrics
        throughput = np.mean([m.throughput for m in recent_metrics])
        gpu_mem = np.mean([np.mean(list(m.gpu_memory_used.values())) for m in recent_metrics])
        gpu_util = np.mean([np.mean(list(m.gpu_utilization.values())) for m in recent_metrics])
        comm_overhead = np.mean([m.communication_overhead for m in recent_metrics])
        pipeline_stalls = np.mean([m.pipeline_stall_time for m in recent_metrics])
        
        # Add trend indicators
        throughput_trend = self._calculate_trend([m.throughput for m in recent_metrics])
        memory_trend = self._calculate_trend([np.mean(list(m.gpu_memory_used.values())) for m in recent_metrics])
        
        # Compute efficiency metrics
        tp_efficiency = np.mean([m.tensor_parallel_efficiency for m in recent_metrics])
        dp_efficiency = np.mean([m.data_parallel_efficiency for m in recent_metrics])
        
        state = np.array([
            throughput / self.max_throughput,  # Normalized throughput
            gpu_mem / 100.0,  # GPU memory utilization (0-1)
            gpu_util / 100.0,  # GPU compute utilization (0-1)
            comm_overhead / self.max_comm_overhead,  # Normalized communication overhead
            pipeline_stalls / self.max_pipeline_stalls,  # Normalized pipeline stalls
            throughput_trend,  # Trend indicator (-1 to 1)
            memory_trend,  # Trend indicator (-1 to 1)
            tp_efficiency,  # Tensor parallel efficiency (0-1)
            dp_efficiency,  # Data parallel efficiency (0-1)
        ])
        
        return np.clip(state, -1.0, 1.0)  # Ensure all values are in [-1, 1]

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend indicator in range [-1, 1]."""
        if len(values) < 2:
            return 0.0
        
        # Use linear regression slope as trend indicator
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Normalize slope to [-1, 1] range
        max_slope = abs(max(values) - min(values)) / len(values)
        normalized_slope = np.clip(slope / max_slope if max_slope > 0 else 0.0, -1.0, 1.0)
        
        return normalized_slope

    def _calculate_reward(self, current_performance: Dict[str, float]) -> float:
        """Calculate reward based on performance metrics and costs."""
        # Configuration weights (should be moved to config file)
        weights = {
            'throughput': 0.4,
            'memory_efficiency': 0.2,
            'communication_efficiency': 0.15,
            'load_balance': 0.15,
            'convergence': 0.1
        }
        
        # Validate weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        # Calculate base reward
        reward = sum(weights[k] * v for k, v in current_performance.items())
        
        # Apply penalties
        if self.current_config:
            # Penalize excessive parallelism
            total_parallel_size = (self.current_config.data_parallel_size * 
                                 self.current_config.tensor_parallel_size * 
                                 self.current_config.pipeline_parallel_size)
            if total_parallel_size > self.num_gpus:
                reward *= 0.5  # Significant penalty for invalid configurations
            
            # Penalize frequent changes
            if len(self.strategy_changes) > 1:
                last_change_time = self.strategy_changes[-1][0]
                if time.time() - last_change_time < self.min_adaptation_interval:
                    reward *= 0.8  # Penalty for too frequent changes
        
        return reward

    def _should_adapt(self, current_performance: Dict[str, float]) -> bool:
        """Determine if strategy adaptation is needed."""
        if not self.enable_dynamic_adaptation:
            return False
            
        # Check cooldown period
        if self.strategy_changes:
            last_change_time = self.strategy_changes[-1][0]
            if time.time() - last_change_time < self.min_adaptation_interval:
                return False
        
        # Check if current performance is significantly degraded
        if len(self.metrics_queue.qsize()) < self.metrics_window_size:
            return False
            
        recent_throughput = [m.throughput for m in list(self.metrics_queue.queue)[-self.metrics_window_size:]]
        avg_throughput = np.mean(recent_throughput)
        current_throughput = recent_throughput[-1]
        
        # Calculate performance degradation
        degradation = (avg_throughput - current_throughput) / avg_throughput if avg_throughput > 0 else 0
        
        # Check resource utilization
        recent_metrics = list(self.metrics_queue.queue)[-self.metrics_window_size:]
        avg_gpu_util = np.mean([np.mean(list(m.gpu_utilization.values())) for m in recent_metrics])
        avg_memory_util = np.mean([np.mean(list(m.gpu_memory_used.values())) for m in recent_metrics])
        
        # Adaptation triggers
        triggers = {
            'performance_degraded': degradation > 0.15,  # 15% degradation
            'low_gpu_utilization': avg_gpu_util < 50.0,  # Below 50% GPU utilization
            'high_memory_pressure': avg_memory_util > 90.0,  # Above 90% memory usage
            'high_communication': np.mean([m.communication_overhead for m in recent_metrics]) > 0.3,  # 30% comm overhead
            'pipeline_inefficiency': np.mean([m.pipeline_stall_time for m in recent_metrics]) > 0.2  # 20% pipeline stalls
        }
        
        # Require at least two triggers for adaptation
        return sum(triggers.values()) >= 2

    def _analyze_metrics(self):
        """Analyze metrics and use RL agent for strategy adaptation."""
        if self.metrics_queue.qsize() < 2:  # Need at least 2 measurements
            return
            
        metrics_list = list(self.metrics_queue.queue)
        current_metrics = metrics_list[-1]
        previous_metrics = metrics_list[-2]
        
        # Convert metrics to state tensor
        current_state = self._get_state_representation()
        previous_state = self._get_state_representation()
        
        # Get action from RL agent
        action = self.rl_agent.select_action(torch.tensor(current_state, dtype=torch.float32))
        
        # Calculate reward
        reward = self._calculate_reward({
            'throughput': current_metrics.throughput,
            'memory_efficiency': 1.0 - np.mean(list(current_metrics.gpu_memory_used.values())),
            'communication_efficiency': 1.0 - current_metrics.communication_overhead,
            'load_balance': 1.0 - current_metrics.load_imbalance,
            'convergence': current_metrics.tensor_parallel_efficiency
        })
        
        # Add experience to agent's memory
        exp = Experience(
            torch.tensor(previous_state, dtype=torch.float32),
            action,
            reward,
            torch.tensor(current_state, dtype=torch.float32),
            False
        )
        self.rl_agent.add_experience(exp)
        
        # Train the agent
        self.rl_agent.train_step()
        
        # Log performance
        self.performance_history.append({
            'timestamp': current_metrics.timestamp,
            'metrics': self._metrics_to_dict(current_metrics),
            'action': action,
            'reward': reward
        })
        
        # Trigger strategy adaptation if needed
        if self._should_adapt({
            'throughput': current_metrics.throughput,
            'memory_efficiency': 1.0 - np.mean(list(current_metrics.gpu_memory_used.values())),
            'communication_efficiency': 1.0 - current_metrics.communication_overhead,
            'load_balance': 1.0 - current_metrics.load_imbalance,
            'convergence': current_metrics.tensor_parallel_efficiency
        }):
            self._trigger_strategy_adaptation(action)
            
    def _trigger_strategy_adaptation(self, action: int):
        """Apply the new strategy selected by the RL agent."""
        logger.info(f"Adapting parallelism strategy to configuration {action}")
        self.strategy_change_timestamps.append(time.time())
        
        # Convert action to strategy configuration
        strategy = self._action_to_strategy(action)
        
        # Apply the new strategy
        self._apply_strategy(strategy)
        
    def _action_to_strategy(self, action: int) -> Dict:
        """Convert RL action to concrete strategy configuration."""
        # Example strategy mappings
        strategies = {
            0: {'dp': 2, 'tp': 2, 'pp': 1},  # Balanced DP and TP
            1: {'dp': 4, 'tp': 1, 'pp': 1},  # Heavy DP
            2: {'dp': 1, 'tp': 4, 'pp': 1},  # Heavy TP
            3: {'dp': 2, 'tp': 1, 'pp': 2},  # With Pipeline
            4: {'dp': 1, 'tp': 2, 'pp': 2}   # Balanced PP and TP
        }
        return strategies[action]
        
    def _apply_strategy(self, strategy: Dict):
        """Apply the selected parallelism strategy."""
        logger.info(f"Applying new strategy: {strategy}")
        # Implementation depends on your parallelism manager
        pass

class DynamicStrategySelector:
    def __init__(self, 
                 monitoring_interval: int = 100,  # iterations
                 adaptation_threshold: float = 0.15,  # 15% performance change
                 history_window: int = 5,  # number of measurements to consider
                 enable_dynamic_adaptation: bool = True):
        self.monitoring_interval = monitoring_interval
        self.adaptation_threshold = adaptation_threshold
        self.history_window = history_window
        self.enable_dynamic_adaptation = enable_dynamic_adaptation
        
        self.metrics_history: List[MonitoringMetrics] = []
        self.current_config: Optional[ParallelismConfig] = None
        self.strategy_changes: List[Tuple[int, ParallelismConfig]] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.metrics_queue: Queue = Queue()
        self.stop_monitoring = threading.Event()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.monitor = StrategyMonitor()
        self.monitor.start_monitoring()
        
    def initialize_strategy(self, 
                          model_size: int,
                          batch_size: int,
                          num_gpus: int,
                          gpu_memory: int,
                          network_bandwidth: float) -> ParallelismConfig:
        """Initialize the parallelism strategy based on initial conditions."""
        # Calculate basic resource requirements
        params_memory = model_size * 4  # 4 bytes per parameter
        available_memory = gpu_memory * 0.85  # Use 85% of GPU memory
        
        # Determine initial parallelism configuration
        if model_size > 1e10:  # Very large model (>10B parameters)
            config = self._configure_large_model(model_size, num_gpus, available_memory)
        elif model_size > 1e9:  # Large model (1-10B parameters)
            config = self._configure_medium_model(model_size, num_gpus, available_memory)
        else:  # Small to medium model (<1B parameters)
            config = self._configure_small_model(model_size, num_gpus, available_memory)
        
        self.current_config = config
        return config

    def _configure_large_model(self, model_size: int, num_gpus: int, 
                             available_memory: int) -> ParallelismConfig:
        """Configure parallelism for very large models (>10B parameters)."""
        # Use hybrid parallelism for large models
        pipeline_parallel_size = min(8, num_gpus // 4)
        tensor_parallel_size = min(8, num_gpus // pipeline_parallel_size)
        data_parallel_size = num_gpus // (pipeline_parallel_size * tensor_parallel_size)
        
        return ParallelismConfig(
            strategy_type=ParallelismType.HYBRID,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            micro_batch_size=1,
            gradient_accumulation_steps=32,
            zero_optimization_stage=3,
            activation_checkpointing=True,
            cpu_offload=True,
            communication_overlap=True
        )

    def _configure_medium_model(self, model_size: int, num_gpus: int, 
                              available_memory: int) -> ParallelismConfig:
        """Configure parallelism for large models (1-10B parameters)."""
        # Use combination of tensor and data parallelism
        tensor_parallel_size = min(4, num_gpus // 2)
        data_parallel_size = num_gpus // tensor_parallel_size
        
        return ParallelismConfig(
            strategy_type=ParallelismType.HYBRID,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            zero_optimization_stage=2,
            activation_checkpointing=True,
            cpu_offload=False,
            communication_overlap=True
        )

    def _configure_small_model(self, model_size: int, num_gpus: int, 
                             available_memory: int) -> ParallelismConfig:
        """Configure parallelism for small to medium models (<1B parameters)."""
        # Primarily use data parallelism
        return ParallelismConfig(
            strategy_type=ParallelismType.DATA,
            data_parallel_size=num_gpus,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            micro_batch_size=32,
            gradient_accumulation_steps=1,
            zero_optimization_stage=1,
            activation_checkpointing=False,
            cpu_offload=False,
            communication_overlap=True
        )

    def select_optimal_strategy(self, model_profile: Dict, 
                              dataset_profile: Dict, 
                              hardware_profile: Dict) -> Dict:
        """Select optimal parallelization strategy based on profiles."""
        args = get_args()
        
        # Get current parallel configurations
        tp_size = get_tensor_model_parallel_world_size()
        dp_size = get_data_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        
        # Calculate memory requirements
        memory_per_gpu = hardware_profile['gpu_info']['gpu_0']['total_memory']
        model_memory = model_profile['total_parameters'] * 4  # 4 bytes per parameter
        activation_memory = model_profile['activation_memory']
        optimizer_memory = model_memory * 2  # Adam optimizer states
        
        # Calculate communication costs
        tp_comm_cost = self._calculate_tp_communication_cost(
            model_profile, hardware_profile)
        pp_comm_cost = self._calculate_pp_communication_cost(
            model_profile, hardware_profile)
        dp_comm_cost = self._calculate_dp_communication_cost(
            model_profile, dataset_profile, hardware_profile)
        
        # Determine optimal strategy
        strategy = {
            'tensor_parallel': {
                'size': self._optimize_tp_size(
                    model_profile, memory_per_gpu, tp_comm_cost),
                'comm_cost': tp_comm_cost
            },
            'pipeline_parallel': {
                'size': self._optimize_pp_size(
                    model_profile, memory_per_gpu, pp_comm_cost),
                'num_micro_batches': self._calculate_optimal_micro_batches(
                    model_profile, dataset_profile),
                'comm_cost': pp_comm_cost
            },
            'data_parallel': {
                'size': self._optimize_dp_size(
                    dataset_profile, memory_per_gpu, dp_comm_cost),
                'comm_cost': dp_comm_cost
            }
        }
        
        # Add Megatron-specific optimizations
        strategy.update({
            'activation_checkpointing': {
                'enabled': model_memory > memory_per_gpu * 0.3,
                'granularity': 'selective' if model_memory < memory_per_gpu * 0.6 else 'full'
            },
            'zero_optimization': {
                'stage': self._determine_zero_stage(model_memory, memory_per_gpu),
                'contiguous_gradients': True,
                'overlap_comm': True
            },
            'mixed_precision': {
                'enabled': True,
                'dtype': 'fp16' if model_profile['supports_half'] else 'bf16'
            }
        })
        
        return strategy

    def _optimize_tp_size(self, model_profile: Dict, 
                         memory_per_gpu: int, 
                         comm_cost: float) -> int:
        """Optimize tensor parallel size based on model and hardware characteristics."""
        if not dist.is_initialized():
            return 1
            
        total_gpus = torch.cuda.device_count()
        if total_gpus < 2:
            return 1
            
        max_tp_size = min(
            total_gpus,
            model_profile['largest_layer_params'] // (memory_per_gpu * 0.1)
        )
        
        # Consider communication overhead
        if comm_cost > 0.2:  # If comm cost is more than 20% of compute
            max_tp_size = min(max_tp_size, total_gpus // 2)
        
        # Ensure tp_size is a power of 2
        tp_size = 1
        while tp_size * 2 <= max_tp_size:
            tp_size *= 2
        
        return tp_size

    def _optimize_pp_size(self, model_profile: Dict, 
                         memory_per_gpu: int, 
                         comm_cost: float) -> int:
        """Optimize pipeline parallel size based on model characteristics."""
        if not dist.is_initialized():
            return 1
            
        total_gpus = torch.cuda.device_count()
        if total_gpus < 2:
            return 1
            
        num_layers = model_profile['num_layers']
        if num_layers < 2:
            return 1
        
        # Calculate minimum GPUs needed for memory
        min_gpus_memory = max(1, model_profile['total_parameters'] * 4 // memory_per_gpu)
        
        # Consider pipeline bubble overhead
        if comm_cost > 0.1:  # If comm cost is more than 10% of compute
            max_pp_size = min(num_layers // 4, total_gpus // 2)
        else:
            max_pp_size = min(num_layers // 2, total_gpus)
        
        pp_size = max(1, min(min_gpus_memory, max_pp_size))
        
        return pp_size

    def _determine_zero_stage(self, model_memory: int, 
                            memory_per_gpu: int) -> int:
        """Determine optimal ZeRO stage based on model and memory constraints."""
        if not dist.is_initialized():
            return 0
            
        memory_ratio = model_memory / memory_per_gpu
        
        if memory_ratio > 2.0:
            return 3  # Use ZeRO-3 for very large models
        elif memory_ratio > 1.0:
            return 2  # Use ZeRO-2 for large models
        elif memory_ratio > 0.5:
            return 1  # Use ZeRO-1 for medium models
        else:
            return 0  # No ZeRO for small models

    def _calculate_optimal_micro_batches(self, model_profile: Dict, 
                                       dataset_profile: Dict) -> int:
        """Calculate optimal number of micro-batches for pipeline parallelism."""
        if not dist.is_initialized():
            return 1
            
        pp_size = get_pipeline_model_parallel_world_size()
        if pp_size <= 1:
            return 1
        
        # Calculate based on pipeline bubble overhead
        bubble_overhead = lambda num_mb: (pp_size - 1) / (pp_size * num_mb)
        
        # Start with 2x pipeline depth to minimize bubble
        num_micro_batches = pp_size * 2
        
        # Increase if bubble overhead is still too high
        while (bubble_overhead(num_micro_batches) > 0.05 and  # 5% overhead threshold
               num_micro_batches < dataset_profile['batch_size']):
            num_micro_batches *= 2
        
        return num_micro_batches

    def start_monitoring(self):
        """Start the monitoring thread for collecting training metrics."""
        if not self.enable_dynamic_adaptation:
            return

        def monitoring_loop():
            while not self.stop_monitoring.is_set():
                metrics = self._collect_training_metrics()
                self.metrics_queue.put(metrics)
                time.sleep(self.monitoring_interval)

        self.monitoring_thread = threading.Thread(target=monitoring_loop)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.monitoring_thread is not None:
            self.stop_monitoring.set()
            self.monitoring_thread.join()

    def _collect_training_metrics(self) -> MonitoringMetrics:
        """Collect current training metrics."""
        # Collect GPU metrics
        gpu_memory_used = []
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            gpu_memory_used.append(memory_allocated / memory_total * 100)

        # Collect CPU metrics
        cpu_memory_used = psutil.virtual_memory().percent

        # These metrics would need to be collected from the training loop
        # Here we're using placeholder values
        metrics = MonitoringMetrics(
            timestamp=time.time(),
            iteration=0,  # To be filled with actual iteration
            throughput=0.0,  # To be filled with actual throughput
            gpu_memory_used=gpu_memory_used,
            gpu_utilization={},
            communication_overhead=0.0,  # To be filled with actual overhead
            pipeline_bubble_overhead=0.0,  # To be filled with actual bubble overhead
            load_imbalance=0.0,  # To be filled with actual imbalance
            tensor_parallel_efficiency=0.0,  # To be filled with actual efficiency
            data_parallel_efficiency=0.0,  # To be filled with actual efficiency
            gradient_sync_time=0.0,  # To be filled with actual sync time
            batch_processing_time=0.0,  # To be filled with actual batch time
            pipeline_stall_time=0.0,  # To be filled with actual stall time
            memory_reserved={}
        )
        
        return metrics

    def update_metrics(self, metrics: MonitoringMetrics):
        """Update metrics history and trigger strategy adaptation if needed."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_window:
            self.metrics_history.pop(0)

        if self.enable_dynamic_adaptation and len(self.metrics_history) >= self.history_window:
            self._adapt_strategy()

    def _adapt_strategy(self):
        """Adapt parallelism strategy based on collected metrics."""
        current_performance = self._evaluate_current_performance()
        if self._should_adapt(current_performance):
            new_config = self._select_new_strategy(current_performance)
            if new_config != self.current_config:
                self.strategy_changes.append((len(self.metrics_history), new_config))
                self.current_config = new_config
                self._log_strategy_change(new_config)

    def _evaluate_current_performance(self) -> Dict[str, float]:
        """Evaluate current training performance metrics."""
        recent_metrics = self.metrics_history[-self.history_window:]
        
        return {
            'throughput': np.mean([m.throughput for m in recent_metrics]),
            'memory_efficiency': np.mean([np.mean(m.gpu_memory_used) for m in recent_metrics]),
            'communication_efficiency': 1.0 - np.mean([m.communication_overhead for m in recent_metrics]),
            'load_balance': 1.0 - np.mean([m.load_imbalance for m in recent_metrics]),
            'convergence': np.mean([m.tensor_parallel_efficiency for m in recent_metrics])
        }

    def _should_adapt(self, current_performance: Dict[str, float]) -> bool:
        """Determine if strategy adaptation is needed."""
        if len(self.metrics_history) < self.history_window * 2:
            return False

        previous_performance = self._evaluate_performance(
            self.metrics_history[-2*self.history_window:-self.history_window]
        )

        # Calculate relative changes in key metrics
        changes = {
            metric: abs(current_performance[metric] - previous_performance[metric]) / 
                   previous_performance[metric]
            for metric in current_performance
        }

        # Return True if any metric changed significantly
        return any(change > self.adaptation_threshold for change in changes.values())

    def _select_new_strategy(self, 
                           current_performance: Dict[str, float]) -> ParallelismConfig:
        """Select a new parallelism strategy based on current performance."""
        if self.current_config is None:
            return self._configure_small_model(1e8, torch.cuda.device_count(), 
                                            torch.cuda.get_device_properties(0).total_memory)

        # Identify performance bottlenecks
        bottlenecks = self._identify_bottlenecks(current_performance)
        
        # Adjust strategy based on bottlenecks
        new_config = self._adjust_strategy_for_bottlenecks(self.current_config, bottlenecks)
        
        return new_config

    def _identify_bottlenecks(self, 
                            performance: Dict[str, float]) -> Dict[str, float]:
        """Identify performance bottlenecks from metrics."""
        bottlenecks = {}
        
        # Memory bottleneck
        if performance['memory_efficiency'] > 90:
            bottlenecks['memory'] = performance['memory_efficiency'] / 100.0
            
        # Communication bottleneck
        if performance['communication_efficiency'] < 0.7:
            bottlenecks['communication'] = 1.0 - performance['communication_efficiency']
            
        # Load imbalance bottleneck
        if performance['load_balance'] < 0.8:
            bottlenecks['load_balance'] = 1.0 - performance['load_balance']
            
        return bottlenecks

    def _adjust_strategy_for_bottlenecks(self, 
                                       current_config: ParallelismConfig,
                                       bottlenecks: Dict[str, float]) -> ParallelismConfig:
        """Adjust parallelism strategy to address identified bottlenecks."""
        new_config = current_config
        
        if 'memory' in bottlenecks:
            new_config = self._adjust_for_memory_bottleneck(new_config)
        
        if 'communication' in bottlenecks:
            new_config = self._adjust_for_communication_bottleneck(new_config)
            
        if 'load_balance' in bottlenecks:
            new_config = self._adjust_for_load_imbalance(new_config)
            
        return new_config

    def _adjust_for_memory_bottleneck(self, 
                                    config: ParallelismConfig) -> ParallelismConfig:
        """Adjust strategy to address memory bottleneck."""
        # Try increasing model parallelism first
        if config.tensor_parallel_size < 4:
            return ParallelismConfig(
                **{**config.__dict__,
                   'tensor_parallel_size': config.tensor_parallel_size * 2,
                   'data_parallel_size': config.data_parallel_size // 2,
                   'activation_checkpointing': True}
            )
        # If model parallelism is maxed out, try pipeline parallelism
        elif config.pipeline_parallel_size < 4:
            return ParallelismConfig(
                **{**config.__dict__,
                   'pipeline_parallel_size': config.pipeline_parallel_size * 2,
                   'data_parallel_size': config.data_parallel_size // 2}
            )
        # If both are maxed out, enable CPU offloading
        else:
            return ParallelismConfig(
                **{**config.__dict__,
                   'cpu_offload': True,
                   'zero_optimization_stage': 3}
            )

    def _adjust_for_communication_bottleneck(self, 
                                          config: ParallelismConfig) -> ParallelismConfig:
        """Adjust strategy to address communication bottleneck."""
        # Increase gradient accumulation to reduce communication frequency
        return ParallelismConfig(
            **{**config.__dict__,
               'gradient_accumulation_steps': config.gradient_accumulation_steps * 2,
               'communication_overlap': True}
        )

    def _adjust_for_load_imbalance(self, 
                                 config: ParallelismConfig) -> ParallelismConfig:
        """Adjust strategy to address load imbalance."""
        # Reduce pipeline stages if pipeline parallelism is causing imbalance
        if config.pipeline_parallel_size > 1:
            return ParallelismConfig(
                **{**config.__dict__,
                   'pipeline_parallel_size': config.pipeline_parallel_size // 2,
                   'data_parallel_size': config.data_parallel_size * 2}
            )
        # Otherwise, adjust micro-batch size
        else:
            return ParallelismConfig(
                **{**config.__dict__,
                   'micro_batch_size': max(1, config.micro_batch_size // 2)}
            )

    def _log_strategy_change(self, new_config: ParallelismConfig):
        """Log details of parallelism strategy change."""
        self.logger.info(f"Adapting parallelism strategy:")
        self.logger.info(f"New configuration: {json.dumps(new_config.__dict__, indent=2)}")
        
        if len(self.strategy_changes) > 1:
            prev_config = self.strategy_changes[-2][1]
            self.logger.info("Changes from previous configuration:")
            for key, new_value in new_config.__dict__.items():
                old_value = getattr(prev_config, key)
                if new_value != old_value:
                    self.logger.info(f"  {key}: {old_value} -> {new_value}")

    def get_current_strategy(self) -> ParallelismConfig:
        """Get the current parallelism strategy configuration."""
        return self.current_config

    def get_strategy_history(self) -> List[Tuple[int, ParallelismConfig]]:
        """Get the history of strategy changes."""
        return self.strategy_changes

    def export_metrics_history(self, filepath: str):
        """Export metrics history to a JSON file."""
        history_data = []
        for metrics in self.metrics_history:
            history_data.append({
                'timestamp': metrics.timestamp,
                'throughput': metrics.throughput,
                'gpu_memory_used': metrics.gpu_memory_used,
                'cpu_memory_used': metrics.cpu_memory_used,
                'communication_overhead': metrics.communication_overhead,
                'load_imbalance': metrics.load_imbalance,
                'convergence': metrics.convergence_rate
            })
            
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)

    def export_strategy_history(self, filepath: str):
        """Export strategy change history to a JSON file."""
        history_data = []
        for iteration, config in self.strategy_changes:
            history_data.append({
                'iteration': iteration,
                'config': config.__dict__
            })
            
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
