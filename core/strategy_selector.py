"""
Strategy Selector for Dynamic Parallelism with AI-driven decision making
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DynamicStrategySelector')

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
    DATA_PARALLEL = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"
    NONE = "none"

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
    def __init__(self,
                strategy_type: ParallelismType,
                data_parallel_size: int = 1,
                tensor_parallel_size: int = 1,
                pipeline_parallel_size: int = 1,
                micro_batch_size: int = 1,
                gradient_accumulation_steps: int = 1,
                pipeline_chunks: int = 8,
                zero_optimization_stage: int = 0,
                activation_checkpointing: bool = False,
                activation_checkpoint_layers: List[int] = None,
                cpu_offload: bool = False,
                communication_overlap: bool = True):
        self.strategy_type = strategy_type
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pipeline_chunks = pipeline_chunks
        self.zero_optimization_stage = zero_optimization_stage
        self.activation_checkpointing = activation_checkpointing
        self.activation_checkpoint_layers = activation_checkpoint_layers or []
        self.cpu_offload = cpu_offload
        self.communication_overlap = communication_overlap
        
    def validate(self, num_gpus: int) -> bool:
        """Validate if the configuration is valid for given number of GPUs."""
        total_parallel_size = (self.data_parallel_size * 
                             self.tensor_parallel_size * 
                             self.pipeline_parallel_size)
        return total_parallel_size <= num_gpus

    @property
    def total_parallel_size(self) -> int:
        """Total number of GPUs required."""
        return (self.data_parallel_size * 
                self.tensor_parallel_size * 
                self.pipeline_parallel_size)
                
    def __eq__(self, other: 'ParallelismConfig') -> bool:
        """Check if two configurations are equal."""
        if not isinstance(other, ParallelismConfig):
            return False
        return (
            self.strategy_type == other.strategy_type and
            self.data_parallel_size == other.data_parallel_size and
            self.tensor_parallel_size == other.tensor_parallel_size and
            self.pipeline_parallel_size == other.pipeline_parallel_size and
            self.micro_batch_size == other.micro_batch_size and
            self.gradient_accumulation_steps == other.gradient_accumulation_steps and
            self.pipeline_chunks == other.pipeline_chunks and
            self.zero_optimization_stage == other.zero_optimization_stage and
            self.activation_checkpointing == other.activation_checkpointing and
            self.cpu_offload == other.cpu_offload and
            self.communication_overlap == other.communication_overlap
        )

class ProcessGroup:
    """Manages process groups for different types of parallelism using PyTorch distributed."""
    
    def __init__(self, world_size: int, config: ParallelismConfig):
        self.world_size = world_size
        self.config = config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.data_parallel_group = None
        self.tensor_parallel_group = None
        self.pipeline_parallel_group = None
        
        if dist.is_initialized():
            self._initialize_process_groups()
    
    def _initialize_process_groups(self):
        """Initialize process groups for each type of parallelism using PyTorch distributed."""
        # Data parallel group
        dp_size = self.config.data_parallel_size
        dp_groups = []
        for i in range(self.world_size // dp_size):
            ranks = list(range(i * dp_size, (i + 1) * dp_size))
            group = dist.new_group(ranks=ranks)
            dp_groups.append(group)
        self.data_parallel_group = dp_groups[self.rank // dp_size]
        
        # Tensor parallel group
        tp_size = self.config.tensor_parallel_size
        if tp_size > 1:
            tp_groups = []
            ranks_per_tp = self.world_size // tp_size
            for i in range(ranks_per_tp):
                ranks = [i + j * ranks_per_tp for j in range(tp_size)]
                group = dist.new_group(ranks=ranks)
                tp_groups.append(group)
            self.tensor_parallel_group = tp_groups[self.rank % ranks_per_tp]
        
        # Pipeline parallel group
        pp_size = self.config.pipeline_parallel_size
        if pp_size > 1:
            pp_groups = []
            ranks_per_pp = self.world_size // pp_size
            for i in range(ranks_per_pp):
                ranks = list(range(i, self.world_size, ranks_per_pp))
                group = dist.new_group(ranks=ranks)
                pp_groups.append(group)
            self.pipeline_parallel_group = pp_groups[self.rank % ranks_per_pp]
    
    def get_data_parallel_rank(self) -> int:
        """Get rank within data parallel group."""
        if not self.data_parallel_group:
            return 0
        return dist.get_rank(group=self.data_parallel_group)
    
    def get_tensor_parallel_rank(self) -> int:
        """Get rank within tensor parallel group."""
        if not self.tensor_parallel_group:
            return 0
        return dist.get_rank(group=self.tensor_parallel_group)
    
    def get_pipeline_parallel_rank(self) -> int:
        """Get rank within pipeline parallel group."""
        if not self.pipeline_parallel_group:
            return 0
        return dist.get_rank(group=self.pipeline_parallel_group)

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
        strategies = {
            0: {"type": ParallelismType.DATA_PARALLEL},
            1: {"type": ParallelismType.TENSOR_PARALLEL},
            2: {"type": ParallelismType.PIPELINE_PARALLEL},
            3: {"type": ParallelismType.HYBRID},
            4: {"type": ParallelismType.NONE}
        }
        return strategies[action]
        
    def _apply_strategy(self, strategy: Dict):
        """Apply the selected parallelism strategy."""
        logger.info(f"Applying new strategy: {strategy}")
        
        # Create new ParallelismConfig from strategy
        new_config = ParallelismConfig(
            strategy_type=strategy['type'],
            data_parallel_size=strategy.get('data_parallel_size', 1),
            tensor_parallel_size=strategy.get('tensor_parallel_size', 1),
            pipeline_parallel_size=strategy.get('pipeline_parallel_size', 1),
            micro_batch_size=strategy.get('micro_batch_size', 1),
            gradient_accumulation_steps=strategy.get('gradient_accumulation_steps', 1),
            pipeline_chunks=strategy.get('pipeline_chunks', 8),
            zero_optimization_stage=strategy.get('zero_stage', 0),
            activation_checkpointing=strategy.get('activation_checkpointing', False),
            activation_checkpoint_layers=strategy.get('checkpoint_layers', []),
            cpu_offload=strategy.get('cpu_offload', False),
            communication_overlap=strategy.get('communication_overlap', True)
        )
        
        # Validate the new configuration
        try:
            new_config.validate(torch.cuda.device_count())
        except ValueError as e:
            logger.error(f"Invalid strategy configuration: {e}")
            return False
            
        # Initialize new process groups
        if self.process_groups is not None:
            old_groups = self.process_groups
            try:
                self.process_groups = ProcessGroup(dist.get_world_size(), new_config)
            except Exception as e:
                logger.error(f"Failed to initialize new process groups: {e}")
                self.process_groups = old_groups
                return False
                
        # Apply configuration changes
        if hasattr(self, 'parallelism_manager'):
            try:
                # Save model state
                old_state = self.parallelism_manager.model.state_dict()
                
                # Apply new configuration
                self.parallelism_manager.reconfigure(
                    new_config,
                    self.process_groups,
                    preserve_state=True,
                    old_state=old_state
                )
                
                # Update current configuration
                self.current_config = new_config
                
                # Log strategy change
                self._log_strategy_change(new_config)
                
                # Record timestamp of change
                self.strategy_changes.append((time.time(), new_config))
                
                logger.info("Successfully applied new parallelism strategy")
                return True
                
            except Exception as e:
                logger.error(f"Failed to apply new strategy: {e}")
                # Attempt to rollback to previous state
                try:
                    self.parallelism_manager.model.load_state_dict(old_state)
                    logger.info("Successfully rolled back to previous state")
                except:
                    logger.error("Failed to rollback to previous state")
                return False
        else:
            logger.error("No parallelism manager available")
            return False

class DynamicStrategySelector:
    """Manages dynamic parallelism strategy selection and transitions."""
    
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
        
        self.process_groups: Optional[ProcessGroup] = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.monitor = StrategyMonitor()
        
    def initialize_strategy(self, model_characteristics: ModelCharacteristics,
                          hardware_characteristics: HardwareCharacteristics) -> ParallelismConfig:
        """Initialize parallelism strategy based on model and hardware characteristics."""
        num_gpus = hardware_characteristics.num_gpus
        gpu_memory = hardware_characteristics.gpu_memory
        model_size = model_characteristics.parameter_size
        
        if model_size > 10:  # >10B parameters
            config = self._configure_large_model(model_size, num_gpus, gpu_memory)
        elif model_size > 1:  # 1-10B parameters
            config = self._configure_medium_model(model_size, num_gpus, gpu_memory)
        else:  # <1B parameters
            config = self._configure_small_model(model_size, num_gpus, gpu_memory)
            
        self.current_config = config
        self.process_groups = ProcessGroup(num_gpus, config)
        return config
    
    def _configure_large_model(self, model_size: int, num_gpus: int, 
                             gpu_memory: int) -> ParallelismConfig:
        """Configure parallelism for very large models (>10B parameters)."""
        # For large models, prioritize model parallelism
        tp_size = min(8, num_gpus)  # Up to 8-way tensor parallelism
        remaining_gpus = num_gpus // tp_size
        
        pp_size = min(4, remaining_gpus)  # Up to 4 pipeline stages
        dp_size = num_gpus // (tp_size * pp_size)
        
        return ParallelismConfig(
            strategy_type=ParallelismType.HYBRID,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            micro_batch_size=1,
            gradient_accumulation_steps=32,
            zero_optimization_stage=1,
            activation_checkpointing=True
        )
    
    def _configure_medium_model(self, model_size: int, num_gpus: int, 
                              gpu_memory: int) -> ParallelismConfig:
        """Configure parallelism for medium models (1-10B parameters)."""
        # Balance between data and model parallelism
        tp_size = min(4, num_gpus)  # Up to 4-way tensor parallelism
        remaining_gpus = num_gpus // tp_size
        
        pp_size = min(2, remaining_gpus)  # Up to 2 pipeline stages
        dp_size = num_gpus // (tp_size * pp_size)
        
        return ParallelismConfig(
            strategy_type=ParallelismType.HYBRID,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            data_parallel_size=dp_size,
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            zero_optimization_stage=2
        )
    
    def _configure_small_model(self, model_size: int, num_gpus: int, 
                             gpu_memory: int) -> ParallelismConfig:
        """Configure parallelism for small models (<1B parameters)."""
        # Prioritize data parallelism for small models
        return ParallelismConfig(
            strategy_type=ParallelismType.DATA_PARALLEL,
            data_parallel_size=num_gpus,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            micro_batch_size=32,
            gradient_accumulation_steps=1,
            zero_optimization_stage=2
        )
    
    def update_metrics(self, metrics: MonitoringMetrics):
        """Update metrics history and trigger strategy adaptation if needed."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_window:
            self.metrics_history.pop(0)
        
        if self.enable_dynamic_adaptation and len(self.metrics_history) >= self.history_window:
            self._adapt_strategy()
    
    def _adapt_strategy(self):
        """Adapt parallelism strategy based on collected metrics."""
        if not self.enable_dynamic_adaptation or len(self.metrics_history) < self.history_window:
            return
            
        current_performance = self._evaluate_current_performance()
        if not self._should_adapt(current_performance):
            return
            
        bottlenecks = self._identify_bottlenecks(current_performance)
        new_config = self._adjust_strategy_for_bottlenecks(
            self.current_config, bottlenecks
        )
        
        if new_config != self.current_config:
            self._log_strategy_change(new_config)
            self.current_config = new_config
            self.strategy_changes.append((len(self.metrics_history), new_config))
            
    def _adjust_strategy_for_bottlenecks(self, current_config: ParallelismConfig,
                                       bottlenecks: Dict[str, float]) -> ParallelismConfig:
        """Adjust parallelism strategy to address identified bottlenecks."""
        new_config = current_config
        
        # Memory bottleneck
        if bottlenecks.get('memory', 0) > 0.8:  # 80% memory usage
            new_config = self._adjust_for_memory_bottleneck(new_config)
            
        # Communication bottleneck
        elif bottlenecks.get('communication', 0) > 0.3:  # 30% comm overhead
            new_config = self._adjust_for_communication_bottleneck(new_config)
            
        # GPU utilization bottleneck
        elif bottlenecks.get('utilization', 0) < 0.5:  # 50% utilization
            new_config = self._adjust_for_utilization_bottleneck(new_config)
            
        return new_config
        
    def _adjust_for_memory_bottleneck(self, config: ParallelismConfig) -> ParallelismConfig:
        """Adjust strategy to address memory bottleneck."""
        if config.strategy_type == ParallelismType.DATA_PARALLEL:
            # Switch to tensor parallel to reduce memory per GPU
            return ParallelismConfig(
                strategy_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=min(config.data_parallel_size, 4),
                pipeline_parallel_size=1,
                activation_checkpointing=True
            )
        elif config.strategy_type == ParallelismType.TENSOR_PARALLEL:
            # Add pipeline parallel to further reduce memory
            return ParallelismConfig(
                strategy_type=ParallelismType.HYBRID,
                tensor_parallel_size=config.tensor_parallel_size,
                pipeline_parallel_size=2,
                activation_checkpointing=True
            )
        return config
        
    def _adjust_for_communication_bottleneck(self, config: ParallelismConfig) -> ParallelismConfig:
        """Adjust strategy to address communication bottleneck."""
        if config.strategy_type == ParallelismType.DATA_PARALLEL:
            # Reduce data parallel size and add pipeline parallel
            return ParallelismConfig(
                strategy_type=ParallelismType.PIPELINE_PARALLEL,
                data_parallel_size=max(1, config.data_parallel_size // 2),
                pipeline_parallel_size=2,
                pipeline_chunks=8
            )
        elif config.strategy_type == ParallelismType.TENSOR_PARALLEL:
            # Reduce tensor parallel size
            return ParallelismConfig(
                strategy_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=max(2, config.tensor_parallel_size // 2)
            )
        return config
        
    def _adjust_for_utilization_bottleneck(self, config: ParallelismConfig) -> ParallelismConfig:
        """Adjust strategy to address GPU utilization bottleneck."""
        if config.strategy_type == ParallelismType.PIPELINE_PARALLEL:
            # Switch to data parallel for better utilization
            return ParallelismConfig(
                strategy_type=ParallelismType.DATA_PARALLEL,
                data_parallel_size=config.pipeline_parallel_size * 2
            )
        elif config.strategy_type == ParallelismType.HYBRID:
            # Simplify to just tensor parallel
            return ParallelismConfig(
                strategy_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=config.tensor_parallel_size
            )
        return config
    
    def _evaluate_current_performance(self) -> Dict[str, float]:
        """Evaluate current training performance metrics."""
        recent_metrics = self.metrics_history[-self.history_window:]
        
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_gpu_util = np.mean([sum(m.gpu_utilization.values()) / len(m.gpu_utilization)
                              for m in recent_metrics])
        avg_comm_overhead = np.mean([m.communication_overhead for m in recent_metrics])
        
        return {
            'throughput': avg_throughput,
            'gpu_utilization': avg_gpu_util,
            'communication_overhead': avg_comm_overhead,
            'memory_utilization': np.mean([sum(m.gpu_memory_used.values()) / 
                                         sum(m.memory_reserved.values())
                                         for m in recent_metrics])
        }
    
    def _should_adapt(self, current_performance: Dict[str, float]) -> bool:
        """Determine if strategy adaptation is needed."""
        if len(self.metrics_history) < self.history_window * 2:
            return False
            
        prev_metrics = self.metrics_history[-self.history_window*2:-self.history_window]
        prev_throughput = np.mean([m.throughput for m in prev_metrics])
        
        throughput_change = ((current_performance['throughput'] - prev_throughput) / 
                           prev_throughput)
        
        return (abs(throughput_change) > self.adaptation_threshold or
                current_performance['gpu_utilization'] < 0.5 or
                current_performance['memory_utilization'] > 0.95)
    
    def _select_new_strategy(self, current_performance: Dict[str, float]) -> ParallelismConfig:
        """Select a new parallelism strategy based on current performance."""
        bottlenecks = self._identify_bottlenecks(current_performance)
        return self._adjust_strategy_for_bottlenecks(self.current_config, bottlenecks)
    
    def _identify_bottlenecks(self, performance: Dict[str, float]) -> Dict[str, float]:
        """Identify performance bottlenecks from metrics."""
        bottlenecks = {}
        
        if performance['memory_utilization'] > 0.9:
            bottlenecks['memory'] = performance['memory_utilization']
            
        if performance['communication_overhead'] > 0.3:
            bottlenecks['communication'] = performance['communication_overhead']
            
        if performance['gpu_utilization'] < 0.5:
            bottlenecks['utilization'] = performance['gpu_utilization']
            
        return bottlenecks
    
    def _log_strategy_change(self, new_config: ParallelismConfig):
        """Log details of parallelism strategy change."""
        self.logger.info(f"Adapting parallelism strategy:")
        self.logger.info(f"- Data Parallel Size: {new_config.data_parallel_size}")
        self.logger.info(f"- Tensor Parallel Size: {new_config.tensor_parallel_size}")
        self.logger.info(f"- Pipeline Parallel Size: {new_config.pipeline_parallel_size}")
        self.logger.info(f"- Micro Batch Size: {new_config.micro_batch_size}")
        self.logger.info(f"- Gradient Accumulation Steps: {new_config.gradient_accumulation_steps}")
        self.logger.info(f"- ZeRO Stage: {new_config.zero_optimization_stage}")
        
    def get_current_strategy(self) -> ParallelismConfig:
        """Get the current parallelism strategy configuration."""
        return self.current_config
    
    def get_strategy_history(self) -> List[Tuple[int, ParallelismConfig]]:
        """Get the history of strategy changes."""
        return self.strategy_changes
    
    def export_metrics_history(self, filepath: str):
        """Export metrics history to a JSON file."""
        history = []
        for metric in self.metrics_history:
            history.append({
                'timestamp': metric.timestamp,
                'iteration': metric.iteration,
                'throughput': metric.throughput,
                'gpu_memory_used': metric.gpu_memory_used,
                'gpu_utilization': metric.gpu_utilization,
                'communication_overhead': metric.communication_overhead
            })
            
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
            
    def export_strategy_history(self, filepath: str):
        """Export strategy change history to a JSON file."""
        history = []
        for iteration, config in self.strategy_changes:
            history.append({
                'iteration': iteration,
                'strategy_type': config.strategy_type.value,
                'data_parallel_size': config.data_parallel_size,
                'tensor_parallel_size': config.tensor_parallel_size,
                'pipeline_parallel_size': config.pipeline_parallel_size,
                'micro_batch_size': config.micro_batch_size,
                'gradient_accumulation_steps': config.gradient_accumulation_steps,
                'zero_optimization_stage': config.zero_optimization_stage
            })
            
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
