"""
Communication Optimizer for Dynamic Distributed Training
Optimizes communication patterns and strategies for distributed training
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from dataclasses import dataclass
import threading
import time
import logging
from enum import Enum
from collections import defaultdict
import networkx as nx
from queue import PriorityQueue
import json

class CommunicationPattern(Enum):
    """Types of communication patterns."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    POINT_TO_POINT = "p2p"
    PIPELINE = "pipeline"

@dataclass
class CommunicationStats:
    """Statistics for communication operations."""
    pattern: CommunicationPattern
    size_bytes: int
    duration_ms: float
    source_rank: Optional[int]
    target_rank: Optional[int]
    collective_group: Optional[str]
    bandwidth_gbps: float
    latency_ms: float

class TopologyManager:
    """Manages GPU topology and interconnect information."""
    
    def __init__(self):
        self.topology_graph = nx.Graph()
        self.nvlink_matrix = None
        self.pcie_matrix = None
        self.numa_groups = defaultdict(set)
        self._initialize_topology()
        
    def _initialize_topology(self):
        """Initialize GPU topology information."""
        num_gpus = torch.cuda.device_count()
        
        # Create placeholder topology (replace with actual NCCL topology discovery)
        self.nvlink_matrix = torch.zeros(num_gpus, num_gpus, dtype=torch.bool)
        self.pcie_matrix = torch.ones(num_gpus, num_gpus, dtype=torch.bool)
        
        # Add nodes and edges to topology graph
        for i in range(num_gpus):
            self.topology_graph.add_node(i, type='gpu')
            self.numa_groups[0].add(i)  # Placeholder NUMA grouping
            
        # Add connections (placeholder - replace with actual topology discovery)
        for i in range(num_gpus):
            for j in range(i + 1, num_gpus):
                if i // 2 == j // 2:  # Simulate NVLink between pairs
                    self.nvlink_matrix[i, j] = True
                    self.nvlink_matrix[j, i] = True
                    self.topology_graph.add_edge(i, j, type='nvlink', bandwidth=50.0)  # 50 GB/s
                else:
                    self.topology_graph.add_edge(i, j, type='pcie', bandwidth=16.0)  # 16 GB/s
                    
    def get_optimal_groups(self, group_size: int) -> List[List[int]]:
        """Get optimal process groups based on topology."""
        num_gpus = torch.cuda.device_count()
        if group_size > num_gpus:
            raise ValueError(f"Group size {group_size} exceeds available GPUs {num_gpus}")
            
        groups = []
        used_gpus = set()
        
        # First, try to create groups within NUMA nodes
        for numa_gpus in self.numa_groups.values():
            available_gpus = numa_gpus - used_gpus
            while len(available_gpus) >= group_size:
                # Find best connected subgroup
                best_group = self._find_best_connected_subgroup(list(available_gpus), group_size)
                groups.append(best_group)
                used_gpus.update(best_group)
                available_gpus = numa_gpus - used_gpus
                
        # Then, create groups from remaining GPUs
        remaining_gpus = set(range(num_gpus)) - used_gpus
        while len(remaining_gpus) >= group_size:
            best_group = self._find_best_connected_subgroup(list(remaining_gpus), group_size)
            groups.append(best_group)
            used_gpus.update(best_group)
            remaining_gpus = set(range(num_gpus)) - used_gpus
            
        return groups
        
    def _find_best_connected_subgroup(self, available_gpus: List[int], group_size: int) -> List[int]:
        """Find the best connected subgroup of GPUs."""
        best_group = None
        best_score = float('-inf')
        
        for combination in self._generate_combinations(available_gpus, group_size):
            score = self._evaluate_group_connectivity(combination)
            if score > best_score:
                best_score = score
                best_group = combination
                
        return best_group
        
    def _evaluate_group_connectivity(self, gpus: List[int]) -> float:
        """Evaluate the connectivity score of a group of GPUs."""
        score = 0.0
        for i in range(len(gpus)):
            for j in range(i + 1, len(gpus)):
                if self.nvlink_matrix[gpus[i], gpus[j]]:
                    score += 1.0
                elif self.pcie_matrix[gpus[i], gpus[j]]:
                    score += 0.3
        return score
        
    def _generate_combinations(self, items: List[int], size: int) -> List[List[int]]:
        """Generate all combinations of given size."""
        if size == 0:
            return [[]]
        if not items:
            return []
        return [[items[0]] + combo for combo in self._generate_combinations(items[1:], size-1)] + \
               self._generate_combinations(items[1:], size)

class CommunicationOptimizer:
    """Optimizes communication patterns in distributed training."""
    
    def __init__(self, 
                 world_size: int,
                 monitoring_interval: float = 0.1,
                 enable_fusion: bool = True,
                 enable_overlap: bool = True,
                 compression_threshold: int = 1024*1024):  # 1MB
        self.world_size = world_size
        self.monitoring_interval = monitoring_interval
        self.enable_fusion = enable_fusion
        self.enable_overlap = enable_overlap
        self.compression_threshold = compression_threshold
        
        self.topology_manager = TopologyManager()
        self.communication_stats: List[CommunicationStats] = []
        self.operation_queue = PriorityQueue()
        self.fusion_buffers: Dict[str, torch.Tensor] = {}
        self.compression_states: Dict[str, Any] = {}
        self.process_groups: Dict[str, dist.ProcessGroup] = {}
        
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        self.group_sizes = []
        self.max_tensor_size = 1024 * 1024 * 1024  # 1GB
        self.compression_enabled = False
        self.compression_ratio = 1.0
        self.quantization_bits = 8
        self.start_time = time.time()
        
    def start_monitoring(self):
        """Start communication monitoring."""
        if self.monitoring_thread is not None:
            return
            
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop communication monitoring."""
        if self.monitoring_thread is None:
            return
            
        self.stop_monitoring.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None
        
    def _monitoring_loop(self):
        """Monitor communication patterns and performance."""
        while not self.stop_monitoring.is_set():
            try:
                self._collect_communication_stats()
                self._optimize_communication_patterns()
                self.validate_communication_pattern()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in communication monitoring: {e}")
                
    def _collect_communication_stats(self):
        """Collect statistics about communication operations."""
        # Get current time for measuring duration
        start_time = time.time()
        
        # Monitor active operations
        active_ops = self._get_pending_operations()
        for op in active_ops:
            # Measure operation characteristics
            size_bytes = op.tensor.numel() * op.tensor.element_size()
            
            # Wait for operation to complete
            if hasattr(op, 'wait'):
                op.wait()
            
            # Calculate duration and bandwidth
            duration_ms = (time.time() - start_time) * 1000
            bandwidth_gbps = (size_bytes * 8) / (duration_ms * 1e6) if duration_ms > 0 else 0
            
            # Estimate latency (simplified model)
            latency_ms = duration_ms - (size_bytes / (bandwidth_gbps * 1e9 / 8))
            
            # Create and store statistics
            stats = CommunicationStats(
                pattern=op.pattern if hasattr(op, 'pattern') else CommunicationPattern.ALL_REDUCE,
                size_bytes=size_bytes,
                duration_ms=duration_ms,
                source_rank=dist.get_rank(),
                target_rank=None,  # For collective operations
                collective_group=str(op.group) if hasattr(op, 'group') else None,
                bandwidth_gbps=bandwidth_gbps,
                latency_ms=max(0, latency_ms)  # Ensure non-negative latency
            )
            self.communication_stats.append(stats)

    def _optimize_communication_patterns(self):
        """Optimize communication patterns based on collected statistics."""
        stats_by_pattern = self._analyze_communication_patterns()
        frequent_patterns = self._identify_frequent_patterns(stats_by_pattern)
        
        # Optimize process groups based on topology
        for pattern, stats in frequent_patterns.items():
            if pattern in [CommunicationPattern.ALL_REDUCE, CommunicationPattern.ALL_GATHER]:
                optimal_groups = self.topology_manager.get_optimal_groups(self.world_size)
                self._update_process_groups(pattern, optimal_groups)
        
        # Schedule operations optimally
        self._optimize_operation_scheduling()
        
        # Apply tensor fusion for small operations
        if self.enable_fusion:
            self._optimize_tensor_fusion()
            
        # Enable computation-communication overlap
        if self.enable_overlap:
            self._optimize_computation_overlap()

    def _analyze_communication_patterns(self) -> Dict[CommunicationPattern, List[CommunicationStats]]:
        """Analyze communication patterns from collected statistics."""
        stats_by_pattern = defaultdict(list)
        for stat in self.communication_stats:
            stats_by_pattern[stat.pattern].append(stat)
        return dict(stats_by_pattern)

    def _identify_frequent_patterns(self, 
                                  stats: Dict[CommunicationPattern, List[CommunicationStats]]) -> Dict[CommunicationPattern, List[CommunicationStats]]:
        """Identify frequently occurring communication patterns."""
        frequent_patterns = {}
        for pattern, pattern_stats in stats.items():
            if len(pattern_stats) >= 5:  # Consider patterns that occur at least 5 times
                frequent_patterns[pattern] = pattern_stats
        return frequent_patterns

    def _optimize_operation_scheduling(self):
        """Schedule communication operations optimally."""
        pending_ops = self._get_pending_operations()
        for op in pending_ops:
            if self._should_postpone(op):
                continue
            if self._can_overlap(op):
                self._schedule_overlap(op)
            else:
                self._execute_operation(op)

    def _optimize_tensor_fusion(self):
        """Fuse small tensor operations to reduce overhead."""
        fusion_opportunities = self._identify_fusion_opportunities()
        for ops in fusion_opportunities:
            self._fuse_operations(ops)

    def _optimize_computation_overlap(self):
        """Optimize overlap of computation and communication."""
        for operation in self._get_pending_operations():
            if self._can_overlap(operation):
                self._schedule_overlap(operation)

    def _should_postpone(self, operation: Any) -> bool:
        """Determine if an operation should be postponed."""
        # Check if operation depends on pending operations
        if hasattr(operation, 'dependencies'):
            return any(dep in self._get_pending_operations() for dep in operation.dependencies)
        return False

    def _execute_operation(self, operation: Any):
        """Execute a communication operation."""
        if operation.pattern == CommunicationPattern.ALL_REDUCE:
            self._direct_all_reduce(operation.tensor, operation.group)
        elif operation.pattern == CommunicationPattern.ALL_GATHER:
            self._direct_all_gather(operation.tensor, operation.group)
        # Add other patterns as needed

    def _identify_fusion_opportunities(self) -> List[List[Any]]:
        """Identify opportunities for operation fusion."""
        small_ops = []
        fusion_groups = []
        
        # Group small operations by pattern and target ranks
        for op in self._get_pending_operations():
            if op.tensor.numel() * op.tensor.element_size() < self.compression_threshold:
                small_ops.append(op)
        
        # Group operations with same pattern and compatible ranks
        current_group = []
        for op in small_ops:
            if not current_group or (
                op.pattern == current_group[0].pattern and
                op.group == current_group[0].group
            ):
                current_group.append(op)
            else:
                if len(current_group) > 1:
                    fusion_groups.append(current_group)
                current_group = [op]
                
        if len(current_group) > 1:
            fusion_groups.append(current_group)
            
        return fusion_groups

    def _fuse_operations(self, operations: List[Any]):
        """Fuse multiple operations into a single operation."""
        if not operations:
            return
            
        pattern = operations[0].pattern
        group = operations[0].group
        
        # Concatenate tensors
        tensors = [op.tensor for op in operations]
        fused_tensor = torch.cat(tensors)
        
        # Create fusion buffer key
        key = f"{pattern}_{group}_{len(operations)}"
        self.fusion_buffers[key] = fused_tensor
        
        # Execute fused operation
        if pattern == CommunicationPattern.ALL_REDUCE:
            self._direct_all_reduce(fused_tensor, group)
        elif pattern == CommunicationPattern.ALL_GATHER:
            self._direct_all_gather(fused_tensor, group)
            
        # Split results back
        start_idx = 0
        for i, tensor in enumerate(tensors):
            size = tensor.numel()
            tensor.copy_(fused_tensor[start_idx:start_idx + size].view_as(tensor))
            start_idx += size

    def _can_overlap(self, operation: Any) -> bool:
        """Determine if an operation can be overlapped with computation."""
        # Check if operation size is large enough to benefit from overlap
        if operation.tensor.numel() * operation.tensor.element_size() < self.compression_threshold:
            return False
            
        # Check if operation has no immediate dependencies
        if hasattr(operation, 'dependencies') and operation.dependencies:
            return False
            
        return True

    def _schedule_overlap(self, operation: Any):
        """Schedule an operation to overlap with computation."""
        # Create non-blocking operation
        if operation.pattern == CommunicationPattern.ALL_REDUCE:
            handle = dist.all_reduce(operation.tensor, group=operation.group, async_op=True)
        elif operation.pattern == CommunicationPattern.ALL_GATHER:
            handle = dist.all_gather(operation.tensor, group=operation.group, async_op=True)
            
        # Add to operation queue with priority based on size
        priority = operation.tensor.numel() * operation.tensor.element_size()
        self.operation_queue.put((-priority, handle))

    def _get_pending_operations(self) -> List[Any]:
        """Get list of pending communication operations."""
        pending = []
        while not self.operation_queue.empty():
            _, op = self.operation_queue.get()
            pending.append(op)
        return pending

    def _update_process_groups(self, pattern: CommunicationPattern, groups: List[List[int]]):
        """Update process groups for a communication pattern."""
        for group_ranks in groups:
            group = dist.new_group(ranks=group_ranks)
            key = f"{pattern.value}_{','.join(map(str, group_ranks))}"
            self.process_groups[key] = group

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress tensor for communication."""
        # Simple 8-bit quantization
        if tensor.dtype == torch.float32:
            # Calculate scale factor for quantization
            max_val = torch.max(torch.abs(tensor))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            
            # Quantize to int8
            quantized = torch.clamp(tensor / scale, -127, 127).round().to(torch.int8)
            
            # Store scale factor for decompression
            self.compression_states[tensor] = scale
            
            return quantized
        return tensor

    def _decompress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decompress tensor after communication."""
        if tensor.dtype == torch.int8:
            # Get scale factor used during compression
            scale = self.compression_states.get(tensor, 1.0)
            
            # Dequantize back to float32
            dequantized = tensor.to(torch.float32) * scale
            
            # Clean up compression state
            if tensor in self.compression_states:
                del self.compression_states[tensor]
                
            return dequantized
        return tensor

    def optimize_all_reduce(self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """Optimized all-reduce operation."""
        if tensor.numel() * tensor.element_size() < self.compression_threshold:
            return self._direct_all_reduce(tensor, group)
            
        compressed_tensor = self._compress_tensor(tensor)
        result = self._direct_all_reduce(compressed_tensor, group)
        return self._decompress_tensor(result)
        
    def optimize_all_gather(self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> List[torch.Tensor]:
        """Optimized all-gather operation."""
        if tensor.numel() * tensor.element_size() < self.compression_threshold:
            return self._direct_all_gather(tensor, group)
            
        compressed_tensor = self._compress_tensor(tensor)
        compressed_results = self._direct_all_gather(compressed_tensor, group)
        return [self._decompress_tensor(t) for t in compressed_results]
        
    def _direct_all_reduce(self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """Direct all-reduce without optimization."""
        dist.all_reduce(tensor, group=group)
        return tensor
        
    def _direct_all_gather(self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> List[torch.Tensor]:
        """Direct all-gather without optimization."""
        world_size = dist.get_world_size(group)
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor, group=group)
        return output

    def export_stats(self, filepath: str):
        """Export communication statistics to a file."""
        stats_data = []
        for stat in self.communication_stats:
            stats_data.append({
                'pattern': stat.pattern.value,
                'size_bytes': stat.size_bytes,
                'duration_ms': stat.duration_ms,
                'bandwidth_gbps': stat.bandwidth_gbps,
                'latency_ms': stat.latency_ms
            })
            
        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)

    def validate_communication_pattern(self) -> bool:
        """Validate current communication pattern is efficient and safe."""
        try:
            # Check communication group sizes
            if dist.is_initialized():
                world_size = dist.get_world_size()
                if any(group_size > world_size for group_size in self.group_sizes):
                    raise ValueError("Communication group size exceeds world size")
                    
            # Validate tensor sizes for each operation
            for op in self.communication_stats:
                tensor_size = op.size_bytes
                if tensor_size > self.max_tensor_size:
                    self.logger.warning(f"Large tensor detected in {op.pattern}: {tensor_size} bytes")
                    
            # Check for communication bottlenecks
            total_comm_time = sum(stat.duration_ms for stat in self.communication_stats[-100:])
            total_time = time.time() - self.start_time
            comm_overhead = total_comm_time / (total_time * 1000)
            
            if comm_overhead > 0.5:  # More than 50% time in communication
                self.logger.warning("High communication overhead detected")
                
            # Validate compression settings
            if self.compression_enabled:
                if not self._validate_compression_settings():
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Communication pattern validation failed: {e}")
            return False
            
    def _validate_compression_settings(self) -> bool:
        """Validate tensor compression settings."""
        try:
            # Check if compression ratio is reasonable
            if self.compression_ratio > 32:
                raise ValueError(f"Unusually high compression ratio: {self.compression_ratio}")
                
            # Validate quantization parameters
            if self.quantization_bits not in [4, 8, 16]:
                raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
                
            # Check if compression is beneficial
            avg_tensor_size = np.mean([stat.size_bytes for stat in self.communication_stats[-100:]])
            if avg_tensor_size < 1024 and self.compression_enabled:  # 1KB threshold
                self.logger.warning("Compression may not be beneficial for small tensors")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Compression validation failed: {e}")
            return False
