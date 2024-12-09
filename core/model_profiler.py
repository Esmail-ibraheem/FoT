"""
Model Profiler for Optimus-Megatron
Analyzes model architecture and computational requirements for optimal parallelization strategy.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time

# Megatron-specific imports
from megatron import get_args
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import init_method_normal
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.transformer import ParallelMLP, ParallelAttention
from megatron.mpu import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)

@dataclass
class LayerProfile:
    name: str
    parameter_count: int
    forward_flops: int
    backward_flops: int
    activation_memory: int
    parameter_memory: int
    communication_cost: float
    compute_intensity: float

@dataclass
class ModelProfile:
    total_parameters: int
    total_layers: int
    attention_layers: int
    ffn_layers: int
    layer_profiles: List[LayerProfile]
    memory_requirement: int
    communication_pattern: Dict[str, float]

class ModelProfiler:
    def __init__(self):
        self.profile_cache = {}
        self.layer_stats = {}
        self.communication_patterns = {}

    def profile_model(self, model: nn.Module, sample_batch: torch.Tensor) -> ModelProfile:
        """Profile the complete model architecture and computational requirements."""
        if id(model) in self.profile_cache:
            return self.profile_cache[id(model)]

        total_params = sum(p.numel() for p in model.parameters())
        layer_profiles = []
        attention_layers = 0
        ffn_layers = 0

        # Profile each layer
        for name, module in model.named_modules():
            if self._is_attention_layer(module):
                attention_layers += 1
                layer_profiles.append(self._profile_attention_layer(name, module, sample_batch))
            elif self._is_ffn_layer(module):
                ffn_layers += 1
                layer_profiles.append(self._profile_ffn_layer(name, module, sample_batch))

        # Calculate communication patterns
        comm_pattern = self._analyze_communication_patterns(model, layer_profiles)

        profile = ModelProfile(
            total_parameters=total_params,
            total_layers=len(layer_profiles),
            attention_layers=attention_layers,
            ffn_layers=ffn_layers,
            layer_profiles=layer_profiles,
            memory_requirement=self._calculate_memory_requirement(layer_profiles),
            communication_pattern=comm_pattern
        )

        self.profile_cache[id(model)] = profile
        return profile

    def _is_attention_layer(self, module: nn.Module) -> bool:
        """Detect if a module is an attention layer."""
        attention_layer_types = (
            'MultiHeadAttention',
            'SelfAttention',
            'ParallelAttention',
            'CrossAttention'
        )
        return (isinstance(module, ParallelAttention) or 
                any(attention_type in module.__class__.__name__ 
                    for attention_type in attention_layer_types))

    def _is_ffn_layer(self, module: nn.Module) -> bool:
        """Detect if a module is a feed-forward network layer."""
        ffn_layer_types = (
            'MLP',
            'FFN',
            'ParallelMLP',
            'FeedForward'
        )
        return (isinstance(module, ParallelMLP) or 
                any(ffn_type in module.__class__.__name__ 
                    for ffn_type in ffn_layer_types))

    def _profile_attention_layer(self, name: str, module: nn.Module, 
                               sample_batch: torch.Tensor) -> LayerProfile:
        """Profile an attention layer's computational characteristics."""
        param_count = sum(p.numel() for p in module.parameters())
        
        # Get Megatron-specific configurations
        args = get_args()
        world_size = get_tensor_model_parallel_world_size()
        
        # Calculate FLOPs for attention mechanism
        batch_size, seq_len, hidden_size = sample_batch.shape
        
        # For Megatron parallel attention, adjust for model parallelism
        if isinstance(module, ParallelAttention):
            hidden_size_per_partition = hidden_size // world_size
            num_attention_heads_per_partition = module.num_attention_heads // world_size
            
            # QKV projection FLOPs
            qkv_flops = 3 * batch_size * seq_len * hidden_size_per_partition * hidden_size
            
            # Attention FLOPs (per partition)
            attention_flops = (batch_size * num_attention_heads_per_partition * 
                             seq_len * seq_len * (hidden_size_per_partition // num_attention_heads_per_partition))
            
            # Output projection FLOPs
            output_flops = batch_size * seq_len * hidden_size_per_partition * hidden_size
        else:
            # Standard attention FLOPs calculation
            num_heads = getattr(module, 'num_attention_heads', args.num_attention_heads)
            qkv_flops = 3 * batch_size * seq_len * hidden_size * hidden_size
            attention_flops = batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)
            output_flops = batch_size * seq_len * hidden_size * hidden_size
        
        forward_flops = qkv_flops + attention_flops + output_flops
        backward_flops = forward_flops * 2  # Approximation for backward pass
        
        # Memory calculations with parallel considerations
        param_memory = param_count * 4  # 4 bytes per parameter
        activation_memory = (batch_size * seq_len * hidden_size * 4 * 4)  # 4 intermediate tensors
        
        # Communication cost estimation for parallel attention
        comm_cost = self._estimate_communication_cost(param_count, activation_memory)
        if isinstance(module, ParallelAttention):
            comm_cost *= (world_size - 1) / world_size  # Account for all-reduce
        
        # Compute intensity (FLOPs per byte of memory access)
        compute_intensity = forward_flops / (param_memory + activation_memory)
        
        return LayerProfile(
            name=name,
            parameter_count=param_count,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            activation_memory=activation_memory,
            parameter_memory=param_memory,
            communication_cost=comm_cost,
            compute_intensity=compute_intensity
        )

    def _profile_ffn_layer(self, name: str, module: nn.Module, 
                          sample_batch: torch.Tensor) -> LayerProfile:
        """Profile a feed-forward network layer's computational characteristics."""
        param_count = sum(p.numel() for p in module.parameters())
        
        # Get Megatron-specific configurations
        args = get_args()
        world_size = get_tensor_model_parallel_world_size()
        
        # Calculate FLOPs for FFN
        batch_size, seq_len, hidden_size = sample_batch.shape
        
        # For Megatron parallel MLP, adjust for model parallelism
        if isinstance(module, ParallelMLP):
            hidden_size_per_partition = hidden_size // world_size
            intermediate_size = module.intermediate_size // world_size
            
            # First linear layer FLOPs
            first_linear_flops = batch_size * seq_len * hidden_size_per_partition * intermediate_size
            
            # Second linear layer FLOPs
            second_linear_flops = batch_size * seq_len * intermediate_size * hidden_size_per_partition
            
            forward_flops = first_linear_flops + second_linear_flops
        else:
            # Standard FFN FLOPs calculation
            intermediate_size = getattr(module, 'intermediate_size', hidden_size * 4)
            forward_flops = (2 * batch_size * seq_len * hidden_size * intermediate_size)
        
        backward_flops = forward_flops * 2  # Approximation for backward pass
        
        # Memory calculations with parallel considerations
        param_memory = param_count * 4  # 4 bytes per parameter
        activation_memory = (batch_size * seq_len * intermediate_size * 4)
        
        # Communication cost estimation for parallel FFN
        comm_cost = self._estimate_communication_cost(param_count, activation_memory)
        if isinstance(module, ParallelMLP):
            comm_cost *= (world_size - 1) / world_size  # Account for all-reduce
        
        # Compute intensity (FLOPs per byte of memory access)
        compute_intensity = forward_flops / (param_memory + activation_memory)
        
        return LayerProfile(
            name=name,
            parameter_count=param_count,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            activation_memory=activation_memory,
            parameter_memory=param_memory,
            communication_cost=comm_cost,
            compute_intensity=compute_intensity
        )

    def _calculate_memory_requirement(self, layer_profiles: List[LayerProfile]) -> int:
        """Calculate total memory requirement for the model."""
        total_param_memory = sum(layer.parameter_memory for layer in layer_profiles)
        total_activation_memory = sum(layer.activation_memory for layer in layer_profiles)
        optimizer_memory = total_param_memory * 2  # For Adam optimizer states
        buffer_memory = int(total_activation_memory * 0.1)  # 10% buffer
        
        return total_param_memory + total_activation_memory + optimizer_memory + buffer_memory

    def _estimate_communication_cost(self, param_count: int, activation_memory: int) -> float:
        """Estimate communication cost for parameter and activation synchronization."""
        # This is a simplified model assuming all-reduce communication pattern
        bytes_to_communicate = param_count * 4  # 4 bytes per parameter
        comm_cost = bytes_to_communicate / (10 * 1024 * 1024 * 1024)  # Assuming 10GB/s bandwidth
        return comm_cost

    def _analyze_communication_patterns(self, model: nn.Module, 
                                     layer_profiles: List[LayerProfile]) -> Dict[str, float]:
        """Analyze communication patterns between model components."""
        return {
            'all_reduce': sum(layer.communication_cost for layer in layer_profiles),
            'pipeline': self._estimate_pipeline_communication(layer_profiles),
            'parameter_server': self._estimate_ps_communication(layer_profiles)
        }

    def _estimate_pipeline_communication(self, layer_profiles: List[LayerProfile]) -> float:
        """Estimate communication cost for pipeline parallelism."""
        total_activation_size = sum(layer.activation_memory for layer in layer_profiles)
        return total_activation_size / (10 * 1024 * 1024 * 1024)  # Assuming 10GB/s bandwidth

    def _estimate_ps_communication(self, layer_profiles: List[LayerProfile]) -> float:
        """Estimate communication cost for parameter server architecture."""
        total_param_size = sum(layer.parameter_memory for layer in layer_profiles)
        return total_param_size / (10 * 1024 * 1024 * 1024)  # Assuming 10GB/s bandwidth

    def get_parallelization_recommendation(self, profile: ModelProfile) -> Dict:
        """Generate parallelization recommendations based on model profile."""
        recommendations = {
            'tensor_parallel_candidates': self._identify_tensor_parallel_layers(profile),
            'pipeline_boundaries': self._identify_pipeline_boundaries(profile),
            'data_parallel_strategy': self._recommend_data_parallel_strategy(profile),
            'memory_optimization': self._recommend_memory_optimizations(profile)
        }
        return recommendations

    def _identify_tensor_parallel_layers(self, profile: ModelProfile) -> List[str]:
        """Identify layers that would benefit from tensor parallelism."""
        candidates = []
        for layer in profile.layer_profiles:
            if (layer.compute_intensity > 100 and  # High compute intensity
                layer.parameter_count > 1000000):  # Large parameter count
                candidates.append(layer.name)
        return candidates

    def _identify_pipeline_boundaries(self, profile: ModelProfile) -> List[int]:
        """Identify optimal pipeline stage boundaries."""
        total_flops = sum(layer.forward_flops + layer.backward_flops 
                         for layer in profile.layer_profiles)
        target_flops_per_stage = total_flops / 4  # Assume 4 pipeline stages
        
        boundaries = []
        current_flops = 0
        
        for i, layer in enumerate(profile.layer_profiles):
            current_flops += layer.forward_flops + layer.backward_flops
            if current_flops >= target_flops_per_stage:
                boundaries.append(i)
                current_flops = 0
                
        return boundaries

    def _recommend_data_parallel_strategy(self, profile: ModelProfile) -> Dict:
        """Recommend data parallel strategy based on model characteristics."""
        return {
            'micro_batch_size': self._calculate_optimal_micro_batch(profile),
            'gradient_accumulation_steps': self._calculate_gradient_accumulation(profile),
            'overlap_communication': profile.communication_pattern['all_reduce'] > 0.1
        }

    def _recommend_memory_optimizations(self, profile: ModelProfile) -> Dict:
        """Recommend memory optimization strategies."""
        return {
            'activation_checkpointing': profile.memory_requirement > 32 * (1024**3),
            'cpu_offloading': profile.memory_requirement > 40 * (1024**3),
            'mixed_precision': True,
            'optimizer_state_partitioning': True
        }

    def _calculate_optimal_micro_batch(self, profile: ModelProfile) -> int:
        """Calculate optimal micro batch size based on memory constraints."""
        memory_per_sample = profile.memory_requirement / 32  # Assume base batch size of 32
        available_memory = 32 * (1024**3)  # Assume 32GB GPU memory
        return max(1, int(available_memory / memory_per_sample))

    def _calculate_gradient_accumulation(self, profile: ModelProfile) -> int:
        """Calculate optimal number of gradient accumulation steps."""
        communication_overhead = profile.communication_pattern['all_reduce']
        if communication_overhead > 0.2:  # High communication overhead
            return 8
        elif communication_overhead > 0.1:  # Medium communication overhead
            return 4
        else:
            return 2
