"""
Model Profiler for Dynamic Distributed Training
Analyzes model characteristics and computational patterns
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from dataclasses import dataclass
import time
import logging
from enum import Enum
from collections import defaultdict
import networkx as nx
import json
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
import threading
from queue import Queue

class LayerType(Enum):
    """Types of neural network layers."""
    LINEAR = "linear"
    CONV = "conv"
    ATTENTION = "attention"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    EMBEDDING = "embedding"
    DROPOUT = "dropout"
    CONTAINER = "container"
    OTHER = "other"

@dataclass
class LayerProfile:
    """Profile information for a single layer."""
    name: str
    type: LayerType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameter_count: int
    macs: int  # Multiply-accumulate operations
    memory_footprint: int  # Bytes
    forward_time: float  # milliseconds
    backward_time: float  # milliseconds
    recomputation_cost: float  # Relative cost of recomputing vs storing
    communication_volume: int  # Bytes
    dependencies: Set[str]  # Layer dependencies

@dataclass
class ModelCharacteristics:
    """Overall model characteristics."""
    total_parameters: int
    total_layers: int
    model_size_bytes: int
    activation_size_bytes: int
    layer_distribution: Dict[LayerType, int]
    computational_graph: nx.DiGraph
    memory_peaks: Dict[str, int]
    communication_patterns: Dict[str, Any]
    pipeline_stages: Optional[List[List[str]]]
    tensor_parallel_groups: Optional[List[List[str]]]

class ModelAnalyzer:
    """Analyzes model architecture and computational patterns."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.graph_module = self._create_graph_module()
        self.computational_graph = nx.DiGraph()
        self.layer_profiles: Dict[str, LayerProfile] = {}
        self.logger = logging.getLogger(__name__)
        
    def _create_graph_module(self) -> GraphModule:
        """Create a GraphModule for analysis."""
        try:
            return symbolic_trace(self.model)
        except Exception as e:
            self.logger.warning(f"Symbolic tracing failed: {e}. Using fallback analysis.")
            return None
            
    def analyze_model(self) -> ModelCharacteristics:
        """Perform comprehensive model analysis."""
        self._build_computational_graph()
        self._analyze_layers()
        self._analyze_memory_patterns()
        self._analyze_communication_patterns()
        
        return ModelCharacteristics(
            total_parameters=self._count_parameters(),
            total_layers=len(self.layer_profiles),
            model_size_bytes=self._calculate_model_size(),
            activation_size_bytes=self._estimate_activation_size(),
            layer_distribution=self._get_layer_distribution(),
            computational_graph=self.computational_graph,
            memory_peaks=self._find_memory_peaks(),
            communication_patterns=self._get_communication_patterns(),
            pipeline_stages=self._suggest_pipeline_stages(),
            tensor_parallel_groups=self._suggest_tensor_parallel_groups()
        )
        
    def _build_computational_graph(self):
        """Build computational dependency graph."""
        if self.graph_module is not None:
            self._build_graph_from_fx()
        else:
            self._build_graph_from_hierarchy()
            
    def _build_graph_from_fx(self):
        """Build graph using FX graph representation."""
        for node in self.graph_module.graph.nodes:
            self.computational_graph.add_node(node.name, node=node)
            for input_node in node.all_input_nodes:
                self.computational_graph.add_edge(input_node.name, node.name)
                
    def _build_graph_from_hierarchy(self):
        """Build graph from module hierarchy (fallback)."""
        def _add_module(name: str, module: nn.Module):
            self.computational_graph.add_node(name, module=module)
            for child_name, child_module in module.named_children():
                child_full_name = f"{name}.{child_name}"
                _add_module(child_full_name, child_module)
                self.computational_graph.add_edge(name, child_full_name)
                
        _add_module("model", self.model)
        
    def _analyze_layers(self):
        """Analyze individual layers."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                self.layer_profiles[name] = self._profile_layer(name, module)
                
    def _profile_layer(self, name: str, layer: nn.Module) -> LayerProfile:
        """Profile a single layer."""
        layer_type = self._determine_layer_type(layer)
        input_shape = self._estimate_input_shape(layer)
        output_shape = self._estimate_output_shape(layer, input_shape)
        
        return LayerProfile(
            name=name,
            type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_count=sum(p.numel() for p in layer.parameters()),
            macs=self._estimate_macs(layer, input_shape),
            memory_footprint=self._estimate_memory_footprint(layer),
            forward_time=self._measure_forward_time(layer),
            backward_time=self._measure_backward_time(layer),
            recomputation_cost=self._estimate_recomputation_cost(layer),
            communication_volume=self._estimate_communication_volume(layer),
            dependencies=self._get_layer_dependencies(name)
        )
        
    def _determine_layer_type(self, layer: nn.Module) -> LayerType:
        """Determine the type of a layer."""
        if isinstance(layer, nn.Linear):
            return LayerType.LINEAR
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return LayerType.CONV
        elif any(attention_name in layer.__class__.__name__.lower() 
                for attention_name in ['attention', 'self', 'multi']):
            return LayerType.ATTENTION
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            return LayerType.NORMALIZATION
        elif isinstance(layer, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            return LayerType.ACTIVATION
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool2d)):
            return LayerType.POOLING
        elif isinstance(layer, nn.Embedding):
            return LayerType.EMBEDDING
        elif isinstance(layer, nn.Dropout):
            return LayerType.DROPOUT
        elif isinstance(layer, (nn.Sequential, nn.ModuleList)):
            return LayerType.CONTAINER
        else:
            return LayerType.OTHER
            
    def _estimate_input_shape(self, layer: nn.Module) -> Tuple[int, ...]:
        """Estimate input shape of a layer."""
        # This would need to be implemented based on model architecture
        return (0,)  # Placeholder
        
    def _estimate_output_shape(self, layer: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Estimate output shape of a layer."""
        # This would need to be implemented based on layer type
        return (0,)  # Placeholder
        
    def _estimate_macs(self, layer: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate multiply-accumulate operations for a layer."""
        layer_type = self._determine_layer_type(layer)
        
        if layer_type == LayerType.LINEAR:
            return input_shape[0] * layer.in_features * layer.out_features
        elif layer_type == LayerType.CONV:
            # Would need proper implementation for conv layers
            return 0
        elif layer_type == LayerType.ATTENTION:
            # Would need proper implementation for attention layers
            return 0
        return 0
        
    def _estimate_memory_footprint(self, layer: nn.Module) -> int:
        """Estimate memory footprint of a layer."""
        param_memory = sum(p.numel() * p.element_size() for p in layer.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in layer.buffers())
        return param_memory + buffer_memory
        
    def _measure_forward_time(self, layer: nn.Module) -> float:
        """Measure forward pass time of a layer."""
        # Would need proper implementation with warm-up and multiple runs
        return 0.0
        
    def _measure_backward_time(self, layer: nn.Module) -> float:
        """Measure backward pass time of a layer."""
        # Would need proper implementation with warm-up and multiple runs
        return 0.0
        
    def _estimate_recomputation_cost(self, layer: nn.Module) -> float:
        """Estimate the cost of recomputing vs storing activations."""
        forward_flops = self._estimate_macs(layer, self._estimate_input_shape(layer))
        memory_saved = self._estimate_memory_footprint(layer)
        return forward_flops / max(1, memory_saved)
        
    def _estimate_communication_volume(self, layer: nn.Module) -> int:
        """Estimate communication volume for layer parameters."""
        return sum(p.numel() * p.element_size() for p in layer.parameters())
        
    def _get_layer_dependencies(self, layer_name: str) -> Set[str]:
        """Get dependencies of a layer."""
        dependencies = set()
        if layer_name in self.computational_graph:
            dependencies.update(pred for pred in self.computational_graph.predecessors(layer_name))
        return dependencies
        
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())
        
    def _calculate_model_size(self) -> int:
        """Calculate total model size in bytes."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())
        
    def _estimate_activation_size(self) -> int:
        """Estimate peak activation memory."""
        return sum(profile.memory_footprint for profile in self.layer_profiles.values())
        
    def _get_layer_distribution(self) -> Dict[LayerType, int]:
        """Get distribution of layer types."""
        distribution = defaultdict(int)
        for profile in self.layer_profiles.values():
            distribution[profile.type] += 1
        return dict(distribution)
        
    def _find_memory_peaks(self) -> Dict[str, int]:
        """Find memory usage peaks."""
        peaks = {}
        current_memory = 0
        
        for node in nx.topological_sort(self.computational_graph):
            if node in self.layer_profiles:
                profile = self.layer_profiles[node]
                current_memory += profile.memory_footprint
                peaks[node] = current_memory
                
                # Simulate memory release for nodes whose outputs are no longer needed
                for pred in self.computational_graph.predecessors(node):
                    if pred in self.layer_profiles:
                        pred_profile = self.layer_profiles[pred]
                        if not any(succ != node for succ in self.computational_graph.successors(pred)):
                            current_memory -= pred_profile.memory_footprint
                            
        return peaks
        
    def _get_communication_patterns(self) -> Dict[str, Any]:
        """Analyze communication patterns."""
        patterns = {
            'all_reduce_volume': self._estimate_gradient_sync_volume(),
            'pipeline_volume': self._estimate_pipeline_communication(),
            'tensor_parallel_volume': self._estimate_tensor_parallel_communication()
        }
        return patterns
        
    def _estimate_gradient_sync_volume(self) -> int:
        """Estimate gradient synchronization volume."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())
        
    def _estimate_pipeline_communication(self) -> int:
        """Estimate pipeline communication volume."""
        # This would need proper implementation based on pipeline configuration
        return 0
        
    def _estimate_tensor_parallel_communication(self) -> int:
        """Estimate tensor parallel communication volume."""
        # This would need proper implementation based on tensor parallel configuration
        return 0
        
    def _suggest_pipeline_stages(self) -> List[List[str]]:
        """Suggest optimal pipeline stage division."""
        if not self.computational_graph:
            return None
            
        # Use layer profiles to balance computation across stages
        sorted_layers = list(nx.topological_sort(self.computational_graph))
        layer_costs = [
            self.layer_profiles[layer].forward_time + self.layer_profiles[layer].backward_time
            for layer in sorted_layers if layer in self.layer_profiles
        ]
        
        # Dynamic programming to find optimal split points
        return self._balance_pipeline_stages(sorted_layers, layer_costs)
        
    def _balance_pipeline_stages(self, layers: List[str], costs: List[float]) -> List[List[str]]:
        """Balance layers across pipeline stages using dynamic programming."""
        # This would need proper implementation of stage balancing algorithm
        return [layers]  # Placeholder
        
    def _suggest_tensor_parallel_groups(self) -> List[List[str]]:
        """Suggest tensor parallel layer groups."""
        if not self.computational_graph:
            return None
            
        # Group layers that can benefit from tensor parallelism
        parallel_groups = []
        current_group = []
        
        for layer_name, profile in self.layer_profiles.items():
            if profile.type in [LayerType.LINEAR, LayerType.CONV]:
                if len(current_group) == 0 or self._can_group_layers(current_group[-1], layer_name):
                    current_group.append(layer_name)
                else:
                    if len(current_group) > 0:
                        parallel_groups.append(current_group)
                    current_group = [layer_name]
                    
        if len(current_group) > 0:
            parallel_groups.append(current_group)
            
        return parallel_groups
        
    def _can_group_layers(self, layer1: str, layer2: str) -> bool:
        """Check if two layers can be grouped for tensor parallelism."""
        # This would need proper implementation based on layer dependencies
        return True  # Placeholder
        
    def export_analysis(self, filepath: str):
        """Export analysis results to a file."""
        analysis_data = {
            'layer_profiles': {
                name: {
                    'type': profile.type.value,
                    'input_shape': profile.input_shape,
                    'output_shape': profile.output_shape,
                    'parameter_count': profile.parameter_count,
                    'macs': profile.macs,
                    'memory_footprint': profile.memory_footprint,
                    'forward_time': profile.forward_time,
                    'backward_time': profile.backward_time,
                    'recomputation_cost': profile.recomputation_cost,
                    'communication_volume': profile.communication_volume,
                    'dependencies': list(profile.dependencies)
                }
                for name, profile in self.layer_profiles.items()
            },
            'model_characteristics': {
                'total_parameters': self._count_parameters(),
                'total_layers': len(self.layer_profiles),
                'model_size_bytes': self._calculate_model_size(),
                'activation_size_bytes': self._estimate_activation_size(),
                'layer_distribution': {k.value: v for k, v in self._get_layer_distribution().items()},
                'memory_peaks': self._find_memory_peaks(),
                'communication_patterns': self._get_communication_patterns()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
