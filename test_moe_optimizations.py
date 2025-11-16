# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Test script to validate MoE optimizations for heretic.

This script compares the performance of the original and optimized MoE implementations
to demonstrate the speedup achieved by borrowing techniques from VLLM.
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, List

from .config import Settings
from .model import Model
from .optimized_moe import OptimizedMoEHandler, optimized_fused_moe


def create_test_moe_model(hidden_dim: int = 4096, num_experts: int = 8, intermediate_size: int = 16384):
    """
    Create a test MoE model for benchmarking.
    
    Args:
        hidden_dim: Hidden dimension of the model
        num_experts: Number of experts
        intermediate_size: Intermediate size for each expert
        
    Returns:
        Dictionary with expert weights
    """
    # Create expert weights
    expert_weights = {}
    
    # First layer (gate/up projections)
    w1 = torch.randn(num_experts, intermediate_size * 2, hidden_dim)
    for i in range(num_experts):
        expert_weights[f"expert_{i}_w1"] = w1[i]
    
    # Second layer (down projections)
    w2 = torch.randn(num_experts, hidden_dim, intermediate_size)
    for i in range(num_experts):
        expert_weights[f"expert_{i}_w2"] = w2[i]
    
    return expert_weights


def benchmark_moe_implementations(
    model: Model,
    test_input: torch.Tensor,
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark different MoE implementations.
    
    Args:
        model: The model to test
        test_input: Test input tensor
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    # Test original implementation
    print("Benchmarking original MoE implementation...")
    original_times = []
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        # Simulate original MoE processing (sequential)
        with torch.no_grad():
            # Get expert weights from first layer
            layer = model.get_layers()[0]
            if hasattr(layer.mlp, 'experts'):
                expert_weights = [layer.mlp.experts[j].w1.weight if hasattr(layer.mlp.experts[j], 'w1') else layer.mlp.experts[j].gate_proj.weight for j in range(len(layer.mlp.experts))]
                w2_weights = [layer.mlp.experts[j].w2.weight if hasattr(layer.mlp.experts[j], 'w2') else layer.mlp.experts[j].down_proj.weight for j in range(len(layer.mlp.experts))]
                
                # Process each expert sequentially (original approach)
                output = torch.zeros_like(test_input)
                for expert_w1, expert_w2 in zip(expert_weights, w2_weights):
                    # Gate and up projection
                    gate_up = F.linear(test_input, expert_w1)
                    gate, up = gate_up.chunk(2, dim=-1)
                    activated = F.silu(gate) * up
                    
                    # Down projection
                    expert_output = F.linear(activated, expert_w2)
                    output += expert_output
                
                # Average the outputs (simplified routing)
                output = output / len(expert_weights)
        
        end_time = time.perf_counter()
        original_times.append(end_time - start_time)
    
    original_avg_time = sum(original_times) / len(original_times)
    results['original_sequential'] = original_avg_time
    
    # Test optimized implementation
    print("Benchmarking optimized MoE implementation...")
    optimized_times = []
    
    # Create optimized handler
    handler = OptimizedMoEHandler(block_size=64, enable_profiling=True)
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Get expert weights from first layer
            layer = model.get_layers()[0]
            if hasattr(layer.mlp, 'experts'):
                expert_weights = [layer.mlp.experts[j].w1.weight if hasattr(layer.mlp.experts[j], 'w1') else layer.mlp.experts[j].gate_proj.weight for j in range(len(layer.mlp.experts))]
                w2_weights = [layer.mlp.experts[j].w2.weight if hasattr(layer.mlp.experts[j], 'w2') else layer.mlp.experts[j].down_proj.weight for j in range(len(layer.mlp.experts))]
                
                # Use optimized MoE processing
                output = optimized_fused_moe(
                    test_input,
                    expert_weights,
                    None,  # No routing weights for simplicity
                    top_k=1,  # Process all experts
                    block_size=64,
                    enable_profiling=False,
                )
        
        end_time = time.perf_counter()
        optimized_times.append(end_time - start_time)
    
    optimized_avg_time = sum(optimized_times) / len(optimized_times)
    results['optimized_parallel'] = optimized_avg_time
    
    # Calculate speedup
    speedup = original_avg_time / optimized_avg_time
    results['speedup'] = speedup
    
    # Print profiling data if available
    if hasattr(handler, 'profiling_data'):
        print(f"Optimized MoE profiling data: {handler.profiling_data}")
    
    return results


def test_glm_specific_optimizations():
    """
    Test GLM-specific optimizations (shared experts).
    """
    print("Testing GLM-specific optimizations...")
    
    # Create a simple GLM-like model with shared experts
    class MockGLMModel:
        def __init__(self):
            self.layers = []
            
            # Create mock layers with shared and regular experts
            for i in range(4):  # 4 layers
                layer = MockGLMLayer()
                self.layers.append(layer)
    
        def get_layers(self):
            return self.layers
    
    class MockGLMLayer:
        def __init__(self):
            # Mock MLP with both shared and regular experts
            self.mlp = MockGLMMLP()
            
    class MockGLMMLP:
        def __init__(self):
            # Mock shared experts (always active)
            self.shared_experts = MockSharedExperts()
            
            # Mock regular experts
            self.experts = [MockExpert() for _ in range(4)]
            
    class MockSharedExperts:
        def __call__(self, x):
            # Simple linear transformation for shared experts
            return F.linear(x, torch.randn(x.shape[-1], x.shape[-1]))
    
    class MockExpert:
        def __init__(self):
            self.w1 = torch.randn(16384, 4096)
            self.w2 = torch.randn(4096, 16384)
            
        def __call__(self, x):
            # Standard expert forward
            gate_up = F.linear(x, self.w1)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            return F.linear(activated, self.w2)
    
    # Create test model
    model = MockGLMModel()
    
    # Test with and without optimizations
    test_input = torch.randn(1, 128, 4096)
    
    # Without optimization
    start_time = time.perf_counter()
    output = model.layers[0].mlp(test_input)
    end_time = time.perf_counter()
    unoptimized_time = end_time - start_time
    
    # With optimization
    from .optimized_moe import optimize_for_glm_model
    optimize_for_glm_model(model, Settings())
    
    start_time = time.perf_counter()
    output = model.layers[0].mlp(test_input)
    end_time = time.perf_counter()
    optimized_time = end_time - start_time
    
    speedup = unoptimized_time / optimized_time
    print(f"GLM optimization speedup: {speedup:.2f}x")
    
    return speedup > 1.0


def main():
    """
    Main test function.
    """
    print("MoE Optimization Validation for Heretic")
    print("=" * 50)
    
    # Test with synthetic data
    test_input = torch.randn(32, 512, 4096)
    expert_weights = create_test_moe_model(4096, 8, 16384)
    
    # Benchmark implementations
    results = benchmark_moe_implementations(None, test_input, 5)
    
    print("\nBenchmark Results:")
    print(f"Original sequential implementation: {results['original_sequential']:.4f}s")
    print(f"Optimized parallel implementation: {results['optimized_parallel']:.4f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    
    # Test GLM-specific optimizations
    glm_speedup = test_glm_specific_optimizations()
    
    print(f"\nGLM-specific optimization speedup: {glm_speedup:.2f}x")
    
    print("\n" + "=" * 50)
    print("Optimization Summary:")
    print("1. Token alignment and batching improves GPU utilization")
    print("2. Block-wise operations reduce memory overhead")
    print("3. Vectorized operations replace sequential loops")
    print("4. Specialized handling for shared experts (GLM)")
    print("5. Quantization support for memory efficiency")
    
    if results['speedup'] > 1.5:
        print("✅ Significant performance improvement achieved!")
    elif results['speedup'] > 1.2:
        print("✅ Good performance improvement achieved!")
    elif results['speedup'] > 1.0:
        print("✅ Modest performance improvement achieved!")
    else:
        print("⚠️  Limited or no improvement detected")


if __name__ == "__main__":
    main()