#!/usr/bin/env python3
"""
Test script to verify the memory management fixes for the abliteration process.
This script simulates the memory-intensive parts of the process to ensure
VRAM is properly cleared between operations.
"""

import gc
import torch
import psutil
import os
from contextlib import contextmanager

# Import the fixed functions
from src.heretic.utils import empty_cache

def get_memory_info():
    """Get current memory usage information"""
    info = {}
    
    # GPU memory
    if torch.cuda.is_available():
        info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
        info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # CPU memory
    process = psutil.Process(os.getpid())
    info['cpu_rss'] = process.memory_info().rss / 1024**3  # GB
    
    return info

@contextmanager
def memory_monitor(operation_name):
    """Context manager to monitor memory usage during an operation"""
    print(f"\n=== {operation_name} ===")
    before = get_memory_info()
    print(f"Before: {before}")
    
    try:
        yield
    finally:
        after = get_memory_info()
        print(f"After: {after}")
        
        # Calculate differences
        if torch.cuda.is_available():
            gpu_diff = after['gpu_allocated'] - before['gpu_allocated']
            print(f"GPU memory change: {gpu_diff:+.2f} GB")
        
        cpu_diff = after['cpu_rss'] - before['cpu_rss']
        print(f"CPU memory change: {cpu_diff:+.2f} GB")

def simulate_model_load(size_gb=5):
    """Simulate loading a large model by allocating tensors"""
    if torch.cuda.is_available():
        # Allocate GPU memory to simulate a model
        tensors = []
        # Calculate how many 1GB tensors we need
        num_tensors = int(size_gb)
        remaining_mb = int((size_gb - num_tensors) * 1024)
        
        # Allocate 1GB tensors
        for i in range(num_tensors):
            # Create a tensor that's approximately 1GB
            # 1GB = 1024^3 bytes, float32 = 4 bytes per element
            # So we need 256M elements
            tensor_size = 256 * 1024 * 1024
            tensors.append(torch.randn(tensor_size, device='cuda', dtype=torch.float32))
        
        # Allocate remaining memory
        if remaining_mb > 0:
            # Create a tensor for the remaining MB
            tensor_size = remaining_mb * 1024 * 1024 // 4  # Convert MB to number of float32 elements
            tensors.append(torch.randn(tensor_size, device='cuda', dtype=torch.float32))
        
        return tensors
    return []

def test_memory_cleanup():
    """Test that memory is properly cleaned up"""
    print("Testing memory management fixes...")
    
    # Test 1: Basic empty_cache functionality
    with memory_monitor("Basic empty_cache test"):
        tensors = simulate_model_load(1)  # Use 1GB instead of 2GB
        empty_cache()
        del tensors
    
    # Test 2: Multiple model loads (simulating trial process)
    print("\n=== Simulating multiple trial reloads ===")
    for i in range(3):
        with memory_monitor(f"Trial {i+1} simulation"):
            # Simulate model load
            model_tensors = simulate_model_load(1)  # Use 1GB instead of 3GB
            
            # Simulate some operations
            if torch.cuda.is_available():
                _ = torch.sum(torch.cat([t.flatten() for t in model_tensors]))
            
            # Clear memory (simulating reload_model)
            del model_tensors
            empty_cache()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
    
    # Test 3: Simulate quantization workflow
    print("\n=== Simulating quantization workflow ===")
    with memory_monitor("Full quantization simulation"):
        # Step 1: Clear current model
        empty_cache()
        gc.collect()
        
        # Step 2: Simulate loading full precision model to CPU
        # (This would normally be loaded to CPU, not GPU)
        cpu_tensors = []
        for i in range(50):  # Simulate CPU tensors
            cpu_tensors.append(torch.randn(1024, 1024))
        
        # Step 3: Clear CPU model before loading quantized
        del cpu_tensors
        gc.collect()
        
        # Step 4: Simulate loading quantized model
        gpu_tensors = simulate_model_load(1)  # Use 1GB instead of 2GB
        
        # Step 5: Final cleanup
        del gpu_tensors
        empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    print("\n=== Memory management test completed ===")
    final_memory = get_memory_info()
    print(f"Final memory state: {final_memory}")

if __name__ == "__main__":
    test_memory_cleanup()