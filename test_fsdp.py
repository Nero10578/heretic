#!/usr/bin/env python3
"""
Test script to verify FSDP implementation in heretic.
This script can be used to test if FSDP is working correctly with multiple GPUs.
"""

import os
import sys
import torch
import torch.distributed as dist

# Add src to path to import heretic modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from heretic.config import Settings
try:
    from heretic.model import Model
except ImportError as e:
    print(f"[red]Failed to import Model: {e}[/]")
    print("[yellow]Please ensure you have installed heretic with FSDP support:[/]")
    print("pip install -e '.[fsdp]'")
    sys.exit(1)
from heretic.utils import fsdp_print_memory_usage, fsdp_empty_cache


def test_fsdp_initialization():
    """Test FSDP initialization with a small model."""
    print("Testing FSDP initialization...")
    
    # Set environment variables for distributed training
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    
    try:
        dist.init_process_group(backend="nccl")
        print(f"* FSDP initialized with {dist.get_world_size()} processes")
        return True
    except Exception as e:
        print(f"[red]Failed to initialize FSDP: {e}[/]")
        return False


def test_model_with_fsdp():
    """Test model loading and basic operations with FSDP."""
    try:
        # Create settings with FSDP enabled
        settings = Settings()
        settings.use_fsdp = True
        settings.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for testing
        
        # Initialize model with FSDP
        print("Loading model with FSDP...")
        model = Model(settings)
        print("* Model loaded successfully with FSDP")
        
        # Test basic operations
        print("Testing basic operations...")
        test_prompts = ["Hello, how are you?", "What is the capital of France?"]
        
        # Test memory management
        print("Testing FSDP-aware memory management...")
        fsdp_print_memory_usage("Before operations: ")
        
        # Test batch processing
        print("Testing batch processing...")
        responses = model.get_responses_batched(test_prompts)
        print(f"* Got {len(responses)} responses")
        
        # Test memory management after operations
        fsdp_print_memory_usage("After operations: ")
        fsdp_empty_cache()
        print("* Cache cleared successfully")
        
        # Test abliteration
        print("Testing abliteration with FSDP...")
        # Create dummy refusal directions for testing
        layers = model.get_layers()
        dummy_directions = torch.randn(len(layers), 1, 4096)  # Adjust size as needed
        
        # Create dummy parameters
        from heretic.model import AbliterationParameters
        dummy_params = {
            component: AbliterationParameters(
                max_weight=1.0,
                max_weight_position=len(layers) - 1,
                min_weight=0.0,
                min_weight_distance=1.0,
            )
            for component in model.get_abliterable_components()
        }
        
        # Apply abliteration
        model.abliterate(dummy_directions, None, dummy_params)
        print("* Abliteration applied successfully")
        
        # Test responses after abliteration
        responses_after = model.get_responses_batched(test_prompts)
        print(f"* Got {len(responses_after)} responses after abliteration")
        
        return True
        
    except Exception as e:
        print(f"[red]Error during FSDP testing: {e}[/]")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("FSDP Test Script for Heretic")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("[red]CUDA is not available. FSDP requires CUDA.[/]")
        return False
    
    # Check if multiple GPUs are available
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s)")
    
    if gpu_count < 2:
        print("[yellow]FSDP is most beneficial with multiple GPUs. Testing with single GPU.[/]")
    
    # Test FSDP initialization
    if not test_fsdp_initialization():
        return False
    
    try:
        # Test model operations with FSDP
        success = test_model_with_fsdp()
        
        if success:
            print("\n[green]FSDP test completed successfully![/]")
        else:
            print("\n[red]FSDP test failed.[/]")
            
        return success
        
    finally:
        # Clean up distributed environment
        if dist.is_initialized():
            dist.destroy_process_group()
            print("* FSDP distributed environment cleaned up")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)