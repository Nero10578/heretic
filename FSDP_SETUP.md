# FSDP Setup Guide for Heretic

This guide explains how to set up your environment to use FSDP (Fully Sharded Data Parallel) with Heretic.

## Prerequisites

### 1. Hardware Requirements

- **Multiple GPUs**: FSDP is most beneficial with 2 or more GPUs
- **NVLink or High-Speed Interconnect**: For optimal performance between GPUs
- **Sufficient GPU Memory**: At least 8GB per GPU recommended

### 2. Software Requirements

#### PyTorch with CUDA Support
FSDP requires PyTorch with CUDA support. Install with:

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### NCCL (NVIDIA Collective Communications Library)
FSDP uses NCCL for GPU-to-GPU communication:

```bash
# Usually installed with PyTorch, but you can verify:
python -c "import torch; print(torch.cuda.nccl.version())"
```

If NCCL is not available, install it:
```bash
# Ubuntu/Debian
sudo apt-get install libnccl2 libnccl2-dev

# Or download from NVIDIA: https://developer.nvidia.com/nccl/nccl-download
```

#### Verify Installation
Check if your environment supports FSDP:

```python
import torch
import torch.distributed as dist

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"NCCL available: {torch.distributed.is_nccl_available()}")
```

## Installation

### 1. Install Heretic with FSDP Support

```bash
# Clone the repository
git clone https://github.com/p-e-w/heretic.git
cd heretic

# Install with FSDP dependencies
pip install -e ".[fsdp]"
```

### 2. Alternative: Manual Installation

If the above doesn't work, install dependencies manually:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install accelerate>=1.10.0
pip install bitsandbytes>=0.43.0
pip install datasets>=4.0.0
pip install hf-transfer>=0.1.9
pip install huggingface-hub>=0.34.4
pip install optuna>=4.5.0
pip install pydantic-settings>=2.10.1
pip install questionary>=2.1.1
pip install rich>=14.1.0
pip install torchao==0.13.0
pip install transformers>=4.55.2
```

## Running with FSDP

### 1. Single Node, Multiple GPUs

For a single machine with multiple GPUs:

```bash
# Enable FSDP with default settings
heretic --use-fsdp true --model <model_name>

# Or use configuration file
heretic --config config_with_fsdp.toml
```

### 2. Multiple Nodes (Advanced)

For multiple machines, you need to set up SSH access and run:

```bash
# On each node, run with appropriate environment variables
RANK=<node_rank> WORLD_SIZE=<total_nodes> MASTER_ADDR=<head_node_ip> MASTER_PORT=<port> \
heretic --use-fsdp true --model <model_name>
```

### 3. Using torchrun (Recommended)

For better process management, use torchrun:

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=<num_gpus> --master_port=<port> \
-m heretic.main --use-fsdp true --model <model_name>

# Example with 4 GPUs
torchrun --nproc_per_node=4 --master_port=29500 \
-m heretic.main --use-fsdp true --model <model_name>
```

## Configuration

### 1. Basic Configuration

Create a configuration file (`config.toml`) with FSDP settings:

```toml
[fsdp]
use_fsdp = true
fsdp_sharding_strategy = "FULL_SHARD"
fsdp_offload_params = false
fsdp_cpu_offload = false
fsdp_auto_wrap_policy = "TRANSFORMER_BASED_WRAP"
fsdp_transformer_layer_cls_to_wrap = "transformers.models.llama.modeling_llama.LlamaDecoderLayer"
fsdp_min_num_params = 1000000

[model]
model = "meta-llama/Llama-2-7b-chat-hf"
batch_size = 4
```

### 2. Memory-Constrained Configuration

For systems with limited GPU memory:

```toml
[fsdp]
use_fsdp = true
fsdp_sharding_strategy = "FULL_SHARD"
fsdp_offload_params = true
fsdp_cpu_offload = true
fsdp_auto_wrap_policy = "SIZE_BASED_WRAP"
fsdp_min_num_params = 500000
```

## Troubleshooting

### 1. Common Issues

#### NCCL Initialization Errors
```
RuntimeError: NCCL error in: ../src/nccl.c
```

**Solutions:**
- Check NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Verify GPU connectivity: `nvidia-smi topo -m`
- Disable firewall between GPUs
- Try different NCCL settings:
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=1  # Disable InfiniBand
  export NCCL_P2P_DISABLE=1  # Disable P2P
  ```

#### Out of Memory Errors
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Enable CPU offloading: `fsdp_cpu_offload = true`
- Reduce batch size
- Use more aggressive sharding: `fsdp_sharding_strategy = "FULL_SHARD"`
- Use smaller model or quantization

#### Slow Performance
**Solutions:**
- Check GPU interconnect: `nvidia-smi topo -m`
- Verify NVLink is being used
- Try different sharding strategies
- Increase batch size for better GPU utilization

### 2. Debug Mode

Enable debug mode for more information:

```bash
heretic --use-fsdp true --debug true --model <model_name>
```

### 3. Performance Tuning

#### Optimal Batch Size
Test different batch sizes to find the sweet spot:

```bash
# Test with different batch sizes
for bs in 1 2 4 8 16; do
    echo "Testing batch size: $bs"
    heretic --use-fsdp true --batch-size $bs --model <model_name>
done
```

#### Memory Monitoring
Monitor GPU memory usage:

```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Or use Python script
python -c "
import torch
while True:
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB')
    time.sleep(1)
"
```

## Testing Your Setup

### 1. Run the Test Script

```bash
python test_fsdp.py
```

This will:
- Check FSDP initialization
- Load a small model with FSDP
- Test basic operations
- Verify memory management

### 2. Expected Output

Successful setup should show:
```
FSDP Test Script for Heretic
==================================================
Found 4 GPU(s)
* FSDP initialized with 4 processes
Loading model with FSDP...
* Model loaded successfully with FSDP
Testing basic operations...
Testing FSDP-aware memory management...
Rank 0, GPU 0: Allocated: X.XXGB, Reserved: Y.YYGB, Peak: Z.ZZGB
...
```

## Performance Comparison

### Without FSDP
- Model loaded on a single GPU
- Limited by GPU memory
- Slower inference for large models

### With FSDP
- Model sharded across multiple GPUs
- Can handle larger models
- Faster inference for large batches
- Better GPU utilization

## Advanced Usage

### 1. Custom Sharding Strategy

For specific use cases, you might want to use different sharding:

```toml
[fsdp]
# For models with lots of small layers
fsdp_sharding_strategy = "SHARD_GRAD_OP"

# For minimal memory usage
fsdp_sharding_strategy = "FULL_SHARD"

# For debugging
fsdp_sharding_strategy = "NO_SHARD"
```

### 2. Hybrid CPU-GPU Offloading

For very large models:

```toml
[fsdp]
use_fsdp = true
fsdp_sharding_strategy = "FULL_SHARD"
fsdp_offload_params = true
fsdp_cpu_offload = true
fsdp_auto_wrap_policy = "SIZE_BASED_WRAP"
fsdp_min_num_params = 100000
```

## Getting Help

If you encounter issues:

1. Check the [FSDP_README.md](FSDP_README.md) for implementation details
2. Run the test script to isolate the problem
3. Enable debug mode for detailed logs
4. Check GPU memory and interconnect
5. Verify NCCL installation and configuration

## Summary

To use FSDP with Heretic:

1. Ensure you have multiple GPUs with NCCL support
2. Install PyTorch with CUDA and NCCL
3. Install Heretic with FSDP dependencies
4. Configure FSDP settings in your config file
5. Run with `--use-fsdp true` or use torchrun for multiple processes

FSDP will significantly improve performance when working with large models on multiple GPUs, addressing the original concern about slow inference with only the Accelerate library.