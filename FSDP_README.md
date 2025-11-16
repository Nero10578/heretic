# FSDP Implementation for Heretic

This document describes the Fully Sharded Data Parallel (FSDP) implementation added to the Heretic LLM abliteration tool.

## Overview

FSDP is a distributed training/inference technique that shards model parameters, gradients, and optimizer states across multiple GPUs. This allows for:

1. **Memory Efficiency**: Each GPU only stores a fraction of the model parameters
2. **Scalability**: Can work with larger models across multiple GPUs
3. **Performance**: Reduced communication overhead compared to other parallelism approaches

## Implementation Details

### Configuration

FSDP can be enabled by setting `use_fsdp = true` in the configuration file or via command line:

```bash
heretic --use-fsdp true --model <model_name>
```

### Configuration Options

The following FSDP-specific configuration options are available:

- `use_fsdp`: Enable/disable FSDP (default: false)
- `fsdp_sharding_strategy`: Sharding strategy (default: "FULL_SHARD")
  - Options: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
- `fsdp_offload_params`: Offload parameters to CPU (default: false)
- `fsdp_cpu_offload`: Offload parameters and gradients to CPU (default: false)
- `fsdp_auto_wrap_policy`: Auto-wrapping policy (default: "TRANSFORMER_BASED_WRAP")
  - Options: "TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"
- `fsdp_transformer_layer_cls_to_wrap`: Transformer layer class to wrap (default: "transformers.models.llama.modeling_llama.LlamaDecoderLayer")
- `fsdp_min_num_params`: Minimum number of parameters for size-based wrapping (default: 1000000)

### Memory Management

FSDP-aware memory management functions have been added:

- `fsdp_empty_cache()`: Clears caches with proper synchronization across processes
- `fsdp_print_memory_usage()`: Prints memory usage for all processes in a distributed setting

### Model Initialization

The Model class now supports FSDP initialization:

1. When FSDP is enabled, the model is wrapped with FSDP
2. Abliteration hooks are registered with the underlying model module
3. Batch processing methods handle distributed data splitting and gathering

### Batch Processing

Batch processing methods have been updated for distributed execution:

- `get_responses_batched()`: Splits data across processes and gathers results
- `get_residuals_batched()`: Handles distributed residual computation
- `get_logprobs_batched()`: Distributed log probability computation

### Abliteration

Abliteration hooks work with FSDP-wrapped models by:

1. Accessing the underlying model through `model.module`
2. Applying modifications to the unwrapped model
3. Ensuring proper synchronization across processes

## Usage

### Basic Usage

```bash
# Enable FSDP with default settings
heretic --use-fsdp true --model <model_name>

# Enable FSDP with custom sharding strategy
heretic --use-fsdp true --fsdp-sharding-strategy SHARD_GRAD_OP --model <model_name>

# Enable FSDP with CPU offloading
heretic --use-fsdp true --fsdp-cpu-offload true --model <model_name>
```

### Configuration File

Add the following to your configuration file:

```toml
[fsdp]
use_fsdp = true
fsdp_sharding_strategy = "FULL_SHARD"
fsdp_offload_params = false
fsdp_cpu_offload = false
fsdp_auto_wrap_policy = "TRANSFORMER_BASED_WRAP"
fsdp_transformer_layer_cls_to_wrap = "transformers.models.llama.modeling_llama.LlamaDecoderLayer"
fsdp_min_num_params = 1000000
```

## Performance Considerations

### When to Use FSDP

FSDP is most beneficial when:

1. Working with large models that don't fit on a single GPU
2. Using multiple GPUs with sufficient interconnect bandwidth
3. The model has a large number of parameters

### When Not to Use FSDP

FSDP may not be beneficial when:

1. Working with small models that fit comfortably on a single GPU
2. Using only one GPU
3. The interconnect between GPUs is slow

### Performance Tips

1. **Batch Size**: Use larger batch sizes to maximize GPU utilization
2. **Sharding Strategy**: FULL_SHARD is usually best for inference
3. **CPU Offloading**: Enable if GPU memory is limited
4. **Auto-Wrapping**: TRANSFORMER_BASED_WRAP is recommended for transformer models

## Testing

A test script is provided to verify FSDP functionality:

```bash
python test_fsdp.py
```

This script will:

1. Initialize FSDP with available GPUs
2. Load a small model with FSDP
3. Test basic operations (inference, abliteration)
4. Verify memory management functions

## Troubleshooting

### Common Issues

1. **NCCL Initialization Errors**:
   - Ensure NCCL is properly installed
   - Check firewall settings for inter-GPU communication
   - Verify all GPUs are visible to the process

2. **Out of Memory Errors**:
   - Enable CPU offloading with `fsdp_cpu_offload = true`
   - Reduce batch size
   - Use a more aggressive sharding strategy

3. **Slow Performance**:
   - Check GPU interconnect bandwidth
   - Verify NCCL is using the correct network interface
   - Consider using a different sharding strategy

### Debug Mode

Enable debug mode to get more detailed information:

```bash
heretic --use-fsdp true --debug true --model <model_name>
```

## Implementation Notes

### Distributed Environment

FSDP requires a properly configured distributed environment:

1. Environment variables are automatically set if not present
2. NCCL backend is used for GPU communication
3. Proper cleanup is performed on exit

### Memory Management

FSDP-aware memory management ensures:

1. All processes are synchronized before cache clearing
2. Memory usage is reported for all processes
3. Peak memory statistics are tracked per process

### Model Compatibility

The implementation is designed to work with:

1. Transformer models from the Hugging Face Hub
2. Models with a standard transformer architecture
3. Models that can be loaded with Accelerate

## Future Enhancements

Potential improvements for future versions:

1. Dynamic sharding strategy selection
2. Automatic performance tuning
3. Support for more model architectures
4. Integration with other distributed training frameworks