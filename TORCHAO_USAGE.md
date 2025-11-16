# Using TorchAO Quantization with Heretic

This document explains how to use torchao quantization with the heretic LM model ablation repository.

## Overview

TorchAO is a PyTorch architecture optimization library that provides various quantization schemes for language models. Heretic now supports torchao quantization as an alternative to bitsandbytes quantization.

## Installation

### Option 1: Using uv (Recommended)

If you're using uv (the package manager used by heretic), you can install torchao as an optional dependency:

```bash
# Install heretic with torchao support
uv pip install -e .[torchao]

# Or install torchao separately
uv pip install torchao>=0.10.0
```

### Option 2: Using pip

```bash
pip install torchao transformers
```

## Configuration

To use torchao quantization, you need to configure it in your `config.toml` file. Here are the key options:

### Basic Configuration

```toml
# Enable torchao quantization
use_torchao = true

# Set the quantization type
torchao_quant_type = "int4_weight_only"

# Group size for weight-only quantization
torchao_group_size = 128

# Whether to include embedding layers in quantization
torchao_include_embedding = false
```

### Quantization Types

The following quantization types are supported:

1. **int4_weight_only** - 4-bit weight-only quantization (default)
   - Good for: CUDA GPUs, general use
   - Memory savings: ~75% reduction

2. **int8_weight_only** - 8-bit weight-only quantization
   - Good for: CUDA GPUs, when 4-bit is too aggressive
   - Memory savings: ~50% reduction

3. **int8_dynamic_activation_int8_weight** - 8-bit dynamic activation and weight quantization
   - Good for: A100/H100 GPUs, balanced performance
   - Memory savings: ~50% reduction

4. **float8_dynamic_activation_float8_weight** - 8-bit float dynamic activation and weight quantization
   - Good for: H100 GPUs, high performance
   - Memory savings: ~50% reduction

5. **float8_weight_only** - 8-bit float weight-only quantization
   - Good for: H100 GPUs, high performance
   - Memory savings: ~50% reduction

6. **autoquant** - Automatically select best quantization
   - Good for: When unsure which to use
   - Note: GPU only, requires micro-benchmarking

### Device-Specific Options

For different devices, torchao automatically selects appropriate layouts:

- **CUDA**: Default layout (optimized for NVIDIA GPUs)
- **CPU**: Int4CPULayout for int4_weight_only
- **XPU**: Int4XPULayout for int4_weight_only

## Example Configurations

### For 4-bit Quantization (Most Memory Efficient)

```toml
use_torchao = true
torchao_quant_type = "int4_weight_only"
torchao_group_size = 128
torchao_include_embedding = false
```

### For 8-bit Quantization (Good Balance)

```toml
use_torchao = true
torchao_quant_type = "int8_weight_only"
torchao_include_embedding = false
```

### For H100 GPUs (High Performance)

```toml
use_torchao = true
torchao_quant_type = "float8_dynamic_activation_float8_weight"
torchao_include_embedding = false
```

### For CPU Inference

```toml
use_torchao = true
torchao_quant_type = "int4_weight_only"
torchao_group_size = 128
device_map = "cpu"
torchao_include_embedding = false
```

## Usage

Once configured, use heretic exactly as you would with bitsandbytes quantization:

```bash
python -m heretic.main
```

The quantization is automatically applied during model loading and abliteration. No additional steps are required.

## Comparison with Bitsandbytes

| Feature | Bitsandbytes | TorchAO |
|----------|--------------|----------|
| 4-bit quantization | ✓ | ✓ |
| 8-bit quantization | ✓ | ✓ |
| Float8 quantization | ✗ | ✓ |
| Auto-quantization | ✗ | ✓ |
| Sparse quantization | ✗ | ✓ |
| Device-specific optimizations | Limited | ✓ |
| Integration with torch.compile | Limited | ✓ |

## Troubleshooting

### Import Errors

If you get import errors for torchao:

```bash
pip install torchao transformers
```

### Memory Issues

If you run out of memory:

1. Try a less aggressive quantization (e.g., int8 instead of int4)
2. Reduce batch size in config
3. Use CPU device_map if GPU memory is limited

### Performance Issues

For best performance:

1. Use `cache_implementation="static"` in generation calls
2. Choose the right quantization type for your hardware
3. Consider autoquant if unsure

## Advanced Usage

### Per-Module Quantization

For advanced users, you can skip quantization for specific layers by modifying the `_create_torchao_config` method in `src/heretic/model.py` to use `ModuleFqnToConfig`.

### Custom Layouts

You can also specify custom layouts for specific hardware by modifying the config creation logic.

## Notes

1. torchao requires `safe_serialization=False` when saving models
2. Some quantization types are hardware-specific
3. Float8 quantization requires recent GPU hardware
4. Autoquant requires GPU and performs micro-benchmarking

For more information about torchao, see: https://github.com/pytorch/ao