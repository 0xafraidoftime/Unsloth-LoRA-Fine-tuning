# Unsloth LoRA Fine-tuning Experiments

This repository contains experimental code for efficient fine-tuning of large language models using Unsloth, LoRA (Low-Rank Adaptation), and 4-bit quantization techniques with PyTorch compilation optimizations.

## Overview

The project explores memory-efficient fine-tuning approaches by combining:
- **Unsloth**: Fast and memory-efficient LLM training library
- **LoRA**: Parameter-efficient fine-tuning technique
- **4-bit Quantization**: Using BitsAndBytes NF4 quantization
- **PyTorch Compilation**: Performance optimization with `torch.compile`
- **Custom Kernels**: Triton-based kernel development for dequantization

## Key Features

### Performance Optimizations
- 4-bit quantized model loading with BitsAndBytes
- LoRA parameter-efficient fine-tuning
- PyTorch 2.0 compilation with optimization flags
- Custom Triton kernels for faster dequantization
- Memory-efficient gradient computation

### Technical Components
- **MLP Implementation**: Custom MLP module with 4-bit quantized linear layers
- **Dequantization Testing**: Comprehensive testing framework for different dequantization methods
- **Memory Management**: Advanced CUDA memory allocation strategies
- **Debugging Tools**: Extensive logging for torch.compile graph analysis

## Installation

### Google Colab
```bash
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
!pip install --no-deps unsloth
```

### Local Environment
```bash
pip install unsloth
```

## Usage

### Basic Model Setup
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit",
    device_map="auto",
    quantization_config=bnb_config,
)
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### Training with SFTTrainer
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=10,
        output_dir="outputs",
        fp16=True,
    ),
)
trainer.train()
```

## Experimental Components

### Custom Dequantization Testing
The notebook includes a comprehensive testing framework for comparing different dequantization methods:
- Unsloth's fast dequantization
- PEFT's dequantization utilities
- Custom Triton kernel implementation (template provided)

### PyTorch Compilation Optimization
Experimental setup for compiling MLP forward passes with advanced optimization flags:
```python
torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": True,
    "triton.cudagraphs": False,
}
```

### Memory-Efficient Linear Layer
Template for implementing custom memory-efficient autograd functions for large linear transformations.

## Environment Configuration

### CUDA Memory Management
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True," \
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
```

### Debug Configuration
Extensive logging setup for analyzing torch.compile behavior and graph breaks.

## Supported Models

- Llama 3.1 8B Instruct (4-bit quantized)
- Llama 3.2 1B Instruct (4-bit quantized)
- Other Unsloth-compatible models

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- Transformers
- BitsAndBytes
- PEFT
- TRL
- Unsloth
- Triton

## Hardware Requirements

- **GPU**: CUDA-capable GPU with at least 8GB VRAM recommended
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for model downloads and outputs

## Dataset

Uses the LAION OIG (Open Instruction Generalist) dataset:
- Source: `unified_chip2.jsonl`
- Sample size: 10% of the full dataset for experimentation

## Performance Notes

- Supports both FP16 and BF16 training
- Automatic mixed precision for memory efficiency
- Gradient checkpointing for reduced memory usage
- Multi-GPU support with device mapping

## Debugging and Monitoring

The notebook includes extensive debugging capabilities:
- Torch.compile graph break analysis
- Memory usage monitoring
- Performance benchmarking for dequantization methods
- Gradient flow verification

## Customization

### Adding Custom Kernels
Template provided for implementing custom Triton kernels for specialized operations.

### Memory Management
Utilities included for cleaning up patched modules and managing CUDA memory.

## Contributing

This is experimental code designed for research and learning. Feel free to:
- Implement the custom Triton kernel templates
- Experiment with different LoRA configurations
- Test on different model architectures
- Optimize memory usage patterns

## Troubleshooting

### Common Issues
- **CUDA OOM**: Reduce batch size or enable gradient checkpointing
- **Graph Breaks**: Check the extensive logging output for compilation issues
- **Quantization Errors**: Verify BitsAndBytesConfig parameters match model requirements

### Environment Reset
Use the provided module cleanup functions to reset patched libraries when needed.

## License

Please refer to the individual library licenses for Unsloth, Transformers, PEFT, and other dependencies.
