# Code Generation AI Model - README

A fine-tuned language model specialized in generating Python code functions from natural language descriptions. This project demonstrates the process of training and using a transformer model for code generation tasks.

## 🚀 Overview

This project fine-tunes a causal language model to generate Python function definitions based on textual prompts. The model can create various Python functions including string manipulation, list operations, and algorithmic implementations.

## ✨ Features

- **Code Generation**: Generate Python function definitions from natural language descriptions
- **Fine-tuning**: Custom training on code generation tasks
- **Multiple Function Types**: Support for various Python programming concepts
- **Efficient Inference**: Optimized for fast code generation
- **Model Export**: Save and share fine-tuned models

## 🛠️ Technology Stack

### Core Technologies
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - Pre-trained models and training utilities
- **TRL (Transformer Reinforcement Learning)** - Training library
- **Unsloth** - Optimization for faster training

### Models
- **AutoModelForCausalLM** - Base language model architecture
- **AutoTokenizer** - Text tokenization and processing

## 📋 Prerequisites

```bash
# Required dependencies
pip install torch transformers trl unsloth accelerate
```

### Hardware Requirements
- GPU with CUDA support (recommended)
- Minimum 8GB RAM
- Sufficient storage for model weights

## 🏗️ Project Structure

```
code-generation-ai/
├── training/
│   ├── train_model.py
│   └── training_config.py
├── inference/
│   ├── generate_code.py
│   └── model_loader.py
├── models/
│   ├── fine-tuned_model/
│   └── ftuned-model/
├── examples/
│   ├── vowel_counter.py
│   ├── list_reverser.py
│   └── palindrome_checker.py
└── README.md
```

## 💻 Usage

### Code Generation Examples

The model can generate various Python functions including:

- **Vowel Counter**: Count vowels in a string
- **List Reverser**: Reverse lists efficiently
- **Palindrome Checker**: Check if strings are palindromes
- **And many more**: Mathematical functions, data structures, algorithms

## 🎯 Training Process

### Model Fine-tuning

The project uses SFTTrainer (Supervised Fine-Tuning) with optimized training parameters:

- **Batch Size**: 2 (per device) with gradient accumulation
- **Learning Rate**: 2e-4 with linear decay
- **Training Steps**: 100 steps with 5 warmup steps
- **Mixed Precision**: FP16/BF16 based on hardware support
- **Optimizer**: AdamW 8-bit for memory efficiency

### Training Configuration
- Maximum sequence length optimized for code generation
- Dataset processing with 2 parallel workers
- Specialized packing for faster training of short sequences

## 🔧 Implementation Details

### Prompt Format
The model uses a structured prompt format to ensure clean code generation, specifically designed to return only function definitions without additional text.

### Model Configuration
- Faster inference enabled through optimization techniques
- Controlled generation with 64 max new tokens
- Cache utilization for improved performance

## 💾 Model Management

### Saving Models
- Local saving of fine-tuned models and tokenizers
- Merged 16-bit model saving for efficient deployment
- Hugging Face Hub integration for model sharing

### Export Options
- Full model checkpoint saving
- Optimized merged model versions
- Hub deployment ready configurations

## 📊 Performance

### Training Efficiency
- Optimized training loop with gradient accumulation
- Memory-efficient training with 8-bit optimizers
- Fast convergence with appropriate learning rate scheduling

### Inference Speed
- 2x faster inference through native optimizations
- Efficient token generation with controlled output length
- GPU-accelerated performance

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd code-generation-ai
pip install -r requirements.txt
```

### Basic Usage
Load the pre-trained model and generate Python functions from natural language descriptions. The model specializes in creating clean, functional Python code based on your requirements.

## 🎨 Capabilities

The model can generate code for various programming domains:

- **Data Structures**: Stacks, queues, linked lists
- **Algorithms**: Searching, sorting, recursion
- **String Operations**: Manipulation, validation, parsing
- **Mathematical Functions**: Calculations, conversions
- **File Operations**: Reading, writing, processing

## 🔍 Quality Assurance

### Generated Code Features
- **Syntactic Correctness**: Valid Python syntax
- **Functional Implementation**: Implements described behavior
- **Clean Formatting**: Proper indentation and structure
- **Focused Output**: Returns only function definitions as requested

## 🌟 Applications

- **Educational Tool**: Learning Python programming
- **Developer Productivity**: Rapid function prototyping
- **Code Assistance**: Generating boilerplate code
- **Learning Resource**: Understanding code implementation patterns


---

**Note**: This project demonstrates the capabilities of fine-tuned language models for specialized code generation tasks and serves as a foundation for building more advanced AI programming assistants.
