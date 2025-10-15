<div align="center">

# 🎯 Vision-Zero
### Scalable VLM Self-Improvement via Strategic Gamified Self-Play

[![arXiv](https://img.shields.io/badge/arXiv-2509.25541-b31b1b.svg)](https://arxiv.org/abs/2509.25541)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/Qinsi1)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset)

![Overview](self-play-taste.png)

*A domain-agnostic framework enabling VLM self-improvement through competitive visual games*

</div>

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🤖 Models & Dataset](#-models--dataset)
- [🛠️ Setup](#️-setup)
- [💪 Training](#-training)
- [📊 Evaluation](#-evaluation)
- [📄 Citation](#-citation)

---

## 🎯 Overview

Although reinforcement learning (RL) can effectively enhance the reasoning capabilities of vision–language models (VLMs), current methods remain heavily dependent on labor-intensive datasets that require extensive manual construction and verification, leading to extremely high training costs and consequently constraining the practical deployment of VLMs. 

To address this challenge, we propose **Vision-Zero**, *a domain-agnostic framework enabling VLM self-improvement through competitive visual games generated from arbitrary image pairs.*

### ✨ Key Features

<details>
<summary><b>🎮 Strategic Self-Play Framework</b></summary>

Vision-Zero trains VLMs in "Who Is the Spy"-style games, where the models engage in strategic reasoning and actions across multiple roles. Through interactive gameplay, models autonomously generate their training data without human annotation.

</details>

<details>
<summary><b>🖼️ Gameplay from Arbitrary Images</b></summary>

Unlike existing gamified frameworks, Vision-Zero can generate games from arbitrary images, thereby enhancing the model's reasoning ability across diverse domains and showing strong generalization to different tasks. We demonstrate this versatility using three distinct types of image datasets: CLEVR-based synthetic scenes, charts, and real-world images.

</details>

<details>
<summary><b>📈 Sustainable Performance Gain</b></summary>

We introduce Iterative Self-Play Policy Optimization (Iterative-SPO), a novel training algorithm that alternates between Self-Play and reinforcement learning with verifiable rewards (RLVR), mitigating the performance plateau often seen in self-play-only training and achieving sustained long-term improvements.

</details>

> 🏆 **Achievement:** Despite using label-free data, Vision-Zero achieves state-of-the-art performance on reasoning, chart question answering, and vision-centric understanding tasks, surpassing other annotation-based methods.


### 🎉 Current Release Status

| Component | Status | Description |
|-----------|---------|-------------|
| 🤖 **Models** | ✅ Available | Pre-trained models on Qwen2.5-VL-7B, InternVL3-8B, InternVL3-14B |
| 📊 **CLEVR Dataset** | ✅ Available | Complete CLEVR-based training dataset |
| 🛠️ **Training Code** | ✅ Available | Full open-source training pipeline |
| 📈 **Chart Dataset** | 🚧 Coming Soon | Chart-based dataset for enhanced reasoning |
| 🌍 **Real-World Dataset** | 🚧 Coming Soon | Real-world image dataset for diverse scenarios |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/vision-zero.git
cd vision-zero

# 2. Set up environment
conda create -n vision-zero python=3.10
conda activate vision-zero
bash setup.sh

# 3. Download a pre-trained model
# Choose from available models in the table below

# 4. Start training or inference
bash run_scripts/run_grpo_vision_zero.sh
```



## 🤖 Models & Dataset

### 🔬 Pre-trained Models

<div align="center">

| Model Family | Size | Dataset | HuggingFace Link |
|--------------|------|---------|------------------|
| **Qwen2.5-VL** | 7B | CLEVR | [![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-Clevr) |
| **Qwen2.5-VL** | 7B | Chart | [![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-Chart) |
| **Qwen2.5-VL** | 7B | Real-World | [![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-RealWorld) |
| **InternVL3** | 8B | CLEVR | [![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Qinsi1/Vision-Zero-InternVL3-8B-Clevr) |
| **InternVL3** | 14B | CLEVR | [![Model](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/Qinsi1/Vision-Zero-InternVL3-14B-Clevr) |

</div>

### 📊 Datasets

| Dataset Type | Description | Link |
|--------------|-------------|------|
| **CLEVR-based** | Synthetic scenes for logical reasoning | [![Dataset](https://img.shields.io/badge/🤗-Dataset-green)](https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset) |



## 🛠️ Setup

> 📢 **Acknowledgment:** This repo is based on [`vlm-r1`](https://github.com/om-ai-lab/VLM-R1) - thanks for their contribution!

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Conda or similar environment manager

### Installation

```bash
# Create and activate environment
conda create -n vision-zero python=3.10
conda activate vision-zero

# Install dependencies
bash setup.sh
```

## 💪 Training

### 📋 Training Pipeline

#### Step 1: 📁 Prepare Dataset and Model

Download one of the available datasets or prepare your own:
- **CLEVR-based**: Available now ✅
- **Chart-based**: Coming soon 🚧  
- **Real-World**: Coming soon 🚧

Configure your training setup in `run_scripts/run_grpo_vision_zero.sh`:

```bash
# Configuration variables
IMAGES_DIR=$IMAGES_DIR          # Path to your images
SCENES_DIR=$SCENES_DIR          # Path to scene descriptions  
MODEL=$MODEL                    # Base model to fine-tune
OUTPUT_BASE_DIR=$OUTPUT_DIR     # Output directory for checkpoints
RUN_NAME="your_run_name"        # Experiment name
```

#### Step 2: 🚀 Start Training

Launch the training process with customizable hyperparameters:

```bash
bash run_scripts/run_grpo_vision_zero.sh
```

> 💡 **Tip:** All hyperparameters can be modified directly in the script file.

#### Step 3: 📊 Evaluation

Evaluate your trained model on out-of-distribution tasks using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit):

```bash
# After training completes and checkpoint is saved
# Use VLMEvalKit for comprehensive evaluation
```

---

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for comprehensive model evaluation on out-of-distribution tasks, ensuring robust performance assessment across various benchmarks.

---

## 📄 Citation

If you find Vision-Zero useful in your research, please consider citing our paper:

```bibtex
@misc{wang2025visionzeroscalablevlmselfimprovement,
    title={Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play}, 
    author={Qinsi Wang and Bo Liu and Tianyi Zhou and Jing Shi and Yueqian Lin and Yiran Chen and Hai Helen Li and Kun Wan and Wentian Zhao},
    year={2025},
    eprint={2509.25541},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2509.25541}
}
```

---

<div align="center">

**🌟 Star this repo if you find it helpful!**
*Made with ❤️ by the Vision-Zero team*

</div>
