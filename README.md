# Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play


![Overview](self-play-taste.png)


**Paper Link:** [https://arxiv.org/abs/2509.25541](https://arxiv.org/abs/2509.25541)


## Abstract

Although reinforcement learning (RL) can effectively enhance the reasoning capabilities of vision–language models (VLMs), current methods remain heavily dependent on labor-intensive datasets that require extensive manual construction and verification, leading to extremely high training costs and consequently constraining the practical deployment of VLMs. 
To address this challenge, we propose **Vision-Zero**, *a domain-agnostic framework enabling VLM self-improvement through competitive visual games generated from arbitrary image pairs.*
Specifically, Vision-Zero encompasses three main attributes:
(1) **Strategic Self-Play Framework:** 
Vision-Zero trains VLMs in "Who Is the Spy"-style games, where the models engage in strategic reasoning and actions across multiple roles. Through interactive gameplay, models autonomously generate their training data without human annotation.
(2) **Gameplay from Arbitrary Images:** Unlike existing gamified frameworks, Vision-Zero can generate games from arbitrary images, thereby enhancing the model’s reasoning ability across diverse domains and showing strong generalization to different tasks.
We demonstrate this versatility using three distinct types of image datasets: CLEVR-based synthetic scenes, charts, and real-world images.
(3) **Sustainable Performance Gain:** We introduce Iterative Self-Play Policy Optimization (Iterative-SPO), a novel training algorithm that alternates between Self-Play and reinforcement learning with verifiable rewards (RLVR), mitigating the performance plateau often seen in self-play-only training and achieving sustained long-term improvements.
Despite using label-free data, Vision-Zero achieves state-of-the-art performance on reasoning, chart question answering, and vision-centric understanding tasks, surpassing other annotation-based methods.


**The current release version includes:**

✅  **Models Release:** Models training on **Qwen2.5-VL-7B, InternVL3-8B, InternVL3-14B** using Vision-Zero. 

✅  **Dataset Release:** **Clevr-based** dataset used in Vision-Zero. 

✅  **Training:** The complete training code is open source.

**To Do List:**
- [ ] Publish **Chart-based** and **RealWorld-based** dataset.



## Vision-Zero Model and Dataset
| Model       |  Link|
| ----------- | --------- |
| Vision-Zero-Qwen-2.5-VL-7B-Clevr  | [Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-Clevr](https://huggingface.co/Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-Clevr)|
| Vision-Zero-Qwen-2.5-VL-7B-Chart  | [Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-Chart](https://huggingface.co/Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-Chart)|
| Vision-Zero-Qwen-2.5-VL-7B-RealWorld     | [Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-RealWorld](https://huggingface.co/Qinsi1/Vision-Zero-Qwen-2.5-VL-7B-RealWorld)|
| Vision-Zero-InternVL3-8B-Clevr  | [Qinsi1/Vision-Zero-InternVL3-8B-Clevr](https://huggingface.co/Qinsi1/Vision-Zero-InternVL3-8B-Clevr)|
| Vision-Zero-InternVL3-14B-Clevr   | [Qinsi1/Vision-Zero-InternVL3-14B-Clevr](https://huggingface.co/Qinsi1/Vision-Zero-InternVL3-14B-Clevr) |
| Vision-Zero-clevr-dataset  | [Qinsi1/Vision-Zero-clevr-dataset](https://huggingface.co/datasets/Qinsi1/Vision-Zero-clevr-dataset) |



## 🛠️ Setup
📢  This repo is based on [`vlm-r1`](https://github.com/om-ai-lab/VLM-R1), thanks for their contribution. 
```bash
conda create -n vision-zero python=3.10
conda activate vision-zero
bash setup.sh
```

## 💪🏻 Training

### Step1: Prepare Dataset and Model
You can download Clevr-based, Chart-based or Real-World dataset available above, or prepare your own dataset. And then update the model, dataset and output address in `run_scripts/run_grpo_vision_zero.sh`
```python
IMAGES_DIR = $IMAGES_DIR
SCENES_DIR = $SCENES_DIR
MODEL = $MODEL
OUTPUT_BASE_DIR = $OUTPUT_DIR
RUN_NAME= [run_name]
```

### Step2: Training Model
Use the following command to train the model. You can modify all hyperparameters in `run_scripts/run_grpo_vision_zero.sh`.
```bash
bash run_scripts/run_grpo_vision_zero.sh
```
### Step3: Evaluation
After training and saving the checkpoint, we use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to directly test the model's performance on out-of-distribution (OOD) tasks.



## Paper and Citation

More technical details can be found in our paper. If you find Vision-Zero useful or relevant to your project and research, please kindly cite our paper:

```
@misc{wang2025visionzeroscalablevlmselfimprovement,
      title={Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play}, 
      author={Qinsi Wang and Bo Liu and Tianyi Zhou and Jing Shi and Yueqian Lin and Yiran Chen and Hai Helen Li and Kun Wan and Wentian Zhao},
      year={2025},
      eprint={2509.25541},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.25541}, 
}
```


