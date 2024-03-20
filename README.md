<div align="center">

# _FaceXFormer_ : A Unified Transformer <br> for Facial Analysis

[Kartik Narayan*](https://kartik-3004.github.io/portfolio/) &emsp; [Vibashan VS*](https://vibashan.github.io) &emsp; [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)  

Johns Hopkins University

<a href='https://kartik-3004.github.io/facexformer_web/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2403.12960'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/kartiknarayan/facexformer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

Official implementation of **[_FaceXFormer_ : A Unified Transformer for Facial Analysis](https://kartik-3004.github.io/facexformer_web/)**.
<hr />

## Highlights

**_FaceXFormer_**, is the first unified transformer for facial analysis:

1️⃣ that is capable of handling a comprehensive range of facial analysis tasks such as face parsing, landmark detection, head pose estimation, attributes recognition, age/gender/race estimation and landmarks visibility prediction<br>
2️⃣ that leverages a transformer-based encoder-decoder architecture where each task is treated as a learnable token, enabling the integration of multiple tasks within a single framework<br>
3️⃣ that effectively handles images "in-the-wild," demonstrating its robustness and generalizability across eight heterogenous tasks, all while maintaining the real-time performance of 37 FPS<br>

<img src='assets/intro_viz.png'>

> **<p align="justify"> Abstract:** *In this work, we introduce FaceXformer, an end-to-end unified transformer
> model for a comprehensive range of facial analysis tasks such as face parsing, landmark detection,
> head pose estimation, attributes recognition, and estimation of age, gender, race, and landmarks visibility.
> Conventional methods in face analysis have often relied on task-specific designs and preprocessing techniques,
> which limit their approach to a unified architecture. Unlike these conventional methods, our FaceXformer
> leverages a transformer-based encoder-decoder architecture where each task is treated as a learnable token,
> enabling the integration of multiple tasks within a single framework. Moreover, we propose a parameter-efficient
> decoder, FaceX, which jointly processes face and task tokens, thereby learning generalized and robust face
> representations across different tasks. To the best of our knowledge, this is the first work to propose a
> single model capable of handling all these facial analysis tasks using transformers.  We conducted a
> comprehensive analysis of effective backbones for unified face task processing and evaluated different task
> queries and the synergy between them. We conduct experiments against state-of-the-art specialized models and
> previous multi-task models in both intra-dataset and cross-dataset evaluations across multiple benchmarks.
> Additionally, our model effectively handles images "in-the-wild," demonstrating its robustness and generalizability
> across eight different tasks, all while maintaining the real-time performance of 37 FPS.* </p>

# :rocket: News
- [03/19/2024] 🔥 We release FaceXFormer.

# Installation
```bash
conda env create --file environment_facex.yml
conda activate facexformer

# Install requirements
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
# Download Models
The models can be downloaded manually from [HuggingFace](https://huggingface.co/kartiknarayan/facexformer) or using python:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
```
The directory structure should finally be:

```
  . ── facexformer ──┌── ckpts/model.pt
                     ├── network
                     └── inference.py                    
```
# Usage

Download trained model from [HuggingFace](https://huggingface.co/kartiknarayan/facexformer) and ensure the directory structure is correct.<br>
For demo purposes, we have released the code for inference on a single image.<br>
It supports a variety of tasks which can be prompted by changing the "task" argument. 

```python
python inference.py --model_path ckpts/model.pt \
                    --image_path image.png \
                    --results_path results \
                    --task parsing \
                    --gpu_num 0

-- task = [parsing, landmarks, headpose, attributes, age_gender_race, visibility]
```
The output is stored in the specified "results_path".

<img src='assets/viz_inthewild.png'>

## TODOs
- Release dataloaders for the datasets used.
- Release training script.

## Citation
If you find _FaceXFormer_ useful for your research, please consider citing us:

```bibtex
```
## Contact
If you have any questions, please create an issue on this repository or contact at knaraya4@jhu.edu
