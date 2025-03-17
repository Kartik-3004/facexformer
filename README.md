<div align="center">

# _FaceXFormer_ : A Unified Transformer <br> for Facial Analysis

[Kartik Narayan*](https://kartik-3004.github.io/portfolio/) &emsp; [Vibashan VS*](https://vibashan.github.io) &emsp; [Rama Chellappa](https://engineering.jhu.edu/faculty/rama-chellappa/) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)  

Johns Hopkins University

<a href='https://kartik-3004.github.io/facexformer/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2403.12960v3'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/kartiknarayan/facexformer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>
<a href='https://colab.research.google.com/drive/1wtUl9THqwkHOK6Zu8myJTjZWatLnuqeh?usp=sharing'><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

</div>

Official implementation of **[_FaceXFormer_ : A Unified Transformer for Facial Analysis](https://kartik-3004.github.io/facexformer/)**.
<hr />

## Highlights

**_FaceXFormer_**, is the first unified transformer for facial analysis:

1️⃣ that is capable of handling a comprehensive range of facial analysis tasks such as face parsing, landmark detection, head pose estimation, attributes recognition, age/gender/race estimation, facial expression recognition, and face visibility prediction<br>
2️⃣ that leverages a transformer-based encoder-decoder architecture where each task is treated as a learnable token, enabling the integration of multiple tasks within a single framework<br>
3️⃣ that effectively handles images "in-the-wild," demonstrating its robustness and generalizability across nine heterogenous tasks, all while maintaining the real-time performance of 33.21 FPS<br>

<p align="center" width="100%">
  <img src='docs/static/images/intro.png'>
</p>

> **<p align="justify"> Abstract:** *In this work, we introduce <i>FaceXFormer</i>, an end-to-end unified 
transformer model capable of performing nine facial analysis tasks including face parsing, landmark detection, 
head pose estimation, attribute prediction, and estimation of age, gender, race, expression, and face visibility 
within a single framework. Conventional methods in face analysis have often relied on task-specific designs 
and pre-processing techniques, which limit their scalability and integration into a unified architecture. 
Unlike these conventional methods, <i>FaceXFormer</i> leverages a transformer-based encoder-decoder architecture 
where each task is treated as a learnable token, enabling the seamless integration and simultaneous processing of 
multiple tasks within a single framework. Moreover, we propose a novel parameter-efficient decoder, FaceX, which 
jointly processes face and task tokens, thereby learning generalized and robust face representations across 
different tasks. We jointly trained <i>FaceXFormer</i> on nine face perception datasets and conducted experiments 
against specialized and multi-task models in both intra-dataset and cross-dataset evaluations across multiple benchmarks, showcasing state-of-the-art or competitive performance. Further, we performed a comprehensive analysis of different 
backbones for unified face task processing and evaluated our model in-the-wild, demonstrating its robustness and generalizability. To the best of our knowledge, this is the first work to propose a single model capable of 
handling nine facial analysis tasks while maintaining real-time performance at 33.21 FPS.* </p>

# :rocket: News
- [03/19/2024] 🔥 We release <i>FaceXFormer</i>.

## Installation
```bash
conda env create --file environment_facex.yml
conda activate facexformer

# Install requirements
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
## Download Models
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
## Usage

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

<p align="center" width="100%">
  <img src='docs/static/images/qualitative.png'>
</p>

## TODOs
- Release dataloaders for the datasets used.
- Release training script.

## Citation
If you find _FaceXFormer_ useful for your research, please consider citing us:

```bibtex
@article{narayan2024facexformer,
  title={FaceXFormer: A Unified Transformer for Facial Analysis},
  author={Narayan, Kartik and VS, Vibashan and Chellappa, Rama and Patel, Vishal M},
  journal={arXiv preprint arXiv:2403.12960},
  year={2024}
}
```
## Contact
If you have any questions, please create an issue on this repository or contact at knaraya4@jhu.edu
