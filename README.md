# LITA: LMM-guided Image-Text Alignment for Art Assessment
The official PyTorch implementation for "LITA: LMM-guided Image-Text Alignment for Art Assessment (Multimedia Modeling 2025)"
## Setup
### 1. Install 
```
docker build -t lita dockerfile/
bash exec.sh
conda create -n lita python=3.10 -y
conda activate lita
pip install -r requirements.txt
```
### 2. Download BAID dataset
- Clone the repository of BAID dataset
```
git clone https://github.com/Dreemurr-T/BAID.git
```
- Download the dataset
```
python BAID/downloading_script/download.py
```
The images will be saved to ```images/``` folder.
### 3. Generate comments with LLaVA
- Clone the repository of LLaVA
```
git clone https://github.com/haotian-liu/LLaVA.git
```
- Generate the comments from LLaVA
```
CUDA_VISIBLE_DEVICES=0 python generate_comments.py
```
Artistic aesthetics and style comments are saved to ```aesthetics_comment.csv``` and ```style_comment.csv```, respectively.
## Train & Test
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Acknowledgements
We borrow some code from [BAID](https://github.com/Dreemurr-T/BAID) and [LLaVA](https://github.com/haotian-liu/LLaVA)
## Citation
```
@InProceedings{10.1007/978-981-96-2061-6_20,
author={Sunada, Tatsumi and Shiohara, Kaede and Xiao, Ling and Yamasaki, Toshihiko}
title="LITA: LMM-Guided Image-Text Alignment for Art Assessment",
booktitle="MultiMedia Modeling",
year="2025",
pages="268--281",
}


```