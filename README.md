# Concept Bottleneck Models with Double Deferral 
## EECS 592 Final Project

This code is based on "Probabilistic Concept Bottleneck Models."<br>
[ArXiv](https://arxiv.org/abs/2306.01574) | [OpenReview](https://openreview.net/forum?id=yOxy3T0d6e)
Code is publicly available at https://github.com/ejkim47/prob-cbm.

The CUB dataset and processing originates from "Concept Bottleneck Models."<br>
[Paper](https://proceedings.mlr.press/v119/koh20a.html)
Code is publicly available at https://github.com/yewsiang/ConceptBottleneck.

Part of code is borrowed from [Evaluating Weakly Supervised Object Localization Methods Right](https://github.com/clovaai/wsolevaluation), [Probabilistic Cross-Modal Embedding](https://github.com/naver-ai/pcme), and [Polysemous Visual-Semantic Embedding (PVSE)](https://github.com/yalesong/pvse).


## Abstract
This study aims to explore the following research question: How can two-stage deferral processes in Concept Bottleneck Models improve performance in human-AI teams? To investigate the proposed question, our proposed work involves extending CBMs to include two stages of deferral: at the concept level and at the class level. Furthermore, we will use standard datasets to evaluate the accuracy of the proposed framework. We hypothesise that the addition of a second-stage deferral will improve the accuracy of decisions made in human-AI teams.


## Usage

### Step 1. Prepare Dataset

For the dataset (CUB 200 2011), please download from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).
Then, from the CUB folder do the following run `data_processing.py` to obtain train/ val/ test splits as well as to extract all relevant task and concept metadata into pickle files. 
Please change the 'data_root' and 'metadataroot' in a config file.

### Step 2. Train a model

Regular CBM with 2 stage deferral:
```bash
python3 ./main_regcbm.py --config ./configs/config_exp.yaml --prob 0.7 --alpha 0.4
```

Regular CBM with no deferral:
```bash
python3 ./main_regcbm.py --config ./configs/config_exp.yaml --prob 0.7 --alpha 0.4 --nodeferral
```

ProbCBM with 2 stage deferral:
```bash
python3 ./main_probcbm.py --config ./configs/config_exp.yaml --prob 0.7 --alpha 0.4
```

ProbCBM with no deferral:
```bash
python3 ./main_probcbm.py --config ./configs/config_exp.yaml --prob 0.7 --alpha 0.4 --nodeferral
```

After training, final results with be available in the 'jsons' folder. 
Note: The full suite of experiments can be run using the run_experiments.sh script. 
