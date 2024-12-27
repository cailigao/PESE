# Perturbation-based Expandable Subspace Ensemble for Class Incremental Face Forgery Detection

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/PESE-v1.0"></a>
  <a href=""><img src="https://img.shields.io/github/stars/cailigao/PESE?color=4fb5ee"></a>
  <a href=""><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2cailigao%2PESE&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false"></a>
</p>


The code repository for "erturbation-based Expandable Subspace Ensemble for Class Incremental Face Forgery Detection"  in PyTorch. 

The rapid development of face forgery generation technologies has made it easier to create increasingly realistic forged faces, posing significant threats to societal security. While existing detection methods perform well on known forgery techniques, they struggle to adapt to emerging forgeries. Efforts to address this challenge have focused on enhancing generalization or using adapter-based incremental learning. However, generalization methods fail to update model parameters with new forgeries, and adapter-based approaches face issues such as overfitting and biased distributions.
To address these limitations, we model face forgery detection as a class incremental learning problem and introduce a new approach, Perturbation-based Expandable Subspace Ensemble (PESE) for Class Incremental Face Forgery Detection. PESE integrates three key components: subspace expansion, parameter perturbation, and subspace ensemble. These components work together to balance model stability and plasticity by expanding subspaces to capture task-specific knowledge, applying parameter perturbation to avoid overfitting, and integrating subspaces to address scale differences and representation misalignments. Additionally, we introduce a new class-incremental face forgery detection dataset, CIL-forgery, to support the evaluation of incremental learning methods. Extensive experiments conducted on this dataset demonstrate the effectiveness of our PESE method in overcoming the challenges posed by emerging face forgery techniques, demonstrating its robustness in adapting to new forgeries while maintaining strong performance on previously learned tasks.

<img src='resources/pese.png' width='900'>

## 🎊 Results

We compared the performance of six advanced methods under three different settings to verify the competitive performance of PESE.

<img src='resources/results.png' width='1400'>

## Requirements
### 🗂️ Environment
1. [torch 2.4.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.19.0](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [easydict](https://github.com/makinacorpus/easydict)

### 🔎 Dataset




## 🔑 Running scripts

Please follow the settings in the `exps` folder to prepare json files, and then run:

```
python main.py --config ./exps/[filename].json
```

**Here is an example of how to run the code** 

if you want to run the CIL-Forgery dataset using Swin-B, you can follow the script: 
```
python main.py --config ./exps/pese.json
```

if you want to run the CIL-Forgery dataset using ViT-B/16-IN1K, you can follow the script: 
```
python main.py --config ./exps/pese_vit.json
```

if you want to run the CIL-Forgery dataset using ViT-B/16-IN21K, you can follow the script: 
```
python main.py --config ./exps/pese_vit_in21k.json
```

After running the code, you will get a log file in the `logs/pese/CIL_Forgery/` folder.

## 👨‍🏫 Acknowledgment

We would like to express our gratitude to the following repositories for offering valuable components and functions that contributed to our work.

- [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox](https://github.com/sun-hailong/LAMDA-PILOT)
