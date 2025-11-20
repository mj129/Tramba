# <p align=center>`Salient Object Detection in Traffic Scene Through the TSOD10K Dataset`</p><!-- omit in toc -->

> **Authors:**
> Yu Qiu*,
> Yuhang Sun*,
> Jie Mei,
> Lin Xiao,
> Jing Xu

This repository contains the proposed TSOD10K dataset, official code, prediction results. The technical report could be found at [**[TIP]**](https://ieeexplore.ieee.org/document/11226901) & [**[arXiv]**](https://arxiv.org/abs/2503.16910).

## Introduction

Traffic Salient Object Detection (TSOD) aims to segment the objects critical to driving safety by combining semantic (e.g., collision risks) and visual saliency. Unlike SOD in natural scene images (NSI-SOD), which prioritizes visually distinctive regions, TSOD emphasizes the objects that demand immediate driver attention due to their semantic impact, even with low visual contrast. This dual criterion, i.e., bridging perception and contextual risk, re-defines saliency for autonomous and assisted driving systems. To address the lack of task-specific benchmarks, we collect the first large-scale TSOD dataset with pixel-wise saliency annotations, named TSOD10K. TSOD10K covers the diverse object categories in various real-world traffic scenes under various challenging weather/illumination variations (e.g., fog, snowstorms, low-contrast, and low-light). Methodologically, we propose a Mamba-based TSOD model, termed Tramba. Considering the challenge of distinguishing inconspicuous visual information from complex traffic backgrounds, Tramba introduces a novel Dual-Frequency Visual State Space module equipped with shifted window partitioning and dilated scanning to enhance the perception of fine details and global structure by hierarchically decomposing high/low-frequency components. To emphasize critical regions in traffic scenes, we propose a traffic-oriented Helix 2D-Selective-Scan (Helix-SS2D) mechanism that injects driving attention priors while effectively capturing global multi-direction spatial dependencies. We establish a comprehensive benchmark by evaluating Tramba and 25 existing NSI-SOD models on TSOD10K, demonstrating Tramba’s superiority. Our research establishes the first foundation for safety-aware saliency analysis in intelligent transportation systems.

## OverView

* **TSOD Dataset**
<p align="center">
  <img src="https://github.com/mj129/Tramba/blob/main/utils/figure/TSOD_examples.jpg" alt="architecture" width="90%">
</p>
<p align="center">
  <img src="https://github.com/mj129/Tramba/blob/main/utils/figure/TSOD_statistics.jpg" alt="architecture" width="80%">
</p>
<p align="center">
  <img src="https://github.com/mj129/Tramba/blob/main/utils/figure/TSOD_crisis_analysis.jpg" alt="architecture" width="80%">
</p>

* **Tramba**
<p align="center">
  <img src="https://github.com/mj129/Tramba/blob/main/utils/figure/MainFigure.jpg" alt="architecture" width="90%">
</p>

## Getting Started

**0. Install**

You could refer to [here](https://github.com/MzeroMiko/VMamba?tab=readme-ov-file#installation).

**1. Download Datasets and Pretrained Weight.**

* **Datasets:** 
You can obtain the TSOD10K dataset from [Google Drive](https://drive.google.com/file/d/1RsXOMO37PHLLTtmyOP3af7ODP1yv7qWw/view).

* **Pretrained Weight for VSSM/Swin/PVT/ResNet** 

[-V: VMamba-B](https://drive.google.com/file/d/1Aew9Arfv8OPCxTdaJcHTnpwSkYVkqe3Y/view) | [-Swin: Swin-B](https://drive.google.com/file/d/1AwMASp-YJDyoCijL25s0hYcHh8Q2cLxv/view) | [-Pvt: PvtV2](https://drive.google.com/file/d/1x7Ca6uZI0Q7nM76Yn5oIf1BEbXDnC6hq/view) | [-Res: ResNet50](https://drive.google.com/file/d/1SUI6kbyekiMCO0cIIJowPHpYh34t5Rmj/view)

**2. Train Tramba.**

```
python run.py --data_root [your_path] --evaluation_root [your_path] --img_size [default:384] --pretrained_model [your_path] --batch_size [default:4] --save_model [result_save_path] --tf_log_path [log_save_path] --pretrained_path [your_path] --resume [ckpt if not None] --see [default:40] --train_epochs [default:80] --decay_epochs [default:60] --decay_factors [default:0.2] --lr [default:1e-4] --method ['Tramba-V-TSOD' or 'Tramba-V-SOD'] --best_MAE [default:None]
```

**3. Test Tramba.**
```
python test_TSOD.py # For TSOD Test
```
```
python test_SOD.py # For SOD Test:
```
**4. Eval Tramba.**
```
python evaluate_TSOD.py # For TSOD Eval
```
```
python evaluate_SOD.py # For SOD Eval
```


## Citation
```
@ARTICLE{qiu2025tramba,
  title={Salient Object Detection in Traffic Scene Through the TSOD10K Dataset},
  author={Qiu, Yu and Sun, Yuhang and Mei, Jie and Xiao, Lin and Xu, Jing},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```
