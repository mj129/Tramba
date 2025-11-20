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

* TSOD Dataset：
![Examples](https://github.com/mj129/Tramba/blob/main/utils/figure/TSOD_examples.jpg)
![Examples](https://github.com/mj129/Tramba/blob/main/utils/figure/TSOD_statistics.jpg)
![Examples](https://github.com/mj129/Tramba/blob/main/utils/figure/TSOD_crisis_analysis.jpg) 

* Tramba:
![framework](https://github.com/mj129/Tramba/blob/main/utils/figure/MainFigure.jpg) 

## Get Start

**0. Install**

You could refer to [here](https://github.com/mczhuge/ICON/tree/main/util).

**1. Download Datasets and Checkpoints.**

* **Datasets:** 

[Baidu | 提取码:ICON](https://pan.baidu.com/s/1zFXR-xIykUhoj86kiQ3GxA)  or [Goole Drive](https://drive.google.com/file/d/1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu/view?usp=sharing)

also you could quikcly download by running:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aHYvxXGMsAS0yN4zhKt8kKorVL--9bLu" -O datasets.zip && rm -rf /tmp/cookies.txt
```

* **Checkpoints:** 

[Baidu | 提取码:ICON](https://pan.baidu.com/s/1zFXR-xIykUhoj86kiQ3GxA)  or [Goole Drive](https://drive.google.com/file/d/1wcL8n3lSc1pswMfDYCOBQJJjwnuhWdwK/view)

also you could quikcly download by running:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wcL8n3lSc1pswMfDYCOBQJJjwnuhWdwK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wcL8n3lSc1pswMfDYCOBQJJjwnuhWdwK" -O checkpoint.zip && rm -rf /tmp/cookies.txt
```

**2. Train ICON.**
```
sh util/scripts/train_icon.sh
```

**3. Test ICON.**
```
sh util/scripts/test_icon.sh
```

**4. Eval ICON.**
```
sh util/scripts/run_sod_eval.sh
sh util/scripts/run_soc_eval.sh
```

# Tramba
You can obtain the TSOD10K dataset from [Google Drive](https://drive.google.com/file/d/1RsXOMO37PHLLTtmyOP3af7ODP1yv7qWw/view).

More details will be announced as soon as possible.
