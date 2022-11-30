# Deep Feature Correspondence (DFC)
Project: Unsupervised Anomaly Detection and Segmentation via Deep Feature Correspondence

Souce code for the paper published in PR Journal "Learning Deep Feature Correspondence for Unsupervised Anomaly Detection and Segmentation".
Download from [here](https://www.sciencedirect.com/science/article/abs/pii/S0031320322003557) or [researchgate](https://www.researchgate.net/publication/361590849_Learning_Deep_Feature_Correspondence_for_Unsupervised_Anomaly_Detection_and_Segmentation)

## Introduction
Developing machine learning models that can detect and localize the unexpected or anomalous structures within images is very important for numerous computer vision tasks, such as the defect inspection of manufactured products.
However, it is challenging especially when there are few or even no anomalous image samples available. 
In this project, we propose an unsupervised mechanism, i.e. deep feature correspondence (DFC), which can be effectively leveraged to detect and segment out the anomalies in images solely with the prior knowledge from anomaly-free samples. 
We develop our DFC in an asymmetric dual network framework that consists of a generic feature extraction network and an elaborated feature estimation network, and detect the possible anomalies within images by modeling and evaluating the associated deep feature correspondence between the two dual network branches.
Furthermore, to improve the robustness of the DFC and further boost the detection performance, we specifically propose a self-feature enhancement (SFE) strategy and a multi-context residual learning (MCRL) network module.
Extensive experiments have been carried out to validate the effectiveness of our DFC and the proposed SFE and MCRL. Our approach is very effective for detecting and segmenting the anomalies that appeared in confined local regions of images, especially the industrial anomalies. It advances the state-of-the-art performances on the benchmark dataset -- MVTec AD. Besides, when applied to a real industrial inspection scene, it outperforms the comparatives by a large margin.

# Qualitatve Resutls
## On MVTec AD dataset
![image](https://github.com/YoungGod/DFC/tree/master/figs/visualization_comparision_objects_lr.jpg)
![image](https://github.com/YoungGod/DFC/tree/master/figs/visualization_comparision_objects.jpg)
![image](https://github.com/YoungGod/DFC/tree/master/figs/visualization_comparision_textures.jpg)

# Dataset Download
BottleCap dataset can be download from [dropbox](https://www.dropbox.com/s/t3wlmw3j5x9lpyh/wine.zip?dl=0) or
[baidu](https://pan.baidu.com/s/1QAxKmFXy45GQx9fIuwg2GA) with pass code: yjyj

# Citation
If you find something useful, wellcome to cite our paper:
```
@article{YANG2022108874,
title = {Learning Deep Feature Correspondence for Unsupervised Anomaly Detection and Segmentation},
journal = {Pattern Recognition},
pages = {108874},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108874},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003557},
author = {Jie Yang and Yong Shi and Zhiquan Qi},
}
```

```
@article{DFR2020,
    title = "Unsupervised anomaly segmentation via deep feature reconstruction",
    journal = "Neurocomputing",
    year = "2020",
    issn = "0925-2312",
    doi = "https://doi.org/10.1016/j.neucom.2020.11.018",
    url = "http://www.sciencedirect.com/science/article/pii/S0925231220317951",
    author = "Yong Shi and Jie Yang and Zhiquan Qi",
}
```
