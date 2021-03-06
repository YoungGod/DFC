# Deep Feature Correspondence (DFC)
Project: Unsupervised Anomaly Detection and Segmentation via Deep Feature Correspondence

Souce code for the paper "Learning Deep Feature Correspondence for Unsupervised Anomaly Detection and Segmentation".
`Coming soon...`

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

