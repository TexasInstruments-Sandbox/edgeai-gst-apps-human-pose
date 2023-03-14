# Edge AI GStreamer Apps for Human Pose Estimation

> Repository to host GStreamer based Edge AI applications for TI devices

This repo adds support for human pose estimation on top of edgeai-gst-apps
## Table of content
- [Supported Devices](#supported-devices)
- [Steps to run](#steps-to-run)
- [Result](#result)
- [About Human Pose Estimation](#about-human-pose-estimation)
- [Model Directory](#model-directory)
- [How to add your own custom post-processing?](#how-to-add-your-own-custom-post-processing)
## Supported Devices

| **DEVICE**              | **Supported**      |
| :---:                   | :---:              |
| AM62A                   | :heavy_check_mark: |
| AM68A                   | :heavy_check_mark: |
| SK-TDA4VM               | :heavy_check_mark: |
| AM69A                   | :heavy_check_mark: |

## Steps to run:

1. Clone this repo in your target under /opt

    ```console
    root@tda4vm-sk:/opt# git clone https://github.com/TexasInstruments/edgeai-gst-apps-human-pose.git
    root@tda4vm-sk:/opt# cd edgeai-gst-apps-human-pose
    ```

2. Download model for human pose estimation

    ```console
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose# ./download_models.sh -d human_pose_estimation
    ```

3. Download sample input video

    ```console
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose# wget --proxy off http://software-dl.ti.com/jacinto7/esd/edgeai-test-data/demo_videos/human_pose_estimation_sample.h264 -O /opt/edgeai-test-data/videos/human_pose_estimation_sample.h264
    ```

4. Run the python app

    ```console
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose# cd apps_python
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose/apps_python# ./app_edgeai.py ../configs/human_pose_estimation.yaml
    ```

5. Compile cpp apps

    ```console
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose# ./scripts/compile_cpp_apps.sh
    ```

5. Run CPP app

    ```console
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose# cd apps_cpp
    root@tda4vm-sk:/opt/edgeai-gst-apps-human-pose/apps_cpp# ./bin/Release/app_edgeai ../configs/human_pose_estimation.yaml
    ```
## Result
<br/>
<p align="center">
<img src="docs_data/human_pose.gif" width="640" title="Resultant Output">
</p> 

## About Human Pose Estimation
Multi person 2D human pose estimation is the task of understanding humans in an image. Given an input image, target is to detect each person and localize their body joints. Multi person 2D pose estimation is the task of understanding humans in an image. Given an input image, target is to detect each person and localize their body joints. 

## YOLO-Pose Based Multi-Person Pose Estimation Models
* YOLO-pose is a heatmap-free approach for joint detection, and 2D multi-person pose  estimation in an image based on the popular YOLO object detection framework. This approach doesn’t require the postprocessing of bottom-up approaches to group detected keypoints into a skeleton as each bounding box has an associated pose, resulting in an inherent grouping of the keypoints. For further details refer to this [paper](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Maji_YOLO-Pose_Enhancing_YOLO_for_Multi_Person_Pose_Estimation_Using_Object_CVPRW_2022_paper.pdf).

* YOLO-Pose based models are supported as part of TI Deep Learning Library(TIDL) with full hardware acceleration. These models can be trained and exported following the instruction in this [repository](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose). 

* The exported models can be further compiled in edgeai-benchmark [repository](https://github.com/TexasInstruments/edgeai-benchmark) with the corresponding [configs](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/human_pose_estimation.py)

## Model Directory

The default model downloaded from the model-zoo will be present under /opt/model_zoo in the target. It is a directory containing the model, artifacts and other necessary information including dataset.yaml file which contains the dataset information of the model and params.yaml file which contains information like the task-type of the model, preprocess information, postprocess information etc.

```
/opt/model_zoo/ONR-KD-7060-human-pose-yolox-s-640x640
└───params.yaml
└───artifacts
    └───allowedNode.txt
    └───detections_tidl_io_1.bin
    └───detections_tidl_net.bin
└───model
    └───yolox_s_pose_ti_lite_49p5_78p0.onnx
```
## How to add your own custom post-processing?

The parent repo of this fork i.e. edgeai-gst-apps supports post-processing for image classification, object detection and semantic segmentation. Since we are adding a new type of task, we need to write out own post-processing logic for it. The application has both python and C++ variants so the same post-processing logic needs to be added to both. It is recommended to start with python and then eventually move to C++. OpenCV, a popular computer vision library is used to draw appropriate detections on the frames.

Post-processing can be simple(ex: image classification) but in some cases the output from the model cannot be directly translated to a visual format. Some complex processing might be needed to convert the output from the deep learing network to a format that can be visualized. A detailed explanation about the post processing code can be found below.

The code changes done to add post-processing logic for human-pose-estimation can be found in this [commit](https://github.com/TexasInstruments/edgeai-gst-apps/commit/ec9774743efafb84905021bd21c94427e18ab251).

### <ins>Basic summary of the code changes</ins>
* **apps_python**: Adding new post process class for human pose estimation in post_process.py
* **apps_cpp**:    Make a new post process class for human pose estimation and modify post_process_image.cpp to call the newly created class appropriately
* **configs**:     Create a new config file with the downloaded/custom model

### <ins>Detailed explanation</ins>