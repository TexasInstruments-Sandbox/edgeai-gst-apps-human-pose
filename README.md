# Edge AI GStreamer Apps

> Repository to host GStreamer based Edge AI applications for TI devices

## Human Pose Estimation
Multi person 2D human pose estimation is the task of understanding humans in an image. Given an input image, target is to detect each person and localize their body joints. Multi person 2D pose estimation is the task of understanding humans in an image. Given an input image, target is to detect each person and localize their body joints. 

## YOLO-Pose Based Multi-Person Pose Estimation Models
* YOLO-pose is a heatmap-free approach for joint detection, and 2D multi-person pose  estimation in an image based on the popular YOLO object detection framework. This approach doesn’t require the postprocessing of bottom-up approaches to group detected keypoints into a skeleton as each bounding box has an associated pose, resulting in an inherent grouping of the keypoints. For further details refer to this [paper](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Maji_YOLO-Pose_Enhancing_YOLO_for_Multi_Person_Pose_Estimation_Using_Object_CVPRW_2022_paper.pdf).

* YOLO-Pose based models are supported as part of TI Deep Learning Library(TIDL) with full hardware acceleration. These models can be trained and exported following the instruction in this [repository](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose). 

* The exported models can be further compiled in edgeai-benchmark [repository](https://github.com/TexasInstruments/edgeai-benchmark) with the corresponding [configs](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/human_pose_estimation.py)
