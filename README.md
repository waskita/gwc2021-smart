# SMART Team Solution

## Installation

```
python==3.8
pytorch==1.7

# install mish-cuda, if you use different pytorch version, you could try https://github.com/thomasbrandon/mish-cuda
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# download pretrained yolov4-p7.pt from https://github.com/WongKinYiu/ScaledYOLOv4
# put yolov4-p7.pt under "./ScaledYOLOv4/weights" folder
```



## Training and Testing

Our solution includes five stages:

- #### Stage One

```
cd ./ScaledYOLOv4/

# Train Scaled-YOLOv4 baseline
sh 1_train_baseline.sh

# Test Scaled-YOLOv4 on test dataset
sh 1_detect.sh
```

You will get test results in "./ScaledYOLOv4/inference/output/baseline.csv"



- #### Stage Two

```
cd ../ScaledYOLOv4_linearhead/

# Train Scaled-YOLOv4-linearhead model
sh 2_train_linearhead.sh

# Finetune Scaled-YOLOv4-linearhead model
sh 2_finetune_linearhead.sh

# Test Scaled-YOLOv4-linearhead on test dataset
sh 2_detect.sh
```
You will get test results in "./ScaledYOLOv4_linearhead/inference/output/linearhead.csv"



- #### Stage Three

```
# Create pseudo label for test dataset using Stage One and Stage Two models
cd ../Postprocess/
python create_pseudo_label.py
```



- #### Stage Four

```
# Retrain Stage One model on pseudo test dataset
cd ../ScaledYOLOv4
sh 4_retrain_baseline.sh

# Test retrained Stage One models on test dataset
sh 4_detect30.sh
sh 4_detect50.sh
sh 4_detect100.sh

# Retrain Stage Two model on pseudo test dataset
cd ../ScaledYOLOv4_linearhead
sh 4_retrain_linearhead.sh

# Test retrained Stage Two models on test dataset
sh 4_detect30.sh
sh 4_detect50.sh
sh 4_detect100.sh
```

You will get test results as follows:

"./ScaledYOLOv4/inference/output/baseline_pseudo30.csv"

"./ScaledYOLOv4/inference/output/baseline_pseudo50.csv"

"./ScaledYOLOv4/inference/output/baseline_pseudo100.csv"

"./ScaledYOLOv4_linearhead/inference/output/linearhead_pseudo30.csv"

"./ScaledYOLOv4_linearhead/inference/output/linearhead_pseudo50.csv"

"./ScaledYOLOv4_linearhead/inference/output/linearhead_pseudo100.csv"



- #### Stage Five

```
cd ../Postprocess

# Ensemble baseline results
python ensemble_baseline.py

# Ensemble linearhead results
python ensemble_linearhead.py

# Ensemble final results
python ensemble_all.py

# Remove duplicate predictions
python rm_SmallinBig.py

```

You will get final result in "./Postprocess/submission.csv"



## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

</details>
