# HumanGPS: Geodesic PreServing Feature for Dense Human Correspondences
Tensorflow implementation of paper "HumanGPS: Geodesic PreServing Feature for Dense Human Correspondences", CVPR 2021

### [Project Page](https://feitongt.github.io/HumanGPS/) | [Video](https://feitongt.github.io/HumanGPS/#Vis_feature) | [Paper](https://arxiv.org/pdf/2103.15573.pdf) | [Data]()

## Setup
* python 3.6
* TensorFlow 2.0
* Tensorflow-Addon
* gin-config
* scikit-learn

```
pip install -r requirements.txt  --user
pip install gdown
```

## Running code
Here we show how to run our code on SMPL intra and inter testing data. You can download the rest of the synthetic SMPL testing data used in the paper [here]().
### 1. Download pretrained model.
```
bash download_model.sh
```
### 2. Evaluate on intra testing data. (Testing data will be uploaded soon)
(a) Run
```
bash download_intra_data.sh
```
to get our smpl intra test dataset.

To evaluate average epe on intra test dataset.

(b) set JOB_NAME="eval_optical_flow_intra" in ./script/inference_local.sh

(c) Run 
```
bash ./script/inference_local.sh
```

### 3. Evaluate on inter testing data. (Testing data will be uploaded soon)
(a) Run
```
bash download_inter_data.sh
```
to get our smpl inter test dataset.

To evaluate average epe on inter test dataset.

(b) set JOB_NAME="eval_optical_flow_inter" in ./script/inference_local.sh

(c) Run 
```
bash ./script/inference_local.sh
```

### 4. Inference on toy examples for visualization.
Check out inference_demo.ipynb for toy examples.




# Citation
If you find this code useful in your research, please cite:

```
@article{tan2021humangps,
  title={HumanGPS: Geodesic PreServing Feature for Dense Human Correspondences},
  author={Tan, Feitong and Tang, Danhang and Dou, Mingsong and Guo, Kaiwen and Pandey, Rohit and Keskin, Cem and Du, Ruofei and Sun, Deqing and Bouaziz, Sofien and Fanello, Sean and others},
  journal={arXiv preprint arXiv:2103.15573},
  year={2021}
}
```