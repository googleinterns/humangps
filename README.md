# HumanGPS (CVPR2021)
### [Project Page](https://feitongt.github.io/HumanGPS/) | [Video](https://feitongt.github.io/HumanGPS/#Vis_feature) | [Paper](https://arxiv.org/pdf/2103.15573.pdf) | [Data]()
Tensorflow implementation of paper HumanGPS: Geodesic PreServing Feature for Dense Human Correspondences.

## Setup
* python 3.6
* TensorFlow 2.0
* Tensorflow-Addon
* gin-config
* scikit-learn

```
pip install -r requirements.txt  --user
```

## Running code
Here we show how to run our code on SMPL intra and inter testing data. You can download the rest of the synthetic SMPL testing data used in the paper [here]().
### Download pretrained model.
```
bash download_model.sh
```
### Evaluate on intra testing data.
Run
```
bash download_intra_data.sh
```
to get our smpl intra test dataset.

To evaluate average epe on intra test dataset.

1. set JOB_NAME="eval_optical_flow_intra" in ./script/inference_local.sh

2. Run 
```
bash ./script/inference_local.sh
```

### Evaluate on inter testing data.
Run
```
bash download_inter_data.sh
```
to get our smpl inter test dataset.

To evaluate average epe on inter test dataset.

1. set JOB_NAME="eval_optical_flow_inter" in ./script/inference_local.sh

2. Run 
```
bash ./script/inference_local.sh
```

### Inference on toy example for visualization.
Check out inference_demo.ipynb




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