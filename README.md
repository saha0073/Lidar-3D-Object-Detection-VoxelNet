# Lidar-3D-Object-Detection-Voxelnet (tensorflow 2.3.1)
![Image of Voxelnet Architecture](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/images/pre.png)

Implementation of [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396) in tensorflow 2.3.1. <br>
This project is inspired by the article of [Gopalakrishna Adusumilli](https://towardsdatascience.com/lidar-point-cloud-based-3d-object-detection-implementation-with-colab-part-1-of-2-e3999ea8fdd4) and the work of [David Stephane](https://github.com/steph1793).

# Skills Employed
* Modeling Techniques: VoxelNet, Lidar 3D Object Detection, PointNet, Convolutional autoencoder-decoder.
* Image Processing Techniques: Lidar data, Point cloud.
* Tech Stack: Python (3.7) 
* Libraries:  Tensorflow (2.3.1), opencv, numba.

# Installation
1. Clone this repository
2. Compile the Cython module
```bash
$ python3 setup build_ext --inplace
```
# Data preparation (Please refer to [Notebook](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/voxelnet_data_prep.ipynb))
Here we used the Kitti Vision Dataset. 
![Image of Kitti Dataset](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/images/kitti.PNG)

1. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Data to download include:
    * Velodyne point clouds (29 GB): input data to VoxelNet
    * Training labels of object data set (5 MB): input label to VoxelNet
    * Camera calibration matrices of object data set (16 MB): for visualization of predictions
    * Left color images of object data set (12 GB): for visualization of predictions

2. In this project, we use the cropped point cloud data for training and validation. Point clouds outside the image coordinates are removed. Update the directories in `data/crop.py` and run `data/crop.py` to generate cropped data. Note that cropped point cloud data will overwrite raw point cloud data.

2. Split the training set into training and validation set according to the protocol [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz). And rearrange the folders to have the following structure:
```plain
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
       └── validation  <--- evaluation data
       |   ├── image_2
       |   ├── label_2
       |   └── velodyne
```

# Train (Please refer to [Notebook](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/voxelnet_training.ipynb)) 

Run  `train.py`. You can find the meaning of each hyperparameter in the script file.
```
$ !python train.py \
--strategy="all" \
--n_epochs=160 \
--batch_size=2 \
--learning_rate=0.001 \
--small_addon_for_BCE=1e-6 \
--max_gradient_norm=5 \
--alpha_bce=1.5 \
--beta_bce=1 \
--huber_delta=3 \
--dump_vis="no" \
--data_root_dir="../DATA_DIR/T_DATA" \
--model_dir="model" \
--model_name="model6" \
--dump_test_interval=40 \
--summary_interval=10 \
--summary_val_interval=10 \
--summary_flush_interval=20 \
--ckpt_max_keep=10 \
```

# Evaluate
1. Run `predict.py`.

```
!python predict.py \
--strategy="all" \
--batch_size=2 \
--dump_vis="yes" \
--data_root_dir="../DATA_DIR/T_DATA/" \
--dataset_to_test="validation" \
--model_dir="model" \
--model_name="model6" \
--ckpt_name="" \
```

2. Then, run the kitty_eval project to compute the performances of the model.
```
./kitti_eval/evaluate_object_3d_offline [DATA_DIR]/validation/label_2 ./predictions [output file]
```

# Performances

In this case of 3D Object segmentation both Classification loss & Regression loss were used as the metric of performance. Here is the snapshot of the log dir visualization with Tensorboard. 
![voxel_training](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/images/voxel_train.PNG)


The predicted bounding boxes are decent. The model was trained only with 100 images for 16 epoch, the prediction quality will improve a lot when trained with more number of images.

![perf1](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/images/000003_front.jpg)
![perf2](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/images/000013_front.jpg)
![perf3](https://github.com/saha0073/Lidar-3D-Object-Detection-VoxelNet/blob/main/images/000100_front.jpg)


Happy Learning!

