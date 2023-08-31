# ReRF: Neural Residual Radiance Fields for Streamably Free-Viewpoint Videos
This repository contains an extended PyTorch implementation of the paper "Neural Residual Radiance Fields for Streamably Free-Viewpoint Videos", presented at CVPR 2023. My extension focuses on enabling the ReRF, originally implemented for object data, to reconstruct real scene llff data.

<br>

## Original Work
PyTorch implementation of paper "Neural Residual Radiance Fields for Streamably Free-Viewpoint Videos", CVPR 2023.

> Neural Residual Radiance Fields for Streamably Free-Viewpoint Videos   
> [Liao Wang](https://aoliao12138.github.io/), [Qiang Hu](https://qianghu-huber.github.io/qianghuhomepage/), 
>[Qihan He](https://www.linkedin.com/in/qihan-he-a378a61b7/), Ziyu Wang, [Jingyi Yu](http://www.yu-jingyi.com/),
>[Tinne Tuytelaars](https://homes.esat.kuleuven.be/~tuytelaa/), [Lan Xu](https://www.xu-lan.com/index.html), [Minye Wu](https://wuminye.com/)  
> CVPR 2023
> 

#### [project page](https://aoliao12138.github.io/ReRF/) | [paper](https://arxiv.org/abs/2304.04452) | [data & model](https://github.com/aoliao12138/ReRF_Dataset)

<br>

## Changes Made in This Repository

Here, I detail the changes and extensions I made to the original implementation:

- Added utility code to transform multiview camera video data to fit the llff data format.
- Inspired by the DVGO extension idea, I implemented the model dmpigo in ReRF to be suitable for llff data.
- Added a new feature to render dynamic multiview videos by implementing the --render_video argument.

<br>

# Installation

```bash
git clone git@github.com:juhyeongdoyle/ReRF.git
cd ReRF
conda env create -f environment.yml
conda activate rerf
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter==2.0.9
```

<br>

# Datasets

This implementation supports data from the [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) repository.

Furthermore, the project includes a preprocessing process that allows the use of any real scene data captured in multiview.

To preprocess and use your multiview real scene data, first unzip your dataset to your desired directory and follow the preprocessing instructions provided in this project.

## Get Started
You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

#### original work - [project page](https://aoliao12138.github.io/ReRF/) | [paper](https://arxiv.org/abs/2304.04452) | [data & model](https://github.com/aoliao12138/ReRF_Dataset) 

<br>

## Data Preprocessing
Before starting the training process, you need to preprocess your multiview video data.

To preprocess, run:
```python
$ python llff_data_util.py --video_path [input data dir] --llff_path [output data dir] --num_frames 100
```

- --video_path: Set this to the directory containing the multiview camera videos and the poses_bounds.npy file.
- --llff_path: This is the directory where the converted data, suitable for training, will be saved.
- --num_frames: Specify the number of frames you wish to convert.


<b>Note</b> : If you do not have the poses_bounds.npy file, you can generate the camera pose data using COLMAP. For guidance on generating this data, please refer to the forward-facing scene tutorial using COLMAP [here](https://sunset1995.github.io/dvgo/tutor_forward_facing.html)

<br>
The expected data structure for the input should be:

```css
📦your multiview video data
 ┣ 📜00.mp4
 ┣ 📜01.mp4
 ...
 ┣ 📜13.mp4
 ┣ 📜14.mp4
 ┗📜poses_bounds.npy
```
After preprocessing, the converted data will be structured as:
```css
📦your llff data
 ┣ 📂0
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜image_0000.png
 ┃ ┃ ...
 ┃ ┃ ┗ 📜image_0014.png
 ┃ ┗ 📜poses_bounds.npy
 ┣ 📂1
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜image_0000.png
 ┃ ┃ ...
 ┃ ┃ ┗ 📜image_0014.png
 ┃ ┗ 📜poses_bounds.npy
 ...
```
By following this preprocessing step, you will have the data in the required format for training with this repository.

<br>

## Train
To train on your scene, first set the --config directory to match your data's directory configuration. Then, run:
```bash
$ LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH python run.py --config [YOUR_DATA_DIRECTORY_PATH]/[YOUR_CONFIG_FILE].py --render_test
```
Replace [YOUR_DATA_DIRECTORY_PATH] and [YOUR_CONFIG_FILE] with the appropriate directory and configuration file for your data.


<br>

## Rendering
render dynamic multiview video while camera and frames are changing, run:
```bash
$ LD_LIBRARY_PATH=./ac_dc:$LD_LIBRARY_PATH PYTHONPATH=./ac_dc/:$PYTHONPATH run.py --config [YOUR_DATA_DIRECTORY_PATH]/[YOUR_CONFIG_FILE].py --render_only --render_video 100
```

Just like in the training phase, set the --config with the appropriate data directory path. If you want to render using a saved model without training, use the --render_only option. Specify the maximum frame number you want to render using the --render_video argument.

Additionally, you can customize the rendering poses by modifying the dictionary inside the config file:

```python
movie_render_kwargs={
    'scale_r': 0.8,   # circling radius
    'scale_f': 8.0,   # the distance to the looking point of focus
    'zdelta': 0.1,    # amplitude of forward motion
    'zrate': 0.1,     # frequency of forward motion
    'N_rots': 1.0,    # number of rotations in 120 frames
    'N_views' : 97    # frame numbers
}
```

<br>


### Your own llff config files
As [DVGO](https://github.com/sunset1995/DirectVoxGO), check the comments in [`configs/default.py`](./configs/default.py) for the configurable settings.
We use [`mmcv`'s config system](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html).
To create a new config, please inherit `configs/default.py` first and then update the fields you want.
As [DVGO](https://github.com/sunset1995/DirectVoxGO), you can change some settings like `N_iters`, `N_rand`, `num_voxels`, `rgbnet_depth`, `rgbnet_width` and so on to achieve the speed and quality tradeoff.

### Extention to new dataset

The scale of the bbox will greatly affect the final result, so it is recommended to adjust it to fit your camera coordinate system.
You can follow [DVGO](https://github.com/sunset1995/DirectVoxGO#:~:text=Extention%20to%20new%20dataset) to adjust it.

<br>


## Acknowledgement

The code base is originated from the [ReRF](https://github.com/sunset1995/DirectVoxGO) implementation. I borrowed some codes from [DVGO](https://github.com/sunset1995/DirectVoxGO) 



