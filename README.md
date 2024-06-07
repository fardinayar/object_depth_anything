## Metric Self-Supervised Monocular Depth Estimation for Object Distance Estimation

![demo](demo.gif)

Code for self-supervised monocular object depth estimation based on **[monodepth2](https://github.com/nianticlabs/monodepth2)**. We have replaced MonoDepth2's default encoder-decoder architecture with a pre-trained relative depth estimation VIT from **[Depth-Anything](https://github.com/LiheYoung/Depth-Anything)** and added the GPS loss from **[G2S](https://github.com/NeurAI-Lab/G2S)** to predict metric depth. Object detection is done by a YOLOv8 model trained on the COCO dataset.

There are many things to do, but for now, we have provided a trained model on the KITTI dataset without searching for the best training settings.

| Model                                        | Resolution | abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
| -------------------------------------------- | ---------- | ------- | ------ | ----- | -------- | ----- | ----- | ----- |
| **[G2S](https://github.com/NeurAI-Lab/G2S)** | 1024×384   | 0.109   | 0.844  | 4.774 | 0.194    | 0.869 | 0.958 | 0.981 |
| This Repo (G2S + Depth-Anything)             | 518×518    | 0.101   | 0.733  | 4.579 | 0.183    | 0.893 | 0.964 | 0.982 |

### Inference on  a custom image folder

1. Clone this repo and create the environment:

   ```bash
   git clone https://github.com/fardinayar/object_depth_anything.git
   conda env create -f environment.yml
   conda activate depth
   ```

2. Run 

   ```bash
   python object_depth_estimation.py --image_folder {path to image folder} --output_type {"csv" or "video"}
   ```

   In the case of "CSV", results will be saved as a CSV file (results.csv) that lists objects in each image with their minimum depth from the camera. In the case of the "video", the same results will be saved in a video (demo.gif).

### TODO

- [ ] Add code for training the depth estimation model on custom dataset
- [ ] Calculate distance instead of depth
- [ ] Evaluate the depth model only in object regions
- [ ] Add more inference settings
