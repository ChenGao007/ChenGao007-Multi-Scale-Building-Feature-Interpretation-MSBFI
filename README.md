This model is a tool applied to the weighted interpretation of composite scale features of built heritage point clouds. The Pointnet++ model can extract representative features of different classes of point clouds at different scales through multiple SA (Set Abstraction) modules (SA1, SA2, and SA3 modules in this model). We extract the outputs of the three SA modules through the trained model and map the features of different modules, weighted according to the research needs of built heritage features (e.g., focusing on mass features, focusing on meso-scale features, focusing on detail-scale features) to the original point cloud by means of gradient-weighted aggregation. The sensitivity of the model to the research object is achieved by adjusting the weights of the features of different modules. In addition to its computational efficiency advantage, the model can assist researchers in the field of built heritage to quantitatively and accurately understand the representative features of the built heritage they study at different scales.

In this repository, we release the code and data for training a PointNet++ classification network on point clouds sampled from 3d shapes, as well as for training a part segmentation network on the ShapeNet Part dataset.

Citation:
If you find our work useful in your research, please consider citing: 
@article{Multi-Scale-Building-Feature-Interpretation-MSBFI,
      Author = {Xiaoyi Zu, Chen Gao},
      Title = {Multi-Scale-Building-Feature-Interpretation-MSBFI},
      Journal = { https://github.com/ChenGao007/ChenGao007-Multi-Scale-Building-Feature-Interpretation-MSBFI.git },
      Year = {2025}
}

The code has been tested with Python 3.9.20, 2.0.0+cu118 on Window11.

Installation:
pip install -r requirements.txt

Usage:
python train_pointnet2.py

To train a model to classify point clouds sampled from the given dataset: 

To generate the P_Grad_CAM:
python gradcam_pointnet2_sa_weighted.py

The weights of the different feature scales are adjusted by: 
Lines 19-21 in gradcam_pointnet2_sa_weighted.py: SA1_weight、SA_weight、SA3_weight


