# SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator

SpiralNet++ is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape correspondence, classification or reconstruction. The basic idea is to serialize the neighboring vertices based on triangular meshes, which allows the model to learn intrinsic geometric features from fusing local geometric structure information with vertex features in a straightforward manner.

## Installation
```bash
pyhon install -r requirements.txt
```
Install [MPI-IS/mesh](https://github.com/MPI-IS/mesh) for generating down- and up- sampling matrices.

## 3D Shape Correspondence on FAUST
```
python -m correspondence.main
```

## 3D Shape Reconstruction on CoMA
```
python -m reconstruction.main
```

## Data
To use your own data, you have to provide at least:
- `data.x`: Node feature matrix with shape `[num_nodese, num_node_features]` and type `torch.float`.
- `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`.

where `data` is inherited from `torch_geometric.data.Data`. Have a look at the classes of `datasets.FAUST` and `datasets.CoMA` for an example.

## Citation
Please cite [our paper](https://arxiv.org/abs/1911.05856) if you use this code in your own work:
```
@inproceedings{gong2019spiralnet++,
  title={SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator},
  author={Gong, Shunwang and Chen, Lei and Bronstein, Michael and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
