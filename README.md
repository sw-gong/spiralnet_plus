# SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator

SpiralNet++ is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape correspondence, classification or reconstruction. The basic idea is to serialize the neighboring vertices based on triangular meshes, which allows the model to learn intrinsic geometric features from fusing local geometric structure information with vertex features in a straightforward manner.

Note that the dataset of [4DFAB](https://arxiv.org/abs/1712.01443) used for evaluating classification performance in this paper has not yet been released. We will open source our code immediately once the dataset is publicly accessible. However, it should be easy to develop a shape classification model with our framework. Feel free to raise issues if you meet problems when applying SpiralNet++ to your problems.

## SpiralConv
SpiralConv is defined in an equivalent manner to the euclidean CNNs by utilizing the nature of the *fixed* spiral serialization of neighboring vertices. We define it as:
<p align="center"><img src="svgs/a9e11ad58a629e9fb55045f0ac158a64.svg" align=middle width=197.04795pt height=49.131389999999996pt/></p>

where <img src="svgs/11c596de17c342edeed29f489aa4b274.svg" align=middle width=9.388665000000001pt height=14.102549999999994pt/> denotes MLPs and <img src="svgs/f2d94cd21b8f8c1c0d6e60b36522ae2e.svg" align=middle width=8.188554000000002pt height=24.56552999999997pt/> is the concatenation operation. Note that we concatenate node features in the spiral sequence following the order defined in <img src="svgs/524d74b8294edeb6931ea94a8ea50f24.svg" align=middle width=39.977025pt height=24.56552999999997pt/>.

To allow the model capture multi-scale resolution equivalently to the dilated CNNs in the Euclidean domain, the implementation of the dilated SpiralConv is straightforward without increased parameters as well as computational resources. You can customize the variable ``dilation`` to gain the power of the dilated spiral convolution.

## Sampling Operation

In the experiments of facial expression classification and face reconstruction, we perform the in-network sampling operations on mesh with the precomputed sampling matrices introduced in [this paper](https://arxiv.org/abs/1807.10267). The down-sampling matrix D is obtained from iteratively contracting vertex pairs that maintain surface error approximations using quadric matrics, and the up-sampling matrix U is obtained from including barycentric coordinates of the vertices that are discarded during the downsampling. It can be simply defined as:
<p align="center"><img src="svgs/495643a79495f6d3ce50d4936365a15e.svg" align=middle width=77.33054999999999pt height=13.156093499999999pt/></p>

where the *sparse* sampling matrix <img src="svgs/9180e00e196978aa798f62467e585afa.svg" align=middle width=80.2329pt height=30.950700000000015pt/> and node feature matrix <img src="svgs/281195f9409164ae6087fe6f0131dcb6.svg" align=middle width=98.84704500000001pt height=27.598230000000008pt/>.

The real magic of our implemtation happens in the body of ``reconstruction.network.Pool``.  Here, we need to perform batch matrix multiplication on GPU w.r.t the sampling operation described above. Because dense matrix multiplication is really slow, we implement **sparse batch matrix multiplication** via scattering add node feature vectors corresponds to *cluster nodes* across a batch of input node feature matrices.

## Installation

The code is developed using Python 3.6 on Ubuntu 16.04. The models were trained and tested with NVIDIA 2080 Ti.
* [Pytorch](https://pytorch.org/) (1.3.0)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) (1.3.0)
* [OpenMesh](https://github.com/nmaxwell/OpenMesh-Python) (1.1.3)
* [MPI-IS Mesh](https://github.com/MPI-IS/mesh): We suggest to install this library from the source.
* [tqdm](https://github.com/tqdm/tqdm)

## 3D Shape Correspondence on FAUST
```
python -m correspondence.main
```

## 3D Face Reconstruction on CoMA
To reproduce the experiment in our paper, you can simply run the script below. In our paper, we only show results of the *Interpolation Experiments* as described in the [paper](https://arxiv.org/abs/1807.10267). However, you can also run experiments under the *Extrapolation Experiment* setting by defining the variable ``split`` of ``extrapolation`` or ``interpolation``. Note that you need to set a test expression (``test_exp``) if performing extrapolation experiment.
```
python -m reconstruction.main
```
The checkpoints of each epoch is saved in the corresponding output folder (specifed by the vairable ``exp_name``).  After training, it outputs the result of the "Mean Error with the Standard Deviation" (in millimeters) as well as "Median Error", which are saved in the file ``euc_error.txt``.

## Data
To create your own dataset, you have to provide data attributes at least:
- `data.x`: Node feature matrix with shape `[num_nodese, num_node_features]` and type `torch.float`.
- `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`. Note that to use this framework, the graph connectivity across all the meshes should be the same.

where `data` is inherited from `torch_geometric.data.Data`. Have a look at the classes of `datasets.FAUST` and `datasets.CoMA` for an example.

Alternatively, you can simply create a regular python list holding `torch_geometric.data.Data` objects

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
