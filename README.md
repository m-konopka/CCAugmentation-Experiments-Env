# Data preprocessing and augmentation framework for crowd counting with convolutional neural network

Repository with code relevant to the engineering thesis realized at Gdansk University of Technology.

Implementation of crowd counting models are based on [C^3 Framework](https://github.com/gjy3035/C-3-Framework).

The repository is experiment envionment for [CCAugmentation Framework](https://github.com/pijuszczyk/CCAugmentation).

## Results

We obtained following results on ShanghaiTech part B dataset. To reproduce it please apply settings `./best-profiles/`.

|         |                                    | MAE   | RMSE  |
|---------|------------------------------------|-------|-------|
|   MCNN  | Original paper                     |  26.4 |  41.3 |
|         | C3 implementation                  |  21.5 |  38.1 |
|         | CCAugmentation (ours)              | 15.34 | 24.28 |
|  CSRNet | Original paper                     |  10.6 |  16.6 |
|         | C3 implementation                  |  10.6 |    16 |
|         | CCAugmentation (ours)              | 10.38 | 16.56 |
| C3F-VGG | Original paper / C3 implementation |  10.6 |  16.6 |
|         | CCAugmentation (ours)              |  9.32 | 14.23 |
|  SANet  | Original paper                     |   8.4 |  13.6 |
|         | C3 implementation                  |  12.1 |  19.2 |
|         | CCAugmentation (ours)              | 16.32 | 25.56 |

## Example predictions

Below we present example prediction for CSRNet model.

![img](./img/csrnet_9.84_15.14.png)

## Citation

```
@inproceedings{ccaugmentation,
  title={Data preprocessing and augmentation framework for crowd counting with convolutional neural network},
  author={Marcin Konopka, Piotr Juszczyk},
  year={2020}
}
```
```
@article{gao2019c,
  title={C$^3$ Framework: An Open-source PyTorch Code for Crowd Counting},
  author={Gao, Junyu and Lin, Wei and Zhao, Bin and Wang, Dong and Gao, Chenyu and Wen, Jun},
  journal={arXiv preprint arXiv:1907.02724},
  year={2019}
}
```