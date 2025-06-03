# 数据集格式修改后的建议

原代码有自己的数据集格式和加载方式，我们的数据集是将所有数据保存在一个npy文件中，同时原代码显示数据的方式也和我们的不同，所以在指标上看虽然我们的更高，但对比图上我们的更差。

数据集加载类应该用：`datasets/nyu.py`中的`NYU_v2_datsetForGeoDSR`。

测试文件:`test.py`

重新测试后的指标：

|      | 4×   | 8×   | 16×  |
| ---- | ---- | ---- | ---- |
| RMSE | 1.46 | 2.70 | 5.14 |

具体结果可以看`result`目录下的日志，适应前的日志可以查看:`workspace/log_GASA.txt`

# GeoDSR

## Learning Continuous Depth Representation via Geometric Spatial Aggregator
## Accepted to AAAI 2023 [[Paper]](http://arxiv.org/abs/2212.03499)
Xiaohang Wang*, Xuanhong Chen*, Bingbing Ni**, Zhengyan Tong, Hang Wang

\* Equal contribution

\*\* Corresponding author


**The official repository with Pytorch**

- This work is for arbitrary-scale RGB-guided depth map super-resolution (DSR).
- Depth map super-resolution (DSR) has been a fundamental task for 3D computer vision. While arbitrary scale DSR is a more realistic setting in this scenario, previous approaches predominantly suffer from the issue of inefficient real-numbered scale upsampling.

[![logo](/docs/img/geodsrlogo.png)](https://github.com/nana01219/GeoDSR)

## Results:
[![results](/docs/img/code.jpg)](https://github.com/nana01219/GeoDSR)
## Dependencies
- python3.7+
- pytorch1.9+
- torchvision
- [Nvidia Apex](https://github.com/NVIDIA/apex) (python-only build is ok.)


### Datasets
We follow [Tang et al.](https://github.com/ashawkey/jiif) and use the same datasets. Please refer to [here](https://github.com/ashawkey/jiif/blob/main/data/prepare_data.md) to download the preprocessed datasets and extract them into `data/` folder.

### Pretrained Models
- Baidu Netdisk (百度网盘)：https://pan.baidu.com/s/1e2rLQFqVHIy2ZZG922XNTA 
- Extraction Code (提取码)：xu7e

- Google Drive: https://drive.google.com/drive/folders/1cIvA_AYh0fve_pDhN6timhCeN6A7MhD2?usp=share_link

Please put the model under `workspace/checkpoints` folder.

### Train
```
python main.py
```
### Test
```
bash test.sh
```



## Licesnse
For academic and non-commercial use only. The whole project is under the MIT license. See [LICENSE](https://github.com/nana01219/GeoDSR/blob/main/LICENSE) for additional details.


## Citation
If you find this project useful in your research, please consider citing:

```
@misc{GeoDSR,
  author = {Wang, Xiaohang and Chen, Xuanhong and Ni, Bingbing and Tong, Zhengyan and Wang, Hang},
  title = {Learning Continuous Depth Representation via Geometric Spatial Aggregator},
  publisher = {arXiv},
  year = {2022}
}
```

## Ackownledgements
This code is built based on [JIIF](https://github.com/ashawkey/jiif). We thank the authors for sharing the codes.

## Related Projects

Learn about our other projects 

[[EQSR]](https://github.com/neuralchen/EQSR): high-quality arbitrary-scale image super-resolution;

[[VGGFace2-HQ]](https://github.com/NNNNAI/VGGFace2-HQ): high resolution face dataset VGGFace2-HQ;

[[RainNet]](https://neuralchen.github.io/RainNet);

[[Sketch Generation]](https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale);

[[SimSwap]](https://github.com/neuralchen/SimSwap): most popular face swapping project;

[[ASMA-GAN]](https://github.com/neuralchen/ASMAGAN): high-quality style transfer project;