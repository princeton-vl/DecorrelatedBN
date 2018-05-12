
Decorrelated Batch Normalization
======================================
Code for reproducing the results in the following paper:

**Decorrelated Batch Normalization**  
Lei Huang, Dawei Yang, Bo Lang, Jia Deng  
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.*
[arXiv:1804.08450](https://arxiv.org/abs/1804.08450)

## Requirements and Dependency
* Install MAGMA (you can find the instructions in  ['Install MAGMA.md'](./Install_MAGMA.md) ). 
Note: MAGMA is required for SVD on GPU. Without MAGMA, you can run the code on CPU only, while all the CNN experiments in the paper are run on GPU.
* Install [Torch](http://torch.ch) with CUDA (for GPU). Note that `cutorch` should be compiled with MAGMA support if you have installed MAGMA and set the environments correctly.
* Install [cudnn v5](http://torch.ch).
* Install the dependency `optnet` by:
```Bash
luarocks install optnet
 ```

## Experiments

#### 1.  Reproduce the results for PCA whitening:
    
*	Run:
```Bash
bash execute_MLP_0debug_MNIST.sh
 ```
This script will download MNIST automatically and you should put the `mnist.t7/` under `./dataset/`. The experiment results will be saved at `./set_result/MLP/`.
	
#### 2. Reproduce the results for MLP architecture:

##### (1) FIM experiments on YaleB dataset 
* Prepare the data: download the YaleB dataset [here](https://www.dropbox.com/s/taw9mlsq29eqv82/YaleB_Torch.zip?dl=0), and put the data files under `/dataset/` so that the paths look like `./dataset/YaleB/YaleB_train.dat` and `./dataset/YaleB/YaleB_test.dat`.
* Run:
```Bash
bash execute_MLP_1FIM_YaleB_best.sh
 ```
The experiment results will be saved at directory:  'set_result/MLP/'. 

You can experiment with different hyperparameters by running these scripts --  `execute_MLP_1FIM_YaleB_HyperP.sh` and `execute_MLP_1FIM_YaleB_HyperP_nnn.sh`.

##### (2) Experiments on PIE dataset 

* Prepare the data: download the PIE dataset [here](https://www.dropbox.com/sh/5pkrtv02wemqxzp/AADlVOs3vDMOEsOpRFa20Uqha?dl=0), and put the data file under `./dataset/` such that the paths look like `./dataset/PIE/PIE_train.dat` and `./dataset/PIE/PIE_test.dat`.
* To experiment with different group sizes, run:
```Bash
bash execute_MLP_2PIE_DBNGroup.sh
 ```

* To obtain different baseline performances, execute:

```Bash
 bash execute_MLP_2PIE.sh
 bash execute_MLP_2PIE_nnn.sh
 ```
 
Note that the experiments until this point can be run on CPU, so MAGMA is not needed in above experiments.

 --------------------
 
#### 3. Reproduce the results for VGG-A architecture on CIFAR-10: 
 *	Prepare the data: follow the instructions for CIFAR-10 in [this project](https://github.com/szagoruyko/cifar.torch) . It will generate a preprocessed dataset and save a 1400MB file. Put this file `cifar_provider.t7` under `./dataset/`.
* Run: 
```Bash
bash execute_Conv_1vggA_2test_adam.sh
bash execute_Conv_1vggA_2test_base.sh
bash execute_Conv_1vggA_2test_ELU.sh
bash execute_Conv_1vggA_2test_var.sh
 ```
Note that if your machine has fewer than 4 GPUs, the environment variable `CUDA_VISIBLE_DEVICES` should be changed accordingly.

#### 4. Analyze the properties of DBN on CIFAR-10 datset: 
*	Prepare the data: same as in VGG-A experiments.
* Run: 
```Bash
bash exp_Conv_4Splain_1deep.lua
bash exp_Conv_4Splain_2large.lua
 ```

#### 5. Reproduce the ResNet experiments on CIFAR-10 datset: 
 *	Prepare the data: download [CIFAR-10](https://yadi.sk/d/eFmOduZyxaBrT) and [CIFAR-100](https://yadi.sk/d/ZbiXAegjxaBcM), and put the data files under `./dataset/`.
 * Run: 
```Bash
bash execute_Conv_2residual_old.sh
bash execute_Conv_3residual_wide_Cifar100_wr_BN_d28_h48_g16_b128_dr0.3_s1_C2.sh
bash execute_Conv_3residual_wide_Cifar100_wr_DBN_scale_L1_d28_h48_g16_b128_dr0.3_s1_C3.sh
bash execute_Conv_3residual_wide_Cifar10_wr_BN_d28_h48_g16_b128_dr0.3_s1_C2.sh
bash execute_Conv_3residual_wide_Cifar10_wr_DBN_scale_L1_d28_h48_g16_b128_dr0.3_s1_C3.sh
 ```


#### 6. Reproduce the ImageNet experiments. 

 *  Clone Facebook's ResNet repo [here](https://github.com/facebook/fb.resnet.torch).
 *  Download ImageNet and put it in: `/tmp/dataset/ImageNet/` (you can also customize the path in `opts.lua`)
 *  Install the DBN module to Torch as a Lua package: go to the directory `./models/imagenet/cuSpatialDBN/` and run  `luarocks make cudbn-1.0-0.rockspec`.
  * Copy the model definitions in `./models/imagenet/` (`resnet_BN.lua`, `resnet_DBN_scale_L1.lua` and `init.lua`) to `./models` directory in the cloned repo `fb.resnet.torch`, for reproducing the results reported in the paper. You also can compare the pre-activation version of residual networks introduced in the [paper](https://arxiv.org/abs/1603.05027) (using the model files 
  `preresnet_BN.lua` and `preresnet_DBN_scale_L1.lua`).  
 * Use the default configuration and our models to run experiments.


## Contact
Email: huanglei@nlsde.buaa.edu.cn. Any discussions and suggestions are welcome!

