Decorrelated Batch Normalization
======================================
Code for reproducing the results in the following paper:

**Decorrelated Batch Normalization**  
Lei Huang, Dawei Yang, Bo Lang, Jia Deng  
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.*
[arXiv:1804.08450](https://arxiv.org/abs/1804.08450)

## Requirements and Dependency
* Install [Torch](http://torch.ch) with CUDA (for GPU)
* Install [cudnn v5](http://torch.ch)
* Install dependent lua packages optnet by:
```Bash
luarocks install optnet
 ```
* Install Magma (you can follow the instruction in the file  ['Install Magma.txt'](./Install_Magma.txt) ). 
	Note: Magma is used for the SVD on GPU. If you don't install Magma, you can not run the code on GPU (For all the experiment on CNNs, we run the experiment on GPU)

## Experiments in the paper

#### 1.  Reproduce the results to show PCA whitening not work:
    
*	Execute script: 
```Bash
   bash execute_MLP_0debug_MNIST.sh
 ```
This script will download MNIST dataset automatically and you should put the 'mnist.t7'(directory) to the './dataset/' directory. The results will be saved at the  directory: 'set_result/MLP/'. 
	
#### 2. Reproduce the results on MLP architecture:

##### (1). FIM experiments on YaleB dataset 

* Dataset preparation: you should download the [YaleB dataset](https://www.dropbox.com/s/taw9mlsq29eqv82/YaleB_Torch.zip?dl=0), and put the data files in the directory: './dataset/' (The final paths of the data files are:'./dataset/YaleB/YaleB_train.dat' and './dataset/YaleB/YaleB_test.dat')

* Execute:

```Bash
bash execute_MLP_1FIM_YaleB_best.sh
 ```
The results will be saved at directory:  'set_result/MLP/'. 

Note that one can get the results by different hyper-parameters configurations by running scripts: 'execute_MLP_1FIM_YaleB_HyperP.sh' and 'execute_MLP_1FIM_YaleB_HyperP_nnn.sh'. 

##### (2). Experiments on PIE dataset 

* Dataset preparations: you should download the [PIE dataset](https://www.dropbox.com/sh/5pkrtv02wemqxzp/AADlVOs3vDMOEsOpRFa20Uqha?dl=0), and put the data file in the directory: './dataset/' (The final paths of the data files are:'./dataset/PIE/PIE_train.dat' and './dataset/PIE/PIE_test.dat')

* For the effects of group size, execute:

```Bash
bash execute_MLP_2PIE_DBNGroup.sh
 ```

* For the performances of different baselines, execute:

```Bash
 bash execute_MLP_2PIE.sh
 bash execute_MLP_2PIE_nnn.sh
 ```
-----------------------------Note that the experiment above is under MLP and run on CPU, and therefore it is not necessary to install Magma for above experiment --------------------
 
#### 3. Reproduce the results on VGG-A architecture over CIFAR-10 datset: 
 *	Dataset preparations: you should follow the CIFAR-10 dataset pre-process as in this [project](https://github.com/szagoruyko/cifar.torch) ,which will generate a pre-processed dataset of 1400 Mb file. Put the cifar_provider.t7 file in the  directory: './dataset/'
 
* Execute: 
```Bash
bash execute_Conv_1vggA_2test_adam.sh
bash execute_Conv_1vggA_2test_base.sh
bash execute_Conv_1vggA_2test_ELU.sh
bash execute_Conv_1vggA_2test_var.sh
 ```
Note that if the running machine has <4 GPU, the parameters 'CUDA_VISIBLE_DEVICES' should be changed.

#### 4. Analyze the properties of DBN on CIFAR-10 datset: 
 *	Dataset preparations: The same datsets as VGG-A experiments
 
* Execute: 
```Bash
bash exp_Conv_4Splain_1deep.lua
bash exp_Conv_4Splain_2large.lua
 ```


#### 5. Residual network experiments on CIFAR-10 datset: 
 *	Dataset preparations: you should download the [CIFAR-10](https://yadi.sk/d/eFmOduZyxaBrT) and [CIFAR-100](https://yadi.sk/d/ZbiXAegjxaBcM) datasets, and put the data file in the directory: './dataset/' 
 * Execute: 
```Bash
bash execute_Conv_2residual_old.sh
bash execute_Conv_3residual_wide_Cifar100_wr_BN_d28_h48_g16_b128_dr0.3_s1_C2.sh
bash execute_Conv_3residual_wide_Cifar100_wr_DBN_scale_L1_d28_h48_g16_b128_dr0.3_s1_C3.sh
bash execute_Conv_3residual_wide_Cifar10_wr_BN_d28_h48_g16_b128_dr0.3_s1_C2.sh
bash execute_Conv_3residual_wide_Cifar10_wr_DBN_scale_L1_d28_h48_g16_b128_dr0.3_s1_C3.sh
 ```

 


#### 6. Run the experiment on imageNet dataset. 

 *  (1) You should clone the facebook residual network project from:https://github.com/facebook/fb.resnet.torch
 *  (2) You should download imageNet dataset and put it on: '/tmp/dataset/imageNet/' directory (you also can change the path, which is set in 'opts.lua')
 *  (3) We run the experiments on multiple GPUs, so DBN module should be compiled to the install directory of Torch. you need go the directory of './models/imagenet/cuSpatialDBN/' to execute: luarocks make cudbn-1.0-0.rockspec
  *  (4) copy the files of './models/imagenet/' direcotry ('preresnet_BN.lua', 'preresnet_DBN_scale_L1.lua' and 'init.lua') to the 'fb.resnet.torch' project's './models' directory.
 *  (5)	Execute the script with the default parameters configuration of the project: https://github.com/facebook/fb.resnet.torch


## Contact
huanglei@nlsde.buaa.edu.cn, Any discussions and suggestions are welcome!

