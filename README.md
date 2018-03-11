Decorrelated Batch Normalization
======================================
## Requirements and Dependency
* Install [Torch](http://torch.ch) with CUDA GPU
* Install [cudnn v5](http://torch.ch)
* Install dependent lua packages optnet by run:
luarocks install optnet
* Install Magga (you can follow the instruction in the file  ['Install Magga.txt'](./Install_Magga.txt) )
	Note: Magga is used for the SVD on GPU. If you don't install Magga, you can not run the code on GPU (For all the experiment on CNNs, we run the experiment on GPU)

## Experiments in the paper

#### 1.  Reproduce the results to show PCA whitening not work:
    
*	Run script:: 
```Bash
   bash execute_MLP_0debug_MNIST.sh
 ```
This script will download MNIST dataset automatically. The results will be saved at 'set_result/MLP/' directory. 
	
#### 2. Reproduce the results on MLP architecture:

##### (1).FIM experiments on YaleB dataset 

* Dataset preparations: you should download the [YaleB dataset](https://www.dropbox.com/sh/5pkrtv02wemqxzp/AADlVOs3vDMOEsOpRFa20Uqha?dl=0), and put the data file in the directory: './dataset/'

* execute:

```Bash
bash execute_MLP_1FIM_YaleB_best.sh
 ```
The results will be saved at  'set_result/MLP/' directory. Note that one can get the results by different hyper-parameters configurations by running scripts: 'execute_MLP_1FIM_YaleB_HyperP.sh' and 'execute_MLP_1FIM_YaleB_HyperP_nnn.sh'. 

##### (2). Experiments on PIE dataset 

* Dataset preparations: you should download the [PIE dataset](https://www.dropbox.com/sh/5pkrtv02wemqxzp/AADlVOs3vDMOEsOpRFa20Uqha?dl=0), and put the data file in the directory: './dataset/'

* For the effects of group size, execute:

```Bash
bash execute_MLP_2PIE_DBNGroup.sh
 ```

* For the performances of different baselines, execute:

```Bash
 bash execute_MLP_2PIE.sh
 bash execute_MLP_2PIE_nnn.sh
 ```
-----------------------------Note that the experiment above is under MLP and run on CPU, and therefore it is not necessary to install Magga for above experiment --------------------
 
#### 3. Reproduce the results on VGG style, BN-Incption and Wide residual network over CIFAR datset: 

 *	Dataset preparations: you should download the [CIFAR-10](https://yadi.sk/d/eFmOduZyxaBrT) and [CIFAR-100](https://yadi.sk/d/ZbiXAegjxaBcM) datasets, and put the data file in the directory: './dataset/' 
 * Execute: 
```Bash
th exp_GoogleNet_dataWhitening.lua –dataPath './dataset/cifar100_whitened.t7'
 ```
  *	To reproduce the experimental results, you can run the script below, which include all the information of experimental configurations: 
```Bash
  bash 2_execute_Conv_CIFAR_VggStyle.sh  
  bash 3_execute_Conv_CIFAR_BNInception.sh 
  bash 4_execute_Conv_CIFAR_wr.sh  
 ```
 


#### 4. Run the experiment on imageNet dataset. 

 *  (1) You should clone the facebook residual network project from:https://github.com/facebook/fb.resnet.torch
 *  (2) You should download imageNet dataset and put it on: '/tmp/dataset/imageNet/' directory (you also can change the path, which is set in 'opts_imageNet.lua')
 *  (3) Copy  'opts_imageNet.lua', 'exp_Conv_imageNet_expDecay.lua', 'train_expDecay.lua', 'module' and 'models' to the project's root path.
 *  (4)	Execute: 
```Bash
th exp_Conv_imageNet_expDecay.lua -model imagenet/resnet_OLM_L1
 ```
You can training other respective model by using the parameter 'model'

## Contact
huanglei@nlsde.buaa.edu.cn, Any discussions and suggestions are welcome!

