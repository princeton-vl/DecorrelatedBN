#!/bin/bash

methods=(vggA_batch vggA_DBN)
lrs=(0.001 0.005 0.01 0.05)
lrD_ks=(1000 2000 4000 8000) 

n=${#methods[@]}
m=${#lrs[@]}
f=${#lrD_ks[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "lrD_k=${lrD_ks[$k]}"
        CUDA_VISIBLE_DEVICES=0	th exp_Conv_1vggA_1validate.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -lrD_k ${lrD_ks[$k]} -optimization adam -seed 1 -max_epoch 2
      done
   done
done
