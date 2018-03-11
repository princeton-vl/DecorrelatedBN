#!/bin/bash

methods=(S1_p_BN S1_p_DBN)
lrs=(0.1)
depths=(20 32 44) 

n=${#methods[@]}
m=${#lrs[@]}
f=${#depths[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "depth=${depths[$k]}"
        CUDA_VISIBLE_DEVICES=5	th exp_Conv_4Splain_1deep.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -depth ${depths[$k]}  -seed 1 
      done
   done
done
