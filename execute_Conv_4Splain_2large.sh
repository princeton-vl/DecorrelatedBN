#!/bin/bash

methods=(S1_p_DBN S1_p_BN)
lrs=(0.4)
depths=(44 32 20) 

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
        CUDA_VISIBLE_DEVICES=0	th exp_Conv_4Splain_2large.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -depth ${depths[$k]}  -seed 1 
      done
   done
done
