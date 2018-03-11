#!/bin/bash

methods=(sgd batch_scale layer_scale DBN DBN_NoAlign)
lrs=(0.1 0.2 0.5 1 2)
groups=(16) 

n=${#methods[@]}
m=${#lrs[@]}
f=${#groups[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "group=${groups[$k]}"
   	th exp_MLP_2perform_PIE.lua -model_method ${methods[$i]} -learningRate ${lrs[$j]} -m_perGroup ${groups[$k]} -optimization simple -seed 1 -max_epoch 100
      done
   done
done
