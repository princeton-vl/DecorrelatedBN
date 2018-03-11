#!/bin/bash

methods=(sgd_F2 batch_F2 layer_F2 DBN_F2)
lrs=(0.1 0.2 0.5 1 2)
groups=(128) 

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
   	th exp_MLP_1FIM_YaleB.lua -model_method ${methods[$i]} -learningRate ${lrs[$j]} -m_perGroup ${groups[$k]} -optimization simple -seed 1 -max_epoch 3000 -FIM_intervalT 20
      done
   done
done
