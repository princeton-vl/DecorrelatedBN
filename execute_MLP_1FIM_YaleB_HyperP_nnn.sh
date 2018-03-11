#!/bin/bash

epsilos=(0.001 0.01 0.1 1)
lrs=(0.1 0.2 0.5 1 2)
Ts=(50 100 200 500) 

n=${#epsilos[@]}
m=${#lrs[@]}
f=${#Ts[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "epsilos=${epsilos[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "T=${Ts[$k]}"
   	th exp_MLP_1FIM_YaleB.lua -model_method nnn_F2 -epcilo  ${epsilos[$i]} -learningRate ${lrs[$j]} -T ${Ts[$k]} -optimization simple -seed 1 -max_epoch 3000 -FIM_intervalT 20
      done
   done
done
