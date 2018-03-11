#!/bin/bash
methods=(sgd DBN DBN_NoAlign)

lrs=(0.1 0.5 1 5)
layers=(1 3)

n=${#methods[@]}
m=${#lrs[@]}
f=${#layers[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "nolayerinear=${layers[$j]}"
   	th exp_MLP_0debug_MNIST.lua -model_method ${methods[$i]} -learningRate ${lrs[$j]} -layer ${layers[$k]}  -optimization simple -seed 1 -batchSize 50000 -max_epoch 1000
      done
   done
done
