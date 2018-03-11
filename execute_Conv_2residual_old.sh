#!/bin/bash
#
#
#

methods=(old_r_BN old_r_DBN_L1 old_r_DBN_scale_L1)
depths=(56 44 32 20)

batchSize=128
weightDecay=0.0001
dr=0
widen_factor=1
nN=0
maxEpoch=160
eStep="{80,120}"
learningRateDecayRatio=0.1


n=${#methods[@]}
m=${#depths[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	

    	echo "methods=${methods[$i]}"
    	echo "depths=${depths[$j]}"
   CUDA_VISIBLE_DEVICES=1	th exp_Conv_2residual_old.lua -model ${methods[$i]} -learningRate 0.1 -depth ${depths[$j]} -max_epoch ${maxEpoch} -seed 1 -dropout ${dr} -m_perGroup 8 -batchSize ${batchSize} -weightDecay ${weightDecay} -widen_factor ${widen_factor}  -noNesterov ${nN} -epoch_step ${eStep} -learningRateDecayRatio ${learningRateDecayRatio}
   done
done
