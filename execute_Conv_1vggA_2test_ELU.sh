CUDA_VISIBLE_DEVICES=1  th exp_Conv_1vggA_2test.lua -model vggA_DBN_ELU -learningRate 4 -optimization simple -lrD_k 2000 -m_perGroup 16 -seed 1 
CUDA_VISIBLE_DEVICES=1  th exp_Conv_1vggA_2test.lua -model vggA_batch_ELU -learningRate 4 -optimization simple -lrD_k 2000 -m_perGroup 16 -seed 1 
