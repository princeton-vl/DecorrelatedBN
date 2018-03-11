CUDA_VISIBLE_DEVICES=2  th exp_Conv_1vggA_2test.lua -model vggA_DBN -learningRate 4 -optimization simple -lrD_k 2000 -m_perGroup 16 -seed 1 
CUDA_VISIBLE_DEVICES=2  th exp_Conv_1vggA_2test.lua -model vggA_batch -learningRate 4 -optimization simple -lrD_k 2000 -m_perGroup 16 -seed 1 
