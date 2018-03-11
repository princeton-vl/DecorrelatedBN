CUDA_VISIBLE_DEVICES=3  th exp_Conv_1vggA_2test.lua -model vggA_DBN -learningRate 0.005 -optimization adam -lrD_k 2000 -m_perGroup 16 -seed 1 
CUDA_VISIBLE_DEVICES=3  th exp_Conv_1vggA_2test.lua -model vggA_batch -learningRate 0.005 -optimization adam -lrD_k 2000 -m_perGroup 16 -seed 1 
