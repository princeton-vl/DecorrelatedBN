--[[
--
-- Author: Lei Huang
--mail: huanglei@nlsde.buaa.edu.cn
--
--]]
require 'module/Linear_ForDebug'
require 'module/Linear_Validation'
require 'module/BatchLinear_FIM'
require 'module/NormLinear_Validation'
require 'module/DecorelateBN_Advance'
require 'module/DecorelateBN_NoAlign'
require 'module/LayerNorm'
require 'module/Affine_module'

function create_model(opt)
  ------------------------------------------------------------------------------

  local model=nn.Sequential()          
 local cfg_hidden=torch.Tensor({128,64,48,48})
  --local cfg_hidden=torch.Tensor({opt.n_hidden_number,opt.n_hidden_number})
  local n=cfg_hidden:size(1)
  
  local nonlinear 
  if opt.mode_nonlinear==0 then  --sigmod
      nonlinear=nn.Sigmoid
  elseif opt.mode_nonlinear==1 then --tanh
      nonlinear=nn.Tanh
  elseif opt.mode_nonlinear==2 then --ReLU
     nonlinear=nn.ReLU
  elseif opt.mode_nonlinear==3 then --ReLU
     nonlinear=nn.ELU
  end 
  
  local linear=nn.Linear_ForDebug
  local module_BN=nn.BatchLinear_FIM
  local module_LN=nn.LayerNorm
  local module_DBN=nn.DecorelateBN_Advance
  local module_nnn=nn.NormLinear_Validation
  local module_affine=nn.Affine_module
 
  local function block_sgd(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  local function block_sgd_FIM(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Linear_Validation(n_input,n_output,true,true,true,false,opt.orth_intial))
    return s
  end
 

  local function block_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input,true)) 
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  local function block_batch_FIM(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input,true)) 
    s:add(nonlinear())
    s:add(nn.Linear_Validation(n_input,n_output,true,true,true,false,opt.orth_intial))

    return s
  end


  local function block_layer(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_LN(n_input,true))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  local function block_layer_FIM(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_LN(n_input,true))
    s:add(nonlinear())
    s:add(nn.Linear_Validation(n_input,n_output,true,true,true,false,opt.orth_intial))

    return s
  end




  local function block_nnn(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_nnn(n_input,n_output,false,false))
    return s
  end

  local function block_nnn_FIM(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_nnn(n_input,n_output,false))
    return s
  end


  local function block_DBN(n_input, n_output)
    local s=nn.Sequential()
    if opt.m_perGroup ~=0 then
       s:add(module_DBN(n_input,opt.m_perGroup,true))
    else
       s:add(module_DBN(n_input,n_input,true))
    end
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end

  local function block_DBN_FIM(n_input, n_output)
    local s=nn.Sequential()
    if opt.m_perGroup ~=0 then
       s:add(module_DBN(n_input,opt.m_perGroup,true))
    else
       s:add(module_DBN(n_input,n_input,true))
    end
    s:add(nonlinear())
    --s:add(linear(n_input,n_output,opt.orth_intial))
    s:add(nn.Linear_Validation(n_input,n_output,true,true,true,false,opt.orth_intial))
    return s
  end




-----------------------------------------model configure-------------------

  if opt.model_method=='sgd' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_sgd_FIM(cfg_hidden[i],opt.n_outputs)) 

       else
        model:add(block_sgd(cfg_hidden[i],cfg_hidden[i+1])) 

       end
     end 
  elseif opt.model_method=='sgd_F2' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_sgd_FIM(cfg_hidden[i],opt.n_outputs)) 

       elseif i==n-1 then
    
        model:add(block_sgd_FIM(cfg_hidden[i],cfg_hidden[i+1])) 
       else
        model:add(block_sgd(cfg_hidden[i],cfg_hidden[i+1])) 

       end
     end 
  elseif opt.model_method=='batch_F2' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_batch_FIM(cfg_hidden[i],opt.n_outputs)) 

       elseif i==n-1 then
         model:add(block_batch_FIM(cfg_hidden[i],cfg_hidden[i+1])) 
       else
        model:add(block_batch(cfg_hidden[i],cfg_hidden[i+1])) 

       end
     end 

  elseif opt.model_method=='layer_F2' then
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_layer_FIM(cfg_hidden[i],opt.n_outputs))

       elseif i==n-1 then
         model:add(block_layer_FIM(cfg_hidden[i],cfg_hidden[i+1]))
       else
        model:add(block_layer(cfg_hidden[i],cfg_hidden[i+1]))

       end
     end

  elseif opt.model_method=='DBN' then
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_DBN(cfg_hidden[i],opt.n_outputs))

       else
        model:add(block_DBN(cfg_hidden[i],cfg_hidden[i+1]))

       end
     end

  elseif opt.model_method=='DBN_F2' then
       model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
    for i=1,n do
       if i==n then
         model:add(block_DBN_FIM(cfg_hidden[i],opt.n_outputs))

       elseif i==n-1 then
         model:add(block_DBN_FIM(cfg_hidden[i],cfg_hidden[i+1]))
       else
        model:add(block_DBN(cfg_hidden[i],cfg_hidden[i+1]))

       end
     end



  elseif opt.model_method=='nnn' then
   model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))

     for i=1,n do
       if i==n then
         model:add(block_nnn_FIM(cfg_hidden[i],opt.n_outputs))


       else
         model:add(block_nnn(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end
  elseif opt.model_method=='nnn_F2' then
   model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))

     for i=1,n do
       if i==n then
         model:add(block_nnn_FIM(cfg_hidden[i],opt.n_outputs))

       elseif i==n-1 then
         model:add(block_nnn_FIM(cfg_hidden[i],cfg_hidden[i+1]))

       else
         model:add(block_nnn(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end

  end
  
  
  model:add(nn.LogSoftMax()) 
 

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local  criterion = nn.ClassNLLCriterion()

  return model, criterion
end
