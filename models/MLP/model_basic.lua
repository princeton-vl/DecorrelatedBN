--[[
-- Author: Lei Huang
--mail: huanglei@nlsde.buaa.edu.cn
--
--]]


require 'module/BatchLinear_FIM'
require 'module/NormLinear_new'
require 'module/DecorelateBN_Advance'
require 'module/DecorelateBN_NoAlign'

require 'module/LayerNorm'
require 'module/Affine_module'

function create_model(opt)

  local model=nn.Sequential()          
  --local cfg_hidden=torch.Tensor({128,128,128,128,128})
  local cfg_hidden=torch.Tensor({opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number})
  local n=cfg_hidden:size(1)
  
  local nonlinear 
  if opt.mode_nonlinear==0 then  --sigmod
      nonlinear=nn.Sigmoid
  elseif opt.mode_nonlinear==1 then --tanh
      nonlinear=nn.Tanh
  elseif opt.mode_nonlinear==2 then --ReLU
     nonlinear=nn.ReLU
  elseif opt.mode_nonlinear==3 then --ELU
     nonlinear=nn.ELU
  end 
  
  local linear=nn.Linear
  local module_BN=nn.BatchNormalization
  local module_LN=nn.LayerNorm
  local module_GN=nn.GlobalNormalization
  local module_DBN=nn.DecorelateBN_Advance
  local module_nnn=nn.NormLinear_new
  local module_affine=nn.Affine_module
  

  local function block_plain(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  
  ----------------------------------batch normalization related----------------------------------------
  
  local function block_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input,_,_,false))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  local function block_batch_scale(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input,_,_,true))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  
  local function block_batch_scale_var(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_BN(n_input,_,_,true))
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end

------------------------------------------layer normalization related-----------------------
  
  local function block_layer(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_LN(n_input,false))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  local function block_layer_scale(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_LN(n_input,false))
    s:add(module_affine(n_input))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  
  local function block_layer_scale_var(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_LN(n_input,false))
    s:add(module_affine(n_input))
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  
 --------------------------------------------------nnn realted--------------------------
  local function block_nnn(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_nnn(n_input,n_output,false))
    return s
  end  
 --------------------------------------------------DBN realted--------------------------
  
  local function block_DBN(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_DBN(n_input,opt.m_perGroup,false))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end    
  local function block_DBN_scale(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_DBN(n_input,opt.m_perGroup,true))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end    
  
  local function block_DBN_scale_var(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_DBN(n_input,opt.m_perGroup,true))
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end 
  
  local function block_DBN_NoAlign(n_input, n_output)
    local s=nn.Sequential()
    s:add(nn.DecorelateBN_NoAlign(n_input,opt.m_perGroup,true))
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
  
  local function block_DBN_NoAlign_var(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.DecorelateBN_NoAlign(n_input,opt.m_perGroup,true))
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end  
  
  
   model:add(linear(opt.n_inputs,cfg_hidden[1],opt.orth_intial))

-----------------------------------------model configure-------------------

  
  if opt.model_method=='sgd' then 
    for i=1,n do
       if i==n then
         model:add(block_plain(cfg_hidden[i],opt.n_outputs)) 
       else
        model:add(block_plain(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
 ------------------------------------------------------configure batch---------------------- 
  elseif opt.model_method=='batch' then   
     for i=1,n do
       if i==n then
         model:add(block_batch(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_batch(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
   
  elseif opt.model_method=='batch_scale' then   
     for i=1,n do
       if i==n then
         model:add(block_batch_scale(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_batch_scale(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
     
  elseif opt.model_method=='batch_scale_var' then   
     for i=1,n do
       if i==n then
         model:add(block_batch_scale_var(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_batch_scale_var(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
 ------------------------------------------------------configure nnn----------------------  
  elseif opt.model_method=='nnn' then   
     for i=1,n do
       if i==n then
        -- model:add(block_nnn(cfg_hidden[i],opt.n_outputs)) 
          model:add(nonlinear())
          model:add(module_nnn(cfg_hidden[i],opt.n_outputs,false))
       else
        -- model:add(block_nnn(cfg_hidden[i],cfg_hidden[i+1])) 
        model:add(nonlinear())
          model:add(module_nnn(cfg_hidden[i],cfg_hidden[i+1],false))
       end
     end
     
------------------------------------------------------configure DBN----------------------     
  elseif opt.model_method=='DBN' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
     
  elseif opt.model_method=='DBN_scale' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN_scale(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN_scale(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
     
 
  elseif opt.model_method=='DBN_var' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN_var(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN_var(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end  
     
  elseif opt.model_method=='DBN_scale_var' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN_scale_var(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN_scale_var(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end  
     
  elseif opt.model_method=='DBN_NoAlign' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN_NoAlign(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN_NoAlign(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end  
     
  elseif opt.model_method=='DBN_NoAlign_var' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN_NoAlign_var(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN_NoAlign_var(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end    
     
-----------------------------------------configure layer normalization----------------------     
  elseif opt.model_method=='layer' then   
     for i=1,n do
       if i==n then
         model:add(block_layer(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_layer(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
   
  elseif opt.model_method=='layer_scale' then   
     for i=1,n do
       if i==n then
         model:add(block_layer_scale(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_layer_scale(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end
     
  elseif opt.model_method=='layer_var' then   
     for i=1,n do
       if i==n then
         model:add(block_layer_var(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_layer_var(cfg_hidden[i],cfg_hidden[i+1])) 
       end
     end     
   elseif opt.model_method=='layer_scale_var' then   
     for i=1,n do
       if i==n then
         model:add(block_layer_scale_var(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_layer_scale_var(cfg_hidden[i],cfg_hidden[i+1])) 
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

