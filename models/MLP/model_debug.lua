--[[
 --Author: Lei Huang
 --mail: huanglei@nlsde.buaa.edu.cn
--]]
 

require 'module/DecorelateBN_Advance'
require 'module/DecorelateBN_NoAlign'




function create_model(opt)
  ------------------------------------------------------------------------------
   

  local model=nn.Sequential()          
  config_table={}
   for i=1, opt.layer do
     table.insert(config_table, opt.n_hidden_number)
   end
  local cfg_hidden=torch.Tensor(config_table)
  
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
  local module_DBN=nn.DecorelateBN_Advance

  local function block_plain(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(linear(n_input,n_output,opt.orth_intial))
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
  
  local function block_DBN_NoAlign(n_input, n_output)
    local s=nn.Sequential()
    s:add(nn.DecorelateBN_NoAlign(n_input,opt.m_perGroup,true))
    s:add(nonlinear())
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
------------------------------------------------------configure DBN----------------------     
  elseif opt.model_method=='DBN' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN(cfg_hidden[i],opt.n_outputs)) 
       else
         model:add(block_DBN(cfg_hidden[i],cfg_hidden[i+1])) 
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
  elseif opt.model_method=='DBN_NoAlign_Last' then   
     for i=1,n do
       if i==n then
         model:add(block_DBN_NoAlign(cfg_hidden[i],opt.n_outputs)) 
         model:add(nn.DecorelateBN_NoAlign(opt.n_outputs,opt.m_perGroup,false)) 
      else
         model:add(block_DBN_NoAlign(cfg_hidden[i],cfg_hidden[i+1])) 
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

