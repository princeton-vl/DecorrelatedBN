
--[[
   The Basic Decorelated Batch normalization version, in which:
   (1) use ZCA to whitening the activation
   (2) include train mode and test mode. in training mode, we train the module
   (3)  only for 2D input in MLP architecture and running on CPU
 
   Author: Lei Huang
   mail: huanglei@nlsde.buaa.edu.cn
]]--
local DecorelateBN_Advance,parent = torch.class('nn.DecorelateBN_Advance', 'nn.Module')

function DecorelateBN_Advance:__init(nDim, m_perGroup, affine, epsilon, momentum)
   parent.__init(self)
   

   --the parameter 'affine' is used to scale the normalized output. If true, scale the output.
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = false
   end

   self.epsilon =  epsilon or 1e-7 --used for revise the correlation matrix, in case of there are eigenValue is 0

   --the parameters 'm_perGroup' indicates the number in each group, which is used for group wised whitening
   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup==0 and nDim or m_perGroup>nDim and nDim or m_perGroup 
   else
     self.m_perGroup =  nDim/2 
   end 
   print('m_perGroup:'.. self.m_perGroup)


   self.nDim=nDim --the dimension of the input   
   self.running_means={}  --the mean used for inference, which is estimated based on each mini-batch with running average
   self.running_projections={} -- the whitening matrix used for inference, which is also estimated based on each mini-batch with running average
   self.momentum = momentum or 0.1 -- running average momentum
   

 
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
   ------------allow nDim % m_perGropu !=0-----
   for i=1,groups do
      if i<groups then --not the last group, so the number in each group is m_perGroups
        local r_mean=torch.zeros(self.m_perGroup)
        local r_projection=torch.eye(self.m_perGroup)
        table.insert(self.running_means, r_mean)
        table.insert(self.running_projections,r_projection)
      else --the last group, the number is nDim-(groups-1)*self.m_perGroup
        local r_mean=torch.zeros(nDim-(groups-1)*self.m_perGroup)
        local r_projection=torch.eye(nDim-(groups-1)*self.m_perGroup)
        table.insert(self.running_means, r_mean)
        table.insert(self.running_projections,r_projection)
      end
       
   end

  -- use extra scaling parameter----------------- 
   if self.affine then
      self.weight = torch.Tensor(nDim)
      self.bias = torch.Tensor(nDim)
      self.gradWeight = torch.Tensor(nDim)
      self.gradBias = torch.Tensor(nDim)
      self:reset()
   end
   
 -----some configures------------
   self.train=true
   self.debug=false
   self.testMode_isRunning=true   --if this value set true, then use running parameter, 
                                  --when do the training,  else false, use the previous parameters 
end


function DecorelateBN_Advance:reset()
  -- self.weight:uniform()
   self.weight:fill(1)
   self.bias:zero()
end

function DecorelateBN_Advance:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')

------------------------------------------train mode -------------------------------
 
  function updateOutput_perGroup_train(data,groupId)
     local nBatch = data:size(1)
     local scale=data.new() --eigValue^(-1/2)

     local mean= data.new()
     local centered = data.new()
     local output=data.new()
     centered:resizeAs(data)
      output:resizeAs(data)
      mean:mean(data, 1)                        -- E(x) = expectation of x.
      self.running_means[groupId]:mul(1 - self.momentum):add(self.momentum, mean) -- add to running mean
      self.buffer:repeatTensor(mean, nBatch, 1)

      -- subtract mean
      centered:add(data, -1, self.buffer)         -- x - E(x)

      ----------------------calcualte the projection matrix----------------------
      self.buffer_1:resize(data:size(2),data:size(2))
      self.buffer_1:addmm(0,self.buffer_1,1/nBatch,centered:t(),centered) --buffer_1 record correlation matrix
           -----------------------matrix decomposition------------- 
     local rotation,eig,_=torch.svd(self.buffer_1)   
      eig:add(self.epsilon)   
      scale:resizeAs(eig)     
      scale:copy(eig)
      scale:pow(-1/2) --scale=eig^(-1/2)
      
      self.buffer_2:resizeAs(rotation) 
      self.buffer_2:cmul(rotation,torch.repeatTensor(scale, (#scale)[1], 1)) --U=D* Eighta^(-1/2)
      
      --rotation to the original space
      self.buffer_1:mm(self.buffer_2,rotation:t())
     
      self.running_projections[groupId]:mul(1 - self.momentum):add(self.momentum, self.buffer_1) -- add to running projection
      output:mm(centered, self.buffer_1)
      
      ----------------record the results of each group--------------
      table.insert(self.eigs, eig)
      table.insert(self.scales, scale)
      table.insert(self.rotations, rotation)
      table.insert(self.centereds, centered)
     
      return output
 end
 
 ------------------------ test mode-----------------------------------------
  function updateOutput_perGroup_test(data,groupId)
     local nBatch = data:size(1)
      local output=data.new()
      output:resizeAs(data)
      self.buffer_1:resizeAs(data):copy(data)
      self.buffer:repeatTensor(self.running_means[groupId], nBatch, 1)
      self.buffer_1:add(-1, self.buffer)
      output:mm(self.buffer_1,self.running_projections[groupId]) 
      return output
  end

---------------------------------------------------------------------------------------
--------------------updateOutput main function-------------------------
----------------------------------------------------------------------------------------

   local nDim=input:size(2)


   assert(nDim  == self.nDim, 'make sure the dimensions of the input is same as the initionization')
   
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
   self.output=self.output or input.new()
   self.output:resizeAs(input)
   
   self.gradInput=self.gradInput or input.new()
   self.gradInput:resizeAs(input)
   
   self.normalized = self.normalized or input.new() --used for the affine transformation to calculate the gradient
   self.normalized:resizeAs(input)
   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer_1 = self.buffer_1 or input.new()
   self.buffer_2 = self.buffer_2 or input.new()

   if self.train == false then
     if self.debug then
       print('--------------------------DBN:test mode***update output***-------------------')
     end
      for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)      
        self.output[{{},{start_index,end_index}}]=updateOutput_perGroup_test(input[{{},{start_index,end_index}}],i)   
      end
   else -- training mode
     
     --------------training mode, initalize the group parameters---------------
      self.eigs={}
      self.scales={}
      self.rotations={}
      self.centereds={}
      if self.debug then
        print('--------------------------DBN:train mode***update output***-------------------')
      end
      for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,nDim)      
         self.output[{{},{start_index,end_index}}]=updateOutput_perGroup_train(input[{{},{start_index,end_index}}],i)   
       end
   end
   self.normalized:copy(self.output)
 
   
   ------------------------------------------------------------------------ 
  -----------------------scale the output-------------------------------- 
 ------------------------------------------------------------------------   
   
   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight, input:size(1), 1)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias, input:size(1), 1)
      self.output:add(self.buffer)
   end

   return self.output
end



function DecorelateBN_Advance:updateGradInput(input, gradOutput)

------------------calculate the K matrix---------------------
  
  function getK_new(eig)
    local revised=1e-45 --used for div 0, in case of that there are tow eigenValuse is the same (It's almost impossible based on our experiments.)
    local K=torch.Tensor(eig:size(1),eig:size(1)):fill(revised)    
    local b_1=torch.Tensor(eig:size(1),eig:size(1)):repeatTensor(eig, eig:size(1), 1)
    local b_2=torch.eye(eig:size(1)):add(b_1:t()):add(-1,b_1):add(K)
    K:fill(1):cdiv(b_2):add(-1, torch.eye(eig:size(1))*(1+revised))
    return K  
  end
 
  
-------update the gradInput per Group in train mode------------------------- 
   function updateGradInput_perGroup_train_new(gradOutput_perGroup,groupId)
     local  eig=self.eigs[groupId]
     local  scale=self.scales[groupId]
     local  rotation=self.rotations[groupId]
     local  centered=self.centereds[groupId]
     local nBatch = gradOutput_perGroup:size(1) 
     self.hat_x=self.hat_x or gradOutput_perGroup.new()
     self.S=self.S or gradOutput_perGroup.new()
     self.M=self.M or gradOutput_perGroup.new()
     self.U=self.U or gradOutput_perGroup.new()
     self.f=self.f or gradOutput_perGroup.new()
     self.FC=self.FC or gradOutput_perGroup.new()
     self.d_hat_x=self.d_hat_x or gradOutput_perGroup.new()
     
     self.hat_x:resizeAs(centered)
     self.d_hat_x:resizeAs(gradOutput_perGroup)
     self.U:resizeAs(rotation)
     self.M:resizeAs(rotation)
     self.S:resizeAs(rotation)
     self.FC:resizeAs(rotation)
     
     self.U:cmul(rotation, torch.repeatTensor(scale, (#scale)[1], 1))
     self.hat_x:mm(centered, self.U)
     self.d_hat_x:mm(gradOutput_perGroup,rotation)
     
     self.FC:addmm(0, self.FC, 1/nBatch, self.d_hat_x:t(), self.hat_x)
     self.f:mean(self.d_hat_x, 1)
     
     local sz = (#self.FC)[1]
     
     local temp_diag=torch.diag(self.FC) --get the diag element of d_EighTa
     self.M:diag(temp_diag) --matrix form
 --------------------------------calculate S-----------------------------    
     self.S:cmul(self.FC:t(), torch.repeatTensor(eig, sz, 1):t())

     self.buffer:resizeAs(eig):copy(eig):pow(1/2)   
     self.buffer_1:cmul(self.FC, torch.repeatTensor(self.buffer, sz, 1):t())
     self.buffer_2:cmul(self.buffer_1, torch.repeatTensor(self.buffer, sz, 1))   
     
     self.S:add(self.buffer_2)
     self.buffer=getK_new(eig)
     
     self.S:cmul(self.buffer:t())
     self.buffer:copy(self.S)
     self.S:add(self.buffer, self.buffer:t())  

     self.S:add(-1, self.M)  -- S-M
     self.buffer_1:resizeAs(self.d_hat_x)
     self.buffer_1:mm(self.hat_x, self.S) --x_hat * (S-M)
     
     self.buffer:repeatTensor(self.f, nBatch, 1)
     self.buffer_1:add(self.d_hat_x):add(-1, self.buffer)
    
    self.buffer:mm(self.buffer_1, self.U:t())   
    
    return self.buffer
  end
  
  
  
  
  
 -------update the gradInput per Group in test mode-------------------------
    
  function updateGradInput_perGroup_test(gradOutput_perGroup,groupId)
     local  running_projection=gradOutput_perGroup:new()
     if self.testMode_isRunning then
     --use the estimated projection matrix
      running_projection=self.running_projections[groupId] --use running projection
     else
       -- use the latest projection matrix, which seem works worse, and not stable

      self.buffer_1:diag(self.scales[groupId])
      running_projection:resizeAs(self.rotations[groupId])
      running_projection:mm(self.rotations[groupId],self.buffer_1)  --self.buffer_2 cache the projection matrix
    end 
     
     
     
  --   local nBatch = gradOutput_perGroup:size(1) 
     self.buffer:resizeAs(gradOutput_perGroup)
     self.buffer:mm(gradOutput_perGroup,running_projection:t())
     return self.buffer
  end
  
  
    
---------------------------------------------------------------------------------------
--------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------
  
  
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
  
   local nDim=input:size(2)
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
   
   
   if self.train == false then -- test mode: in which the whitening parameter is fixed (not the function of the input)
      for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)      
        self.gradInput[{{},{start_index,end_index}}]=updateGradInput_perGroup_test(gradOutput[{{},{start_index,end_index}}],i)   
      end
      
   else --train mode 
       for i=1,groups do 
         local start_index=(i-1)*self.m_perGroup+1
         local end_index=math.min(i*self.m_perGroup,nDim)
         self.gradInput[{{},{start_index,end_index}}]=updateGradInput_perGroup_train_new(gradOutput[{{},{start_index,end_index}}],i)   
       end

    end
   
   
   ------------------------------------------------------------------------ 
  -----------------------scale the  gradInput-------------------------------- 
 ------------------------------------------------------------------------   
   if self.affine then
      self.buffer:repeatTensor(self.weight, input:size(1), 1)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end

function DecorelateBN_Advance:setTrainMode(isTrain)
    if isTrain ~= nil then
      assert(type(isTrain) == 'boolean', 'isTrain has to be true/false')
      self.train = isTrain
    else
      self.train=true
     end
end


function DecorelateBN_Advance:accGradParameters(input, gradOutput, scale)
    if self.affine then
        scale =scale or 1.0
      self.buffer_2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer_2:cmul(gradOutput)
      self.buffer:sum(self.buffer_2, 1) 
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) 
      self.gradBias:add(scale, self.buffer)
   end
end
