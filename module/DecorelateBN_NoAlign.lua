
--[[
The Basic Decorelated Batch normalization version, in which:
  (1) use PCA to whitening the activation, (which actually doesn't work)
  (2) include train mode and test mode. in training mode, we train the module
  (3)  only for 2D input in MLP architecture and running on CPU

  Author: Lei Huang
  mail: huanglei@nlsde.buaa.edu.cn
]]--
local DecorelateBN_NoAlign,parent = torch.class('nn.DecorelateBN_NoAlign', 'nn.Module')

function DecorelateBN_NoAlign:__init(nDim, m_perGroup, affine, eps, momentum)
   parent.__init(self)
     
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = false
   end

   self.eps = eps or 1e-5
  -- self.eps =  0
   if m_perGroup~=nil then
       self.m_perGroup = m_perGroup==0 and nDim or m_perGroup>nDim and nDim or m_perGroup 
   else
     self.m_perGroup =  nDim/2 
   end 
   print('m_perGroup:'.. self.m_perGroup)

  self.threshold=0
   
   self.nDim=nDim
   
   self.momentum = momentum or 0.1
   self.running_means={}
   self.running_projections={}
   
   self.testMode_isRunning=true  --if this value set true, then use running parameter, when do the training,  else false, use the previous parameters
   
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
   ------------allow nDim % m_perGropu !=0
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
   
   if self.affine then
      self.weight = torch.Tensor(nDim)
      self.bias = torch.Tensor(nDim)
      self.gradWeight = torch.Tensor(nDim)
      self.gradBias = torch.Tensor(nDim)
      self:reset()
   end
   --for debug
    self.debug=false
end

function DecorelateBN_NoAlign:reset()
   self.weight:uniform()
   self.bias:zero()
end


function DecorelateBN_NoAlign:updateOutput(input)
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
      eig:add(self.eps)
      scale:resizeAs(eig)     
      scale:copy(eig)
      scale:pow(-1/2) --scale=eig^(-1/2)
      self.buffer_1:diag(scale)   --self.buffer_1 cache the scale matrix  
      self.buffer_2:resizeAs(rotation) 
      self.buffer_2:mm(rotation,self.buffer_1) --U=D* Eighta^(-1/2)
      self.running_projections[groupId]:mul(1 - self.momentum):add(self.momentum, self.buffer_2) -- add to running projection
     
      output:mm(centered, self.buffer_2)

      ----------------record the results of per groupt--------------
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
   assert(nDim  == self.nDim, 'make sure the dimensions of the input is same as the initionazation')
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
    
   self.output=self.output or input.new()
   self.output:resizeAs(input)
   self.gradInput=self.gradInput or input.new()
   self.gradInput:resizeAs(input)
   
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)
   self.buffer = self.buffer or input.new()
   self.buffer_1 = self.buffer_1 or input.new()
   self.buffer_2 = self.buffer_2 or input.new()

   if self.train == false then
      if self.debug then
         print('--------------------------DBN:test mode-------------------')
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
        print('--------------------------DBN:train mode-------------------')
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

function DecorelateBN_NoAlign:updateGradInput(input, gradOutput)

------------------calculate the K matrix---------------------
  function getK(scale)
    local K=torch.Tensor(scale:size(1),scale:size(1)):fill(0)
    local revise=0    --1e-100
    for i=1,scale:size(1) do
      for j=1,scale:size(1) do
        if (i~=j) and torch.abs(scale[i]-scale[j])> self.threshold then
          K[i][j]=1/(scale[i]-scale[j]+revise)
        end
      
      end   
    end  
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
     
     self.hat_x:resizeAs(centered)
     self.U:resizeAs(rotation)
     self.M:resizeAs(rotation)
     self.S:resizeAs(rotation)
     self.FC:resizeAs(rotation)
     
     
     self.buffer:diag(scale)
     self.U:mm(rotation, self.buffer)
     self.hat_x:mm(centered, self.U)
     
     self.FC:addmm(0, self.FC, 1/nBatch, gradOutput_perGroup:t(), self.hat_x)
     self.f:mean(gradOutput_perGroup, 1)
     
     local temp_diag=torch.diag(self.FC) --get the diag element of d_EighTa
    self.M:diag(temp_diag) --matrix form
 --------------------------------calculate S-----------------------------    
     self.buffer:diag(eig)
     self.S:mm(self.buffer, self.FC:t())
     
     self.buffer=getK(eig)
     
     self.S:cmul(self.buffer:t())
     self.buffer:copy(self.S)
     self.S:add(self.buffer, self.buffer:t())
     

      
     self.S:add(-1, self.M)  -- S-M
     self.buffer_1:resizeAs(gradOutput_perGroup)
     self.buffer_1:mm(self.hat_x, self.S) --x_hat * (S-M)
     
     self.buffer:repeatTensor(self.f, nBatch, 1)
     self.buffer_1:add(gradOutput_perGroup):add(-1, self.buffer)
   
    
   -- self.S:resizeAs(self.buffer_1):mm(self.buffer_1, self.M:diag(scale)) 
    self.buffer:mm(self.buffer_1, self.U:t())   
    
    return self.buffer
  end
  
  
  
  
  
 -------update the gradInput per Group in test mode-------------------------
    
  function updateGradInput_perGroup_test(gradOutput_perGroup,groupId)
     local  running_projection=gradOutput_perGroup:new()
     if self.testMode_isRunning then
      running_projection=self.running_projections[groupId] --use running projection
     else
       --local  rotation=self.rotations[groupId] --try use the last projection????

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
   
   
   if self.affine then
      self.buffer:repeatTensor(self.weight, input:size(1), 1)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end



function DecorelateBN_NoAlign:accGradParameters(input, gradOutput, scale)
    if self.affine then
      scale = scale or 1.0
      self.buffer_2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer_2:cmul(gradOutput)
      self.buffer:sum(self.buffer_2, 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) -- sum over mini-batch
      self.gradBias:add(scale, self.buffer)
   end
end
