--[[
   This is the variation version for experiment of spatial_DBN, which:
   (1)using cpu to calculate the eigenValue, GPU do other things
]]--

require('cunn')
local Spatial_DBN_opt,parent = torch.class('nn.Spatial_DBN_opt', 'nn.Module')


function Spatial_DBN_opt:__init(nDim,m_perGroup, eps,momentum,  affine, eps_first)
    parent.__init(self)

    if affine ~= nil then
        assert(type(affine) == 'boolean', 'affine has to be true/false')
        self.affine = affine
    else
        self.affine = false
    end

    if eps_first ~= nil then
        assert(type(eps_first) == 'boolean', 'eps_first has to be true/false')
        self.eps_first =eps_first 
    else
        self.eps_first = true
    end

    self.eps = eps or 1e-7

   if m_perGroup~=nil then
      self.m_perGroup = m_perGroup==0 and nDim or m_perGroup>nDim and nDim or m_perGroup 
   else
     self.m_perGroup =  nDim/2 
   end 
   print('m_perGroup:'.. self.m_perGroup)

    self.train = true
    self.debug = false

    self.nDim = nDim
    self.momentum = momentum or 0.1
    self.running_means={}
    self.running_projections={}

    local groups=torch.floor((nDim-1)/self.m_perGroup)+1
    self.n_groups = groups
 --  print("n_groups:"..self.n_groups)
    for i=1, groups do
        local length = self.m_perGroup
        if i == groups then
            length = nDim - (groups - 1) * self.m_perGroup
        end
        local r_mean=torch.zeros(length)
        local r_projection=torch.eye(length)
        table.insert(self.running_means, r_mean)
        table.insert(self.running_projections, r_projection)
    end
    local length = self.m_perGroup
    self.eye_ngroup = torch.eye(length):cuda()
    self.initial_K = torch.CudaTensor(length, length):fill(0)

    length = nDim - (groups - 1) * self.m_perGroup
    self.eye_ngroup_last = torch.eye(length):cuda()
    self.initial_K_last = torch.CudaTensor(length, length):fill(0)
    if self.affine then
      print('-----------------using scaling-------------------')
        self.weight = torch.Tensor(nDim)
       self.bias = torch.Tensor(nDim)
       self.gradWeight = torch.Tensor(nDim)
       self.gradBias = torch.Tensor(nDim)
       self:reset()
    end

end

function Spatial_DBN_opt:reset()
   self.weight:fill(1)
   self.bias:zero()

end

function Spatial_DBN_opt:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
 
----------------------------- mode for train data---------------------
   
  function update_perGroup(data,groupId)
 
     local scale=data.new() --eigValue^(-1/2)

     local temp_1D= data.new()
     local centered = data.new()
     
     local nBatch=data:size(1)
     local nFeature=data:size(2)
      centered:resizeAs(data)
     
      temp_1D:mean(data, 1)  -- E(x) = expectation of x.
      self.running_means[groupId]:mul(1 - self.momentum):add(self.momentum, temp_1D)
      --self.buffer:repeatTensor(mean, nBatch, 1)

      -- subtract mean
      --centered:add(data, -1, self.buffer)         -- x - E(x)
      centered:copy(data):add(-temp_1D:expand(nBatch,nFeature))

      ----------------------calcualte the projection matrix----------------------
      self.buffer:resize(nFeature,nFeature)
      self.buffer:addmm(0,self.buffer,1/nBatch,centered:t(),centered)  --buffer_1 record correlation matrix
           -----------------------matrix decomposition------------- 
      if self.eps_first then
        if groupId ~= self.n_groups then
            --print(self.buffer:size())
            self.buffer:add(self.eye_ngroup * self.eps)
        else
          self.buffer:add(self.eye_ngroup_last * self.eps)
        end
      end
      local rotation, eig, _ = torch.svd(self.buffer)
      -- local eig,rotation=torch.eig(float_mem) --reuse the buffer: 'buffer' record e, 'buffer_2' record V    
      -- local eig, rotation = torch.symeig(float_mem, 'V', 'L')
      -- eig = eig:select(2, 1)

      if not self.eps_first then
        eig:add(self.eps) 
      end
      
      if self.debug then
         local eig_mean=eig:mean()
         local eig_median=eig[eig:size(1)/2]
         local eig_max=eig[1]
         local eig_min=eig[-1]
         print(string.format("eigValue--mean= %6.6f, median = %6.6f, max= %6.6f, min = %6.6f", eig_mean, eig_median,eig_max,eig_min))
      end
      scale:resizeAs(eig)     
      scale:copy(eig)
      scale:pow(-1/2) --scale=eig^(-1/2)
      self.buffer_2:resizeAs(rotation) 
      --self.buffer_2:cmul(rotation, torch.repeatTensor(scale, (#scale)[1], 1))
      self.buffer_2:cmul(rotation, scale:view(1,nFeature):expandAs(rotation))

      --rotation to the original space
      self.buffer:mm(self.buffer_2,rotation:t())
      self.running_projections[groupId]:mul(1 - self.momentum):add(self.momentum, self.buffer) -- add to running projection
      
      self.buffer_1:resizeAs(centered)
      self.buffer_1:mm(centered, self.buffer)
  
      ----------------record the results of per groupt--------------
      table.insert(self.eigs, eig)
      table.insert(self.scales, scale)
      table.insert(self.rotations, rotation)
      table.insert(self.centereds, centered)

      return self.buffer_1
 end
 
 
  ------------------------ mode for test data---------------------
 
  function test_perGroup(data,groupId)
     local nBatch = data:size(1)
     local nFeature=data:size(2)
   
    -- print('-----------test-dbn---')
      self.buffer_1:resizeAs(data):copy(data)
    -- self.buffer:repeatTensor(self.running_means[groupId], nBatch, 1)
    self.buffer_1:add(-1,self.running_means[groupId]:view(1,nFeature):expandAs(data))

      self.buffer:resizeAs(self.buffer_1)
      self.buffer:mm(self.buffer_1,self.running_projections[groupId])
      return self.buffer
  end
 
---------------------------------------------------------------------------------------
-------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------
   local nBatch = input:size(1)
   local nDim=input:size(2)
   local iH=input:size(3)
   local iW=input:size(4)
   self.eigs={}
   self.scales={}
   self.rotations={}
   self.centereds={}
   -- assert(nDim % self.m_perGroup == 0, 'make sure the all dimsions mod the dimsion of per group is zero')
   
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
    
   self.output=self.output or input.new()
   self.output:resize(#input)
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)

   self.gradInput=self.gradInput or input.new()
   self.gradInput:resize(#input)
  
   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer_1 = self.buffer_1 or input.new()   -- just for NxD data
   self.buffer_2 = self.buffer_2 or input.new()   -- just for DXD data
   self.input_temp= self.input_temp or input.new() 
   self.output_temp= self.output_temp or input.new() 
   self.input_temp=input:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):t()  --transfoer to 2D data
   --local output_temp=input.new() 
   self.output_temp:resizeAs(self.input_temp)
   
   if self.train == false then
      for i=1,groups do 
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)      
        self.output_temp[{{},{start_index,end_index}}]=test_perGroup(self.input_temp[{{},{start_index,end_index}}],i)   
      end
   else -- training mode
   -- print('-------------------------training mode-----------------') 
    for i=1,groups do 
     
        local start_index=(i-1)*self.m_perGroup+1
        local end_index=math.min(i*self.m_perGroup,nDim)    
        self.output_temp[{{},{start_index,end_index}}]=update_perGroup(self.input_temp[{{},{start_index,end_index}}],i)   
     end
   end
     self.output:copy(self.output_temp:t():reshape(nDim, nBatch,iH*iW):transpose(1,2):reshape(nBatch,nDim,iH,iW)) 
     self.normalized:copy(self.output)


     if self.affine then
        self.buffer:repeatTensor(self.weight:view(1, nDim, 1, 1),
                nBatch, 1, iH, iW)
             self.output:cmul(self.buffer)
            self.buffer:repeatTensor(self.bias:view(1, nDim, 1, 1),
                       nBatch, 1, iH, iW)
             self.output:add(self.buffer)
    end

   return self.output
end

function Spatial_DBN_opt:updateGradInput(input, gradOutput)


------------------calculate the K matrix---------------------

    function getK(eig, is_last_group)
        local K
        if not is_last_group then
            K=self.initial_K:clone()
            local b_1 = torch.repeatTensor(eig, eig:size(1), 1)
            local b_2 = self.eye_ngroup:clone():add(b_1:t()):add(-1, b_1):add(K)
            K:fill(1):cdiv(b_2):add(-1, self.eye_ngroup*(1))
        else
            K=self.initial_K_last:clone()
            local b_1 = torch.repeatTensor(eig, eig:size(1), 1)
            local b_2 = self.eye_ngroup_last:clone():add(b_1:t()):add(-1, b_1):add(K)
            K:fill(1):cdiv(b_2):add(-1, self.eye_ngroup_last*(1))
        end
        return K 
    end
  
  
  
  function updateGradInput_perGroup(gradOutput_perGroup,groupId)
     local  eig=self.eigs[groupId]
     local  scale=self.scales[groupId]
     local  temp_DD=self.rotations[groupId]  -- to save memory, by using temp_DD for buffering the data with size DXD
     local  temp_ND=self.centereds[groupId]  -- to save memory, by using temp_ND for buffering the data with size NXD
     
     local nBatch = gradOutput_perGroup:size(1) 
     local nFeature=gradOutput_perGroup:size(2)
     
     --self.hat_x=self.hat_x or gradOutput_perGroup.new()
     --self.K=self.K or gradOutput_perGroup.new()
    -- self.M=self.M or gradOutput_perGroup.new()
     self.U=self.U or gradOutput_perGroup.new()
     self.f=self.f or gradOutput_perGroup.new()
    -- self.FC=self.FC or gradOutput_perGroup.new()
     --self.d_hat_x=self.d_hat_x or gradOutput_perGroup.new()
     

     self.U:resizeAs(temp_DD)
     --self.K:resizeAs(temp_DD)
     self.buffer_2:resizeAs(temp_DD)
    -- self.FC:resizeAs(temp_DD)
     self.buffer_1:resizeAs(temp_ND)


    -- self.hat_x:resizeAs(temp_ND)
    -- self.d_hat_x:resizeAs(temp_ND)


     --self.U:cmul(temp_DD, torch.repeatTensor(scale, (#scale)[1], 1))
     self.U:cmul(temp_DD, scale:view(1,nFeature):expandAs(temp_DD))
     
     self.buffer_1:mm(temp_ND, self.U)  --buffer_1---->hat_x


     -- self.d_hat_x = torch.mm(self.buffer_2,rotation)
    -- self.d_hat_x = torch.mm(rotation:t(), gradOutput_perGroup:t()):t()
     temp_ND = torch.mm(temp_DD:t(), gradOutput_perGroup:t()):t() -- temp_ND ---->d_hat_x
     temp_DD:addmm(0, temp_DD, 1/(nBatch), temp_ND:t(), self.buffer_1) -- temp_DD ------>FC
     self.f:mean(temp_ND, 1)

     local sz = (#temp_DD)[1]
 --------------------------------calculate S-----------------------------    
     --Eighta* FC^T
    -- self.buffer_2:cmul(temp_DD:t(), torch.repeatTensor(eig, sz, 1):t())  ---self.buffer_2 ---------->S
     self.buffer_2:cmul(temp_DD:t(), eig:view(nFeature,1):expandAs(temp_DD) )  ---self.buffer_2 ---------->S

     --Eighta(1/2) * F_C * Eighta(1/2)
     local eig12 = eig:clone():pow(1/2)
     self.buffer:resizeAs(self.buffer_2)
     --self.buffer:cmul(temp_DD, torch.repeatTensor(eig12, sz, 1):t())
   -- self.buffer:cmul( torch.repeatTensor(eig12, sz, 1))
    self.buffer:cmul(temp_DD, eig12:view(nFeature,1):expandAs(temp_DD))
    self.buffer:cmul(eig12:view(1,nFeature):expandAs(temp_DD) )

     self.buffer_2:add(self.buffer)    --Eighta*FC^T+Eighta(1/2)*FC*Eighta(1/2)
     self.buffer=getK(eig, groupId == self.n_groups)  ---K, note that getK use the buffer_2, so if you want to use buffer_2 to store some data, be careful!!!!
     
     self.buffer_2:cmul(self.buffer:t())
     self.buffer:copy(self.buffer_2)
     self.buffer_2:add(self.buffer, self.buffer:t())  ----  buffer_2-----------keep record S 

     ---------------calculate M--------------------------------- 
     local temp_mask
     if groupId == self.n_groups then
        temp_mask = self.eye_ngroup_last
     else
        temp_mask = self.eye_ngroup
     end
     self.buffer = temp_DD:clone():maskedFill(torch.eq(temp_mask, 0), 0)   --buffer -----------> M
     self.buffer_2:add(-1, self.buffer)  -- S-M
     self.buffer:resizeAs(temp_ND)
     self.buffer:mm(self.buffer_1, self.buffer_2) --x_hat * (S-M)
    -- temp_ND:add(self.buffer):add(-1, torch.repeatTensor(self.f, (nBatch), 1))
    temp_ND:add(self.buffer):add(-1,self.f:expandAs(temp_ND))

     --self.buffer:add(self.d_hat_x):add(-1, torch.repeatTensor(self.f, (nBatch), 1))

      -------------get the result as the form required
    --self.buffer_2:resizeAs(self.hat_x):copy(torch.mm(self.buffer_1, self.U:t()))

    return torch.mm(temp_ND,self.U:t())
  end
  

-------update the gradInput per Group in test mode-------------------------
  function updateGradInput_perGroup_test(gradOutput_perGroup,groupId)
       local running_projection=self.running_projections[groupId] --use running projection
        self.buffer_1:resizeAs(gradOutput_perGroup)
        self.buffer_1:mm(gradOutput_perGroup,running_projection:t())
       return self.buffer_1
     end
---------------------------------------------------------------------------------------
-------------------updateGradInput main function-------------------------
----------------------------------------------------------------------------------------



   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   local nBatch=input:size(1)
   local nDim=input:size(2)
   local iH=input:size(3)
   local iW=input:size(4)
   local groups=torch.floor((nDim-1)/self.m_perGroup)+1
   
   self.input_temp=gradOutput:view(nBatch,nDim,iH*iW):transpose(1,2):reshape(nDim,nBatch*iH*iW):t()  --transfoer to 2D data
   --local output_temp=input.new() 
   self.output_temp:resizeAs(self.input_temp)
  
    if self.train==false then 
        for i=1,groups do 
            local start_index=(i-1)*self.m_perGroup+1
           local end_index=math.min(i*self.m_perGroup,nDim)   
            self.output_temp[{{},{start_index,end_index}}]=updateGradInput_perGroup_test(self.input_temp[{{},{start_index,end_index}}],i)   
         end

    else
        for i=1,groups do 
            local start_index=(i-1)*self.m_perGroup+1
           local end_index=math.min(i*self.m_perGroup,nDim)   
           self.output_temp[{{},{start_index,end_index}}]=updateGradInput_perGroup(self.input_temp[{{},{start_index,end_index}}],i)   
         end
    end 
     self.gradInput:copy(self.output_temp:t():reshape(nDim, nBatch,iH*iW):transpose(1,2):reshape(nBatch,nDim,iH,iW)) 
     
     if self.affine then
     self.buffer:repeatTensor(self.weight:view(1, nDim, 1, 1),
                   nBatch, 1, iH, iW)
       self.gradInput:cmul(self.buffer)
     end


   return self.gradInput
end

function Spatial_DBN_opt:setTrainMode(isTrain)
  if isTrain ~= nil then
      assert(type(isTrain) == 'boolean', 'isTrain has to be true/false')
      self.train = isTrain
  else
    self.train=true  

  end
end

function Spatial_DBN_opt:clearState()
         self.buffer:set()
         self.buffer_1:set()
         self.buffer_2:set()
      --   self.input_temp:set()
      --   self.output_temp:set()
         self.gradInput:set()
        -- self.K:set()
         self.U:set()
         self.f:set()
      --  self.eye_ngroup:set()
      --  self.eye_ngroup_last:set()
      --  self.initial_K:set()
      --  self.initial_K_last:set()
        self.output:set()
         self.eigs=nil
         self.scales=nil
         self.rotations=nil
         self.centereds=nil

  return 
end

function Spatial_DBN_opt:accGradParameters(input, gradOutput, scale)

  if self.affine then
    scale = scale or 1.0
    local nBatch = input:size(1)
    local nFeature = input:size(2)
    local iH = input:size(3)
    local iW = input:size(4)
    self.buffer_2:resizeAs(self.normalized):copy(self.normalized)
    self.buffer_2 = self.buffer_2:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
    self.buffer:sum(self.buffer_2, 1) -- sum over mini-batch
    self.buffer_2:sum(self.buffer, 3) -- sum over pixels
    self.gradWeight:add(scale, self.buffer_2)
    self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
    self.buffer_2:sum(self.buffer, 3)
    self.gradBias:add(scale, self.buffer_2) -- sum over mini-batch
  end

end
