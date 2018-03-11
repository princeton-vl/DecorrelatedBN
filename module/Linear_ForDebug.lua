--[[
--re-implimation of the linear module for debug. You can use this module to debug your code, e.g. observe the output or the inteval varibles.
--
--]]

local Linear_ForDebug, parent = torch.class('nn.Linear_ForDebug', 'nn.Module')

function Linear_ForDebug:__init(inputSize, outputSize,orth_flag,isBias)
   parent.__init(self)
   if isBias ~= nil then
      assert(type(isBias) == 'boolean', 'isBias has to be true/false')
      self.isBias = isBias
   else
      self.isBias = true
   end

   self.weight = torch.Tensor(outputSize, inputSize)
  if self.isBias then
   self.bias = torch.Tensor(outputSize)
   self.gradBias = torch.Tensor(outputSize)
  end
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   
   self.FIM=torch.Tensor()
   self.conditionNumber={}
   self.epcilo=10^-100
   self.counter=0
   self.FIM_updateInterval=100
   
   --for debug
   
   self.printDetail=false
   self.debug=false
   self.debug_detailInfo=false
   self.printInterval=1
   self.count=0
   
  if orth_flag ~= nil then
      assert(type(orth_flag) == 'boolean', 'orth_flag has to be true/false')
    if orth_flag then
      self:reset_orthogonal()
    else
    self:reset()
    end
  else
    self:reset()
  end
end

function Linear_ForDebug:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
       
        if self.isBias then 
         self.bias[i] = torch.uniform(-stdv, stdv)
        end
     end
   else
      self.weight:uniform(-stdv, stdv)
     if self.isBias then
       self.bias:uniform(-stdv, stdv)
     end
    end
   return self
end


function Linear_ForDebug:reset_orthogonal()
    local initScale = 1.1 -- math.sqrt(2)
   -- local initScale =  math.sqrt(2)
    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))
    local n_min = math.min(self.weight:size(1), self.weight:size(2))
    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)
    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)
    self.bias:zero()
end

function Linear_ForDebug:updateOutput(input)
   --self.bias:fill(0)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.isBias then
        self.output:addr(1, self.addBuffer, self.bias)
      end
   else
      error('input must be vector or matrix')
   end


   if self.printDetail then
     print("Linear_ForDebug: activation, number fo example=20")
     print(self.output[{{1,20},{}}])  
   end
   if self.debug and (self.count % self.printInterval==0)then
     -- local input_norm=torch.norm(input,1)/input:numel()
      local input_norm=torch.norm(input,1)
      local output_norm=torch.norm(self.output,1)
    --  print('debug_LinearModule--input_norm_elementWise:'..input_norm..' --output_norm_elementWise:'..output_norm)
      
   end

   if self.debug_detailInfo and (self.count % self.printInterval==0)then
      local input_mean=input:mean(1)
      local input_normPerDim=torch.norm(input,1,1)/input:size(1)
      local output_mean=self.output:mean(1)
      local output_normPerDim=torch.norm(self.output,1,1)/self.output:size(1)
      print('debug_LinearModule--input_mean:') 
      print(input_mean)
      print('debug_LinearModule--input_normPerDim:') 
      print(input_normPerDim)    
      print('debug_LinearModule--output_mean:') 
      print(output_mean)    
      print('debug_LinearModule--output_normPerDim:') 
      print(output_normPerDim)        
   end
   
   return self.output
end

function Linear_ForDebug:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end
   if self.printDetail then
     print("Linear_ForDebug: gradOutput, number fo example=20")
     print(gradOutput[{{1,20},{}}]) 
    end 
    if self.debug and (self.count % self.printInterval==0)then
      local gradOutput_norm=torch.norm(gradOutput,1)
      local gradInput_norm=torch.norm(self.gradInput,1)
    end
      
    if self.debug_detailInfo and (self.count % self.printInterval==0)then
      local gradInput_mean=self.gradInput:mean(1)
      local gradInput_normPerDim=torch.norm(self.gradInput,1,1)/self.gradInput:size(1)
      
      local gradOutput_mean=gradOutput:mean(1)
      local gradOutput_normPerDim=torch.norm(gradOutput,1,1)/gradOutput:size(1)
      print('debug_LinearModule--gradInput_mean:') 
      print(gradInput_mean)
      print('debug_LinearModule--gradInput_normPerDim:') 
      print(gradInput_normPerDim)    
      print('debug_LinearModule--gradOutput_mean:') 
      print(gradOutput_mean)    
      print('debug_LinearModule--gradOutput_normPerDim:') 
      print(gradOutput_normPerDim)        
   end
      
      
      return self.gradInput
   end
end

function Linear_ForDebug:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
     
      if self.isBias then
        self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
   
    if self.debug and (self.count % self.printInterval==0) then
      local weight_norm=torch.norm(self.weight,1)
      local bias_norm=torch.norm(self.bias,1)
      local gradWeight_norm=torch.norm(self.gradWeight,1)
      local gradBias_norm=torch.norm(self.gradBias,1)
    
       print('debug_LinearModule:--weight_norm_elementWise:'..weight_norm..' --bias_norm_elementWise:'..bias_norm)
       print(' --gradWeight_norm_elementWise:'..gradWeight_norm..' --gradBias_norm_elementWise:'..gradBias_norm)
      
   end
   
   if self.debug_detailInfo and (self.count % self.printInterval==0)then

      local gradWeight_mean_rowWise=self.gradWeight:mean(1)
      local gradWeight_mean_columnWise=self.gradWeight:mean(2)
      local gradWeight_normPerDim_rowWise=torch.norm(self.gradWeight,1,1)/self.gradWeight:size(1)
      local gradWeight_normPerDim_columnWise=torch.norm(self.gradWeight,1,2)/self.gradWeight:size(2)
      
      local weight_mean_rowWise=self.weight:mean(1)
      local weight_mean_columnWise=self.weight:mean(2)
      local weight_normPerDim_rowWise=torch.norm(self.weight,1,1)/self.weight:size(1)
      local weight_normPerDim_columnWise=torch.norm(self.weight,1,2)/self.weight:size(2)
     
      print('debug_LinearModule--gradWeight_mean_rowWise:') 
      print(gradWeight_mean_rowWise)
      print('debug_LinearModule--gradWeight_mean_columnWise:') 
      print(gradWeight_mean_columnWise)    
      print('debug_LinearModule--gradWeight_normPerDim_rowWise:') 
      print(gradWeight_normPerDim_rowWise)    
      print('debug_LinearModule--gradWeight_normPerDim_columnWise:') 
      print(gradWeight_normPerDim_columnWise)       
      
      
      print('debug_LinearModule--weight_mean_rowWise:') 
      print(weight_mean_rowWise)
      print('debug_LinearModule--weight_mean_columnWise:') 
      print(weight_mean_columnWise)    
      print('debug_LinearModule--weight_normPerDim_rowWise:') 
      print(weight_normPerDim_rowWise)    
      print('debug_LinearModule--weight_normPerDim_columnWise:') 
      print(weight_normPerDim_columnWise)     
   end
   
   self.count=self.count+1 --the ending of all the operation in this module
end

-- we do not need to accumulate parameters when sharing
Linear_ForDebug.sharedAccUpdateGradParameters = Linear_ForDebug.accUpdateGradParameters


function Linear_ForDebug:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
