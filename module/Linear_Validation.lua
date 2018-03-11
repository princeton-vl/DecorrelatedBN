--[[
--Linear module in which the methods to calculate the condition number of the FIM(input/output's correlation matrix) are added.

Author: Lei Huang
 mail: huanglei@nlsde.buaa.edu.cn

--]]
local Linear_Validation, parent = torch.class('nn.Linear_Validation', 'nn.Module')

function Linear_Validation:__init(inputSize, outputSize,validate_input, validate_FIM, validate_gradInput,validate_output,orth_flag)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   self.correlate=torch.Tensor(outputSize,outputSize) 
   self.train=true
  
   --FIM related condition number
   self.FIM=torch.Tensor()
   self.conditionNumber_FIM={} 
   self.conditionNumber_FIM_90PerCent={}

   self.eig_FIM={}
   self.updateFIM_flag=false
      
   --Output and gradInput realted condition number 
   self.conditionNumber_input={}
   self.conditionNumber_output={}
   self.conditionNumber_gradInput={}
   self.eig_input={}
   self.eig_output={}
   self.eig_gradInput={}
   self.epcilo=10^-100
   
   self.validate_input=validate_input
   self.validate_FIM=validate_FIM 
   self.validate_gradInput=validate_gradInput 
   self.validate_output=validate_output 
   
   self.counter=0
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

function Linear_Validation:reset(stdv)
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
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
   return self
end


function Linear_Validation:reset_orthogonal()
    local initScale = 1.1 -- math.sqrt(2)

    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))
    local n_min = math.min(self.weight:size(1), self.weight:size(2))
    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)
    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)
    self.bias:zero()
end

function Linear_Validation:updateOutput(input)
   --self.bias:fill(0)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end

  ---------calculate the condition number of the  correlation matrix of the input-----------------
   if  self.train and self.validate_input then
      self.buffer_input=self.buffer_input or input.new()
      self.buffer_input:resize(input:size(2),input:size(2))
      self.buffer_input:addmm(0,self.buffer_input, 1/input:size(1),input:t(), input)

       _,self.buffer,_=torch.svd(self.buffer_input) --SVD Decompositon for singular value
       
       table.insert(self.eig_input,self.buffer:clone())
       self.buffer:add(self.epcilo)
       local conditionNumber=torch.abs(torch.max(self.buffer)/torch.min(self.buffer))
       self.conditionNumber_input[#self.conditionNumber_input + 1]=conditionNumber
    end 
   
   
  ---------calculate the condition number of the  correlation matrix of the output-----------------
  
    if  self.train and self.validate_output then
      self.correlate:addmm(0,self.correlate, 1/self.output:size(1),self.output:t(), self.output)

       _,self.buffer,_=torch.svd(self.correlate) --SVD Decompositon for singular value
       table.insert(self.eig_output,self.buffer:clone())
       self.buffer:add(self.epcilo)
       local conditionNumber=torch.abs(torch.max(self.buffer)/torch.min(self.buffer))
       self.conditionNumber_output[#self.conditionNumber_output + 1]=conditionNumber
    end 
   
   return self.output
end

function Linear_Validation:updateGradInput(input, gradOutput)
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
      
 -----------------------------validate gradInput-------
 --condition nubmer and eigValue
 --------------------------------------
      
      if  self.train and self.validate_gradInput then
        self.buffer_input:addmm(0,self.buffer_input, 1/self.gradInput:size(1),self.gradInput:t(), self.gradInput)
       _,self.buffer,_=torch.svd(self.buffer_input) --SVD Decompositon for singular value
       table.insert(self.eig_gradInput, self.buffer:clone())
       self.buffer:add(self.epcilo)
       local conditionNumber=torch.abs(torch.max(self.buffer)/torch.min(self.buffer))
       self.conditionNumber_gradInput[#self.conditionNumber_gradInput + 1]=conditionNumber
    end 
      
      
     --------------------------------------------------
      --calculate the FIM----------
      --------------------------------------------------
    if  self.validate_FIM and self.updateFIM_flag then
      print('--------calculate condition number of FIM----------------')
      
      local batchNumber=input:size(1)
      self.buffer_FIM=self.buffer_FIM or input.new()
      self.buffer=self.buffer or input.new()
      
      local eleNumber=gradOutput:size(2)*input:size(2)
      self.FIM=torch.Tensor(eleNumber,eleNumber):zero()
      self.buffer_FIM:resize(gradOutput:size(2),input:size(2))
      for i=1,batchNumber do
        self.buffer_FIM:mm(gradOutput[{{i},{}}]:t(),input[{{i},{}}])                
        self.buffer=torch.reshape(self.buffer_FIM,eleNumber,1)        
        self.FIM:addmm(self.buffer,self.buffer:t())
      end
      self.FIM:mul(1/batchNumber)
      
      ---calculate condition number----------------------
       _,self.buffer_FIM,_=torch.svd(self.FIM) --SVD Decompositon for singular value
       
       table.insert(self.eig_FIM, self.buffer_FIM:clone())
       self.buffer_FIM:add(self.epcilo)
       local conditionNumber=torch.abs(torch.max(self.buffer_FIM)/torch.min(self.buffer_FIM))
       print('Linear module:FIM conditionNumber='..conditionNumber)
       self.conditionNumber_FIM[#self.conditionNumber_FIM + 1]=conditionNumber

       local index=torch.floor(self.buffer_FIM:size(1)*0.9)
       local conditionNumber_90PerCent=torch.abs(torch.max(self.buffer_FIM)/self.buffer_FIM[index])
       print('Linear module:FIM conditionNumber_90PerCent='..conditionNumber_90PerCent..'--at location:'..index)
       self.conditionNumber_FIM_90PerCent[#self.conditionNumber_FIM_90PerCent + 1]=conditionNumber_90PerCent
     end
      return self.gradInput
   end
end

function Linear_Validation:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
end


function Linear_Validation:update_FIM_flag(flag)
   self.updateFIM_flag=flag or false
end

-- we do not need to accumulate parameters when sharing
Linear_Validation.sharedAccUpdateGradParameters = Linear_Validation.accUpdateGradParameters


function Linear_Validation:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
