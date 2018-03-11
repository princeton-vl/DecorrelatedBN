--[[
--  The Torch implemenation of the natural neural network: Natural Neural Networks, NIPS 2015
-- The method are wrapped as one module called 'NormLinear_new' 
--Author: Lei Huang
-- mail: huanglei@nlsde.buaa.edu.cn
--]]


local NormLinear_new, parent = torch.class('nn.NormLinear_new', 'nn.Module')

function NormLinear_new:__init(inputSize, outputSize,affine)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   self.proMatrix=torch.eye(inputSize)
   self.mean=torch.Tensor(inputSize)
   self.mean:fill(0)
   
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = false
   end  
     
  
   self.debug=false -- whether use debug mode..
   self.useSVD=true -- whether use SVD do get the eigenValue, generally speaking, SVD is more stable and efficient
   self.useCenteredEstimation=true -- whether use centered data to estimate the correlation matrix sigma, generally, not use centered is more stable
   
     self.FIM=torch.Tensor()
   self.conditionNumber={}
   self.epcilo=10^-100
   self.updateFIM_flag=false
   self.printInterval=50
   self.count=0
   self:reset()
end

function NormLinear_new:reset(stdv)
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

function NormLinear_new:updateOutput(input)

   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      
      --buffers that are reused
      self.addBuffer = self.addBuffer or input.new() -- used for plus the bias, have the size of input(1)
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      self.input=self.input or input.new() --used to store the input for calulate the Correlation matrix 
      
      self.buffer=self.buffer or input.new()
      
      self.W = self.W or input.new()
      self.W:resizeAs(self.weight)
      
      self.buffer_1=self.buffer_1 or input.new()
      self.buffer_1:resizeAs(input)
     self.input:resizeAs(input):copy(input)
--------------------------------running mode----------------------     
     ------------------y=(x-E(x))*U^T * V^T+d_i
      self.buffer:repeatTensor(self.mean,nframe,1) 
      self.buffer_1:add(input,-1,self.buffer) --subtract mean: x-E(x), sotre in buffer_1
      self.W:mm(self.weight,self.proMatrix) --V*U
      self.output:addmm(0, self.output, 1, self.buffer_1, self.W:t())--(x-E(x))*U^T * V^T
      self.output:addr(1, self.addBuffer, self.bias)
      
   return self.output
end

function NormLinear_new:updateGradInput(input, gradOutput)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      self.W:mm(self.weight,self.proMatrix)
       self.gradInput:addmm(0, 1, gradOutput, self.W)  
    end
    
      
    --------------------------------------------------
      --calculate the FIM----------
      --------------------------------------------------
    if self.updateFIM_flag then
      print('--------calculate condition number----------------')
      
      local batchNumber=input:size(1)
      self.buffer_FIM=self.buffer_FIM or input.new()
      self.buffer=self.buffer or input.new()
      self.normalizedInput=self.normalizedInput or input.new()
      self.buffer:repeatTensor(self.mean,input:size(1),1) --buffer has the same dimensions as input
      self.buffer_FIM:add(input,-1,self.buffer)  
      self.normalizedInput:resizeAs(self.buffer_FIM)   
      self.normalizedInput:mm(self.buffer_FIM,self.proMatrix:t()) 
      
      local eleNumber=gradOutput:size(2)*self.normalizedInput:size(2)
      self.FIM=torch.Tensor(eleNumber,eleNumber):zero()
      self.buffer_FIM:resize(gradOutput:size(2),self.normalizedInput:size(2))
      for i=1,batchNumber do

        self.buffer_FIM:mm(gradOutput[{{i},{}}]:t(),self.normalizedInput[{{i},{}}])                
        self.buffer=torch.reshape(self.buffer_FIM,eleNumber,1)        
        self.FIM:addmm(self.buffer,self.buffer:t())
      end
      self.FIM:mul(1/batchNumber)
      
      ---calculate condition number----------------------
       _,self.buffer_FIM,_=torch.svd(self.FIM) --SVD Decompositon for singular value
       self.buffer_FIM:add(self.epcilo)
       local conditionNumber=torch.abs(torch.max(self.buffer_FIM)/torch.min(self.buffer_FIM))
       print('Normlinear module: conditionNumber='..conditionNumber)
       self.conditionNumber[#self.conditionNumber + 1]=conditionNumber
       
     end
    return self.gradInput
 
end

function NormLinear_new:accGradParameters(input, gradOutput, scale)
     scale = scale or 1
 --   print('------------accGradParameters--------------')
     assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
     self.buffer:repeatTensor(self.mean,input:size(1),1) --buffer has the same dimensions as input
     self.buffer_1:add(input,-1,self.buffer) --subtract mean: x-E(x)
     
     self.buffer:mm(self.buffer_1,self.proMatrix:t())
     self.gradWeight:addmm(scale, gradOutput:t(), self.buffer)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
end



function NormLinear_new:updatePromatrix(epsilo)
     print('------------update Norm--------------')
    self.buffer_sigma=self.buffer_sigma or self.input.new()
    self.centered_input=self.centered_input or self.input.new()
    self.b=self.b or self.input.new()
    self.b:resizeAs( self.bias)
    self.b:zero()
        
    self.W:mm(self.weight, self.proMatrix)
    self.b:addmv(1,self.bias,-1,self.W,self.mean)
     
    local nBatch=self.input:size()[1]

    self.mean=torch.mean(self.input,1)[1]
   
     --------begin centering the self.input------
     self.buffer_1:repeatTensor(self.mean,nBatch,1)
     self.centered_input:add(self.input,-1,self.buffer_1) 
     --------------------end: centering the input--------  
     
     self.buffer_sigma:resize(self.input:size(2),self.input:size(2)) --buffer_sigma is used for store the sigma

     self.buffer_sigma:addmm(0,self.buffer_sigma,1/nBatch,self.centered_input:t(),self.centered_input)
     self.buffer_sigma:add(epsilo,torch.eye(self.buffer_sigma:size(1)))
     -----------------------matrix decomposition------------- 
     if self.useSVD then
       self.buffer_1,self.buffer,_=torch.svd(self.buffer_sigma) --reuse the buffer: 'buffer' record e, 'buffer_1' record V    
       self.buffer:pow(-1/2)
       self.buffer_sigma:diag(self.buffer)   --self.buffer_sigma cache the scale matrix
     else
       self.buffer,self.buffer_1=torch.eig(self.buffer_sigma,'V') --reuse the buffer: 'buffer' record e, 'buffer_1' record V    
       self.buffer=self.buffer:select(2,1)  -- the first colum is the real eign value
       self.buffer:pow(-1/2)
       self.buffer_sigma:diag(self.buffer)   --self.buffer_sigma cache the scale matrix    
     end
     
     self.proMatrix:mm(self.buffer_sigma,self.buffer_1:t())
     
     self.weight:mm(self.W,torch.inverse(self.proMatrix))

    ----------self.bias=self.b+torch.mv(self.W,self.mean)
      self.bias:addmv(1,self.b,1,self.W,self.mean)
  
    if self.debug then
       self.buffer:resizeAs(self.centered_input)
       self.buffer:mm(self.centered_input,self.proMatrix:t())--record the  normalized input
       self.buffer_1:resize(self.buffer:size(2),self.buffer:size(2))
       self.buffer_1:addmm(0,self.buffer_1,1/Ns,self.buffer:t(),self.buffer) ---the validate matrix
        local W_norm=torch.norm(self.W)
        print('debug_NormLinear_newModeul:--W_norm:'..W_norm)
     
        print("------debug_NormLinear_newModeul:--diagonal of validate matri------")
        for i=1,self.buffer_1:size(1) do
          print(i..': '..self.buffer_1[i][i])
        end
    end
end

-- we do not need to accumulate parameters when sharing
NormLinear_new.sharedAccUpdateGradParameters = NormLinear_new.accUpdateGradParameters

function NormLinear_new:update_FIM_flag(flag)
   self.updateFIM_flag=flag or false
end

function NormLinear_new:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
