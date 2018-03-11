--[[
  The Torch implementation of  Layer Normalization: NIPS 2016
    
  Author: Lei Huang
 mail: huanglei@nlsde.buaa.edu.cn

  ]]--
local LayerNorm,parent = torch.class('nn.LayerNorm', 'nn.Module')

function LayerNorm:__init(nOutput, affine, eps, momentum)
   parent.__init(self)
   
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = false
   end
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   if self.affine then
      self.weight = torch.Tensor(nOutput)
      self.bias = torch.Tensor(nOutput)
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
      self:reset()
   end
   
end

function LayerNorm:reset()
  self.weight:uniform()
   self.bias:zero()
end

function LayerNorm:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nDim=input:size(2)
   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer2 = self.buffer2 or input.new()
   self.centered = self.centered or input.new()
   self.centered:resizeAs(input)
   self.std = self.std or input.new()
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)
   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)
      
   self.buffer:mean(input, 2)                        -- u=\Sigma_{i=1}^nDim x_i.
   self.buffer2:repeatTensor(self.buffer, 1, nDim)
      -- subtract mean
   self.centered:add(input, -1, self.buffer2)         -- x - u

      -- calculate standard deviation over dimension
   self.buffer:resizeAs(self.centered):copy(self.centered):cmul(self.buffer) -- [x - E(x)]^2

      -- 1 / E([x - E(x)]^2)
   self.std:mean(self.buffer, 2):add(self.eps):sqrt():pow(-1)

   self.buffer:repeatTensor(self.std, 1,nDim)
      
      -- divide standard-deviation + eps
   self.output:cmul(self.centered, self.buffer)
   self.normalized:copy(self.output)

   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:resizeAs(self.output):repeatTensor(self.weight, nBatch, 1)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias, nBatch, 1)
      self.output:add(self.buffer)
   end
 
   return self.output
end

function LayerNorm:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
 --  assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)
   local nDim=input:size(2)

   self.gradInput:cmul(self.normalized, gradOutput)
   self.buffer:mean(self.gradInput, 2)
   self.gradInput:repeatTensor(self.buffer, 1,nDim)
   self.gradInput:cmul(self.normalized):mul(-1)

   self.buffer:mean(gradOutput, 2)
   self.buffer2:repeatTensor(self.buffer, 1, nDim)
   self.gradInput:add(gradOutput):add(-1, self.buffer2)
   
   self.buffer:repeatTensor(self.std, 1, nDim)
   self.gradInput:cmul(self.buffer)
   
   if self.affine then
      self.buffer:resizeAs(self.gradInput):repeatTensor(self.weight, nBatch, 1)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end

function LayerNorm:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      self.buffer2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer2:cmul(gradOutput)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) -- sum over mini-batch
      self.gradBias:add(scale, self.buffer)
   end
end

