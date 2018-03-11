--[[

]]--
local Affine_module,parent = torch.class('nn.Affine_module', 'nn.Module')

function Affine_module:__init(nFeature,initial, withBias)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: dimensionality of input. ')
   assert(nFeature ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
    
    if withBias ~=nil then
      assert(type(withBias) == 'boolean', 'withBias has to be true/false')
      self.withBias = withBias
   else
      self.withBias = true
   end
   self.initial=initial or 1
   self.weight = torch.Tensor(nFeature)
   self.gradWeight = torch.Tensor(nFeature)
   if self.withBias then
       self.bias = torch.Tensor(nFeature)
       self.gradBias = torch.Tensor(nFeature)
    end
    self:reset()
end

function Affine_module:reset()
   --self.weight:uniform()
   self.weight:fill(self.initial) -- the initial scale
   if self.withBias then
     self.bias:zero()
   end
end

function Affine_module:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.output:resizeAs(input):copy(input)
      -- multiply with gamma and add beta
   self.buffer:repeatTensor(self.weight, nBatch, 1)
   self.output:cmul(self.buffer)
   if self.withBias then 
     self.buffer:repeatTensor(self.bias, nBatch, 1)
     self.output:add(self.buffer)
   end
   return self.output
end

function Affine_module:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
 --  assert(self.train == true, 'should be in training mode when self.train is true')
    local nBatch = input:size(1)
   
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.buffer:repeatTensor(self.weight, nBatch, 1)
    self.gradInput:cmul(self.buffer)
   return self.gradInput
end

function Affine_module:accGradParameters(input, gradOutput, scale)
      scale = scale or 1.0
      self.buffer2=self.buffer2 or input.new()
      self.buffer2:resizeAs(input):copy(input)
      self.buffer2:cmul(gradOutput)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer)
     if self.withBias then
      self.buffer:sum(gradOutput, 1) -- sum over mini-batch
      self.gradBias:add(scale, self.buffer)
     end
end

