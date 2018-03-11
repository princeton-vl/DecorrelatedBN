
local Spatial_Scaling,parent = torch.class('nn.Spatial_Scaling', 'nn.Module')

function Spatial_Scaling:__init(nFeature, initial, withBias, BIUni)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   if withBias ~=nil then
      assert(type(withBias) == 'boolean', 'withBias has to be true/false')
      self.withBias = withBias
   else
      self.withBias = true
   end

   if BIUni ~=nil then
     -- assert(type(withBias) == 'boolean', 'withBias has to be true/false')
      self.is_BIUni = true
      self.n_input=BIUni
   else
      self.is_BIUni = false
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

function Spatial_Scaling:reset()
   --self.weight:uniform()
   self.weight:fill(self.initial) -- the initial scale
   --print(self.initial)
   if self.withBias then
     if self.is_BIUni then
        local stdv = 1./math.sqrt(self.n_input)
        print('uniform--'..stdv)
       self.bias:uniform(-stdv, stdv)
     else
       self.bias:zero()
     end
   end
end

function Spatial_Scaling:updateOutput(input)
   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.output:resizeAs(input):copy(input)

      -- multiply with scale and add bias
   self.output:cmul(self.weight:view(1,nFeature,1,1):expandAs(self.output))


  if self.withBias then
  --  print(self.bias)
    self.output:add(self.bias:view(1,nFeature,1,1):expandAs(self.output))
  end
   return self.output
end

function Spatial_Scaling:updateGradInput(input, gradOutput)
   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')

   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
 
   
   self.gradInput:cmul(self.weight:view(1,nFeature,1,1):expandAs(self.gradInput))

   return self.gradInput
end

function Spatial_Scaling:accGradParameters(input, gradOutput, scale)

      scale = scale or 1.0
      local nBatch = input:size(1)
      local nFeature = input:size(2)
      local iH = input:size(3)
      local iW = input:size(4)
      self.buffer2=self.buffer2 or input.new()
      self.buffer2:resizeAs(input):copy(input)
      self.buffer2 = self.buffer2:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
      self.buffer:sum(self.buffer2, 1) 
      self.buffer2:sum(self.buffer, 3) 
      self.gradWeight:add(scale, self.buffer2)
      if self.withBias then
     --   print(self.bias)
        self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
        self.buffer2:sum(self.buffer, 3)
        self.gradBias:add(scale, self.buffer2) 
      end

end
