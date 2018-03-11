require 'nn'
require 'cunn'
local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
  
local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(backend.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(backend.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ELU(1))

  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = backend.SpatialMaxPooling

ConvBNReLU(3,64)
vgg:add(MaxPooling(2,2,2,2))

ConvBNReLU(64,128)
vgg:add(MaxPooling(2,2,2,2))

ConvBNReLU(128,256)
ConvBNReLU(256,256)
vgg:add(MaxPooling(2,2,2,2))

ConvBNReLU(256,512)
ConvBNReLU(512,512)
vgg:add(MaxPooling(2,2,2,2))

-- In the last block of convolutions the inputs are smaller than
-- the kernels and cudnn doesn't handle that, have to use cunn
backend = nn
ConvBNReLU(512,512)
ConvBNReLU(512,512)
vgg:add(MaxPooling(2,2,2,2))
vgg:add(nn.View(512))

classifier = nn.Sequential()
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ELU(1))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,10))
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
