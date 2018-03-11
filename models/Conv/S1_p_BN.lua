--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'
require 'cudnn'
--require 'cudbn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
--local SBatchNorm = nn.SpatialBatchNormalization
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
     if stride ~= 1 then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
               :add(nn.Identity())
    
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))

      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
     

      s:add(SBatchNorm(n))

      return nn.Sequential()
            :add(s)
         :add(ReLU(true))
   end


   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = opt.hidden_number
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(opt.num_feature,iChannels,3,3,1,1,1,1))
      model:add(SBatchNorm(iChannels))
      model:add(ReLU(true))
      model:add(layer(basicblock, iChannels, n))
      model:add(layer(basicblock, iChannels, n, 2))
      model:add(layer(basicblock, iChannels, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(iChannels):setNumInputDims(3))
      model:add(nn.Linear(iChannels, opt.num_classes))

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
      --      v.bias = nil
      --      v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name, value)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(value)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization',opt.BNScale)
   BNInit('nn.SpatialBatchNormalization',opt.BNScale)
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'cudnn_deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   --model:get(1).gradInput = nil

   return model
end

return createModel(opt)
