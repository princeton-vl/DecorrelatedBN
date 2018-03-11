 --[[This script is based on the wide residual network project:https://github.com/szagoruyko/wide-residual-networks--]]

require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
dofile './provider.lua'

local c = require 'trepl.colorize'
local json = require 'cjson'
paths.dofile'augmentation.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
--local iterm = require 'iterm'
--require 'iterm.dot'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('compare the Decorelated BatchNormalizaiton method with baselines on wide-resnet architechture')
cmd:text()
cmd:text('Options')

cmd:option('-dataset','./dataset/cifar_provider.t7','')
cmd:option('-model','S1_p_BN','')
cmd:option('-max_epoch',160,'maximum number of iterations')
cmd:option('-epoch_step',"{80,120}",'epoch step: no lr annealing if it is larger than the maximum')
cmd:option('-save',"log_exp_4Splain_deep" ,'subdirectory to save logs')
cmd:option('-batchSize',128,'the number of examples per batch')

cmd:option('-optimMethod','sgd','the methods: options:sgd,rms,adagrad,adam')
cmd:option('-learningRate',0.1,'initial learning rate')
cmd:option('-learningRateDecay',0,'')
cmd:option('-learningRateDecayRatio',0.2,'')
cmd:option('-weightDecay',0.0005,'weight Decay for regularization')
cmd:option('-dampening',0,'')
cmd:option('-momentum',0.9,'momentum')
cmd:option('-m_perGroup',16,'the number of per group')
cmd:option('-eps',1e-5,'the revisation for DBN')

cmd:option('-BNScale',1,'the initial value for BN scale')
cmd:option('-scaleIdentity',0,'1 indicates scaling the Identity shortcut;0 indicates not')
cmd:option('-noNesterov',0,'1 indicates dont use nesterov momentum;0 indicates not')

cmd:option('-widen_factor',1,'')
cmd:option('-depth',32,'')
cmd:option('-hidden_number',48,'')

cmd:option('-optimMethod','sgd','')
cmd:option('-init_value',10,'')
cmd:option('-shortcutType','A','')
cmd:option('-nesterov',false,'')
cmd:option('-dropout',0,'')
cmd:option('-hflip',true,'')
cmd:option('-randomcrop',4,'')
cmd:option('-imageSize',32,'')
cmd:option('-randomcrop_type','reflection','')
cmd:option('-cudnn_fastest',true,'')
cmd:option('-cudnn_deterministic',false,'')
cmd:option('-optnet_optimize',true,'')
cmd:option('-generate_graph',false,'')
cmd:option('-multiply_input_factor',1,'')

cmd:option('-seed',1,'the step to debug the weights')
cmd:option('-T',389,'the step to debug the weights')

opt = cmd:parse(arg)

opt.rundir = cmd:string('console/exp_vggE_Cifar100/Info', opt, {dir=true})
paths.mkdir(opt.rundir)

cmd:log(opt.rundir .. '/log', opt)

cutorch.manualSeed(opt.seed)

--opt = xlua.envparams(opt)
if opt.noNesterov==1 then opt.nesterov=false end

opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset)
opt.num_classes = provider.testData.labels:max()
opt.num_feature = provider.testData.data:size(2)

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = dofile('models/Conv/'..opt.model..'.lua'):cuda()
  -- print('-------------------')
do
   local function add(flag, module) if flag then model:add(module) end end
   add(opt.hflip, nn.BatchFlip():float())
   add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())
   model:add(net)

   cudnn.convert(net, cudnn)
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
   if opt.cudnn_deterministic then
      model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end
   print(net)
   print('Network has', #model:findModules'cudnn.SpatialConvolution', 'convolutions')

   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.generate_graph then
      iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
   end
   if opt.optnet_optimize then
      --optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

log_name='vggE_Cifar100_'..opt.model..'_depth'..opt.depth..'_h'..opt.hidden_number..'_lr'..opt.learningRate..'_G'..opt.m_perGroup..'_b'..opt.batchSize..'_BI'..opt.BNScale..'_s'..opt.scaleIdentity..'_eps'..opt.eps..'_wD'..opt.weightDecay..'_mm'..opt.momentum..'_nN'..opt.noNesterov..'_lD'..opt.learningRateDecay
opt.save=opt.save..'/'..log_name
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

local parameters,gradParameters = model:getParameters()

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()

-- a-la autograd
local f = function(inputs, targets)
   model:forward(inputs)
   local loss = criterion:forward(model.output, targets)
   local df_do = criterion:backward(model.output, targets)
   model:backward(inputs, df_do)
   return loss
end

print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)

function train()
  model:training()

  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all minibatches have equal size
  indices[#indices] = nil

  local loss = 0

  for t,v in ipairs(indices) do
    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    optim[opt.optimMethod](function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      local loss_Iter=f(inputs, targets)
      
      confusion:batchAdd(model.output, targets)
       print(string.format("Iter: %6s,  loss = %6.6f", iteration,loss_Iter))            

      losses[#losses+1]=loss_Iter
      loss = loss + loss_Iter
       iteration=iteration+1

      return f,gradParameters
    end, parameters, optimState)
  end
 confusion:updateValids()
  train_acc = confusion.totalValid * 100
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t '):format(train_acc))
  train_accus[#train_accus+1]=train_acc
  --confusion:zero()
  return loss / #indices
end


function test()
  model:evaluate()
  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local data_split = provider.testData.data:split(opt.batchSize,1)
  local labels_split = provider.testData.labels:split(opt.batchSize,1)

  for i,v in ipairs(data_split) do
    confusion:batchAdd(model:forward(v), labels_split[i])
  end

  confusion:updateValids()

 if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
       os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
       if f then base64im = f:read'*all' end
     end
      local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  return confusion.totalValid * 100
end

iteration=0
losses={}
losses_epoch={}
train_accus={}
test_accus={}

train_times={}
test_times={}
scale_epoch={}
for epoch=1,opt.max_epoch do
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  -- drop learning rate and reset momentum vector
  if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
     torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
    opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
    optimState = tablex.deepcopy(opt)
  end
  local function t(f) local s = torch.Timer(); return f(), s:time().real end

  local loss, train_time = t(train)
  local test_acc, test_time = t(test)

  print(('Test accuracy: '..c.cyan'%.2f'..' %%\t '):format(test_acc))
  
  losses_epoch[#losses_epoch+1]=loss
  test_accus[#test_accus+1]=test_acc
  train_times[#train_times+1]=train_time
  test_times[#test_times+1]=test_time

  print('train time:'..train_time)

  print('test time:'..test_time)
  

  log{
     loss = loss,
     epoch = epoch,
     test_acc = test_acc,
     lr = opt.learningRate,
     train_time = train_time,
     test_time = test_time,
   }


end
results={}
results.opt=opt
results.losses=losses
results.train_accus=train_accus
results.test_accus=test_accus
results.losses_epoch=losses_epoch
results.train_times=train_times
results.test_times=test_times
torch.save('set_result/Conv/result_4Splain_deep_'..opt.model..'_depth'..opt.depth
..'_h'..opt.hidden_number..'_lr'..opt.learningRate
..'_G'..opt.m_perGroup..'_b'..opt.batchSize..'_wf'..opt.widen_factor..'_s'..opt.scaleIdentity
..'_wD'..opt.weightDecay..'_mm'..opt.momentum
..'_nN'..opt.noNesterov..'_lD'..opt.learningRateDecay
..'BS'..opt.BNScale..'_seed'..opt.seed
..'.dat',results)
