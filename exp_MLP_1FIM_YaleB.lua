-------------------------------------------------------------------------------------------------
require 'xlua'
require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'


require 'image'

require 'models/MLP/model_DBN_FIM'
local c = require 'trepl.colorize'

-- threads
threadNumber=2
torch.setnumthreads(threadNumber)

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('compare the Decorelated BatchNormalizaiton method with baselines on MLP architechture')
cmd:text()
cmd:text('Options')

cmd:option('-model_method','sgd_F2','the methods: options: sgd_F2, batch_F2, nnn_F2, layer_F2, DBN_F2')
cmd:option('-mode_nonlinear',2,'nonlinear module: 1 indicates tanh, 0 indicates sigmoid, 2 indecates Relu')
cmd:option('-max_epoch',3000,'maximum number of iterations')
cmd:option('-n_hidden_number',128,'the dimension of the hidden laysers')
cmd:option('-save',"log_MLP_1FIM_YaleB" ,'subdirectory to save logs')
cmd:option('-inputScaled',true,'whether preoprocess the input, scale to (0,1)')
cmd:option('-inputCentered',true,'whether preoprocess the input, minus the mean')

cmd:option('-batchSize',2033,'the number of examples per batch')
cmd:option('-learningRate',1,'learning rate')
cmd:option('-weightDecay',0,'weight Decay for regularization')
cmd:option('-momentum',0,'momentum')

-------------for DBN and DBN_var method----------------
cmd:option('-m_perGroup',128,'the number of per group')
cmd:option('-m_perGroup_WDBN',128,'the number of per group')

cmd:option('-optimization','simple','the methods: options:adam,simple,rms,adagrad,lbfgs')
---------------for nnn BatchLinear_NoBP realted method----------------
cmd:option('-T',63,'the interval to update the coefficient')
cmd:option('-epcilo',1,'the revision term for natural neural network')
cmd:option('-lrD_epoch',10000,'the epoch to half the learningRate')

cmd:option('-conditionMode','FIM','optiom: FIM or AH, AH means approximate Hession')
cmd:option('-seed',1,'the random seed')
cmd:option('-FIM_intervalT',20,'')


cmd:text()
-- parse input params
opt = cmd:parse(arg)

--opt.rundir = cmd:string('log_MLP_8Final', opt, {dir=true})
--paths.mkdir(opt.rundir)
-- create log file
--cmd:log(opt.rundir .. '/log', opt)

torch.manualSeed(opt.seed)    -- fix random seed so program runs the same every time
 trainData = torch.load('./dataset/YaleB/YaleB_train.dat')
 testData = torch.load('./dataset/YaleB/YaleB_test.dat')

opt.FIM_number=2000
--opt.FIM_intervalT=10000 --the interval to calculate the condition Number

opt.orth_intial=false
opt.Ns=0.1*opt.T --used for nnn and BN_NoBP method.

opt.printInterval=10
counter_forFIM_calculation=0
temp_buffer=torch.Tensor()

if opt.optimization == 'lbfgs' then
  opt.optimState = {
    learningRate = opt.learningRate,
    maxIter = 2,
    nCorrection = 10
  }
 optimMethod = optim.lbfgs
elseif opt.optimization == 'simple' then
  opt.optimState = {
    learningRate =opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = 0
  }
 optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  opt.optimState = {
    learningRate = opt.learningRate,
  }
 optimMethod = optim.adagrad
elseif opt.optimization == 'rms' then
  opt.optimState = {
    learningRate = opt.learningRate,
    alpha=0.9
  }
 optimMethod = optim.rmsprop
 elseif opt.optimization == 'adam' then
  opt.optimState = {
    learningRate = opt.learningRate
  }
 optimMethod = optim.adam
elseif opt.optimization == 'adadelta' then
  opt.optimState = {
    learningRate = opt.learningRate
  }
 optimMethod = optim.adadelta
else
  error('Unknown optimizer')
end



opt.n_inputs=trainData.data:size(2) 
opt.n_outputs=trainData.labels:max()
 -- scale to (0,1)
if opt.inputScaled then
  for i=1, testData.data:size(1) do
   testData.data[i]:div(255)
  end
  for i=1, trainData.data:size(1) do
   trainData.data[i]:div(255)
  end
end
--0 mean-----
if opt.inputCentered then
  local mean =  trainData.data:mean()
  trainData.data:add(-mean)
  local mean =  testData.data:mean()
  testData.data:add(-mean)
end

function evaluateAccuracy(prediction, y)
    -- load
  correct=0;
  length=y:size(1)
  for i=1, length do
    if prediction[i][1]==y[i] then--the type of prediction is 2 D Tensor, y is vector
      correct=correct+1;
    end
      
  end
    accuracy=correct/length
    return accuracy
end

model, criterion = create_model(opt)
 
 confusion = optim.ConfusionMatrix(opt.n_outputs)
print('Will save at '..opt.save)
paths.mkdir(opt.save)

log_name=opt.model_method..'_'..opt.optimization..'_lr'..opt.learningRate..'_g'..opt.m_perGroup..'.log'

testLogger = optim.Logger(paths.concat(opt.save, log_name))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

 parameters, gradParameters = model:getParameters()
 print(model)
------------------------------------------------------------------------
-- training       
------------------------------------------------------------------------

 function train()
  model:training()
  epoch = epoch or 1
 
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local targets = torch.Tensor(opt.batchSize)
  local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

 -- local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = trainData.data:index(1,v)
    targets:copy(trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
     --print(outputs)
      confusion:batchAdd(outputs, targets)
      if iteration % opt.printInterval ==0 then
          print(string.format("minibatches processed: %6s, loss = %6.6f", iteration, f))
      end
      losses[#losses + 1] = f

      if f>1000 then
         --print('loss is exploding, aborting.')
         --os.exit()
      end

     timeCosts[#timeCosts+1]=torch.toc(start_time)
     -- print(string.format("time Costs = %6.6f", timeCosts[#timeCosts]))
      iteration=iteration+1
   counter_forFIM_calculation=counter_forFIM_calculation+1

      return f,gradParameters
    end

   optimMethod (feval, parameters, opt.optimState)
   
      if ((string.match(opt.model_method,'nnn'))  and iteration % opt.T ==0) then
    ------------------start:update the proMatrix of NNN-------------------------------
        local index = torch.randperm(trainData.data:size(1))[{{1, opt.Ns}}]:long()
        local  batch_inputs=trainData.data:index(1,index)
        local  batch_targets = trainData.labels:index(1,index)
        local  batch_outputs = model:forward(batch_inputs)
    
        --model:updateNormLinearParameter(batch_inputs, dloss_doutput, scale,  opt.Ns, opt.epcilo)
        for k,v in pairs(model:findModules('nn.NormLinear_Validation')) do
        v:updatePromatrix(opt.epcilo)
        end
     end
    ------------------end:update the proMatrix of NNN-------------------------------

 
     --------------------------start: calculate the condition number of FIM---------------------------
    if  counter_forFIM_calculation % opt.FIM_intervalT == 0 then
       print('---------start the method to calculate FIM---------------')
       model:evaluate() --do not evaluate the validation of the output/input/gradInput
       for j=1,model:size() do
         if string.match(model.modules[j].__typename , 'nn.DecorelateBN') or model.modules[j].__typename=='nn.BatchLinear_FIM' then
           model.modules[j]:setTrainMode(true) --use the training mode or testing mode to calculate the p(y|x)
         end
       end
       --start: data
       local indices = torch.randperm(trainData.data:size(1)):long()[{{1,opt.FIM_number}}]
       local inputs = trainData.data:index(1,indices)
       local labels = trainData.labels:index(1,indices)

       --end: data
       local outputs = model:forward(inputs)
      local loss_FIM = criterion:forward(outputs, labels:squeeze())
       local dloss_doutput_FIM

       if opt.conditionMode=='FIM' then
           print('--------FIM--------')
      --------------------start: calculate expected dloos_doutput-------------------
        local classProbabilities=torch.exp(outputs)
        local dloss_doutput_exp=torch.Tensor():resizeAs(outputs):zero()

        for i=1, opt.n_outputs do
           local temp_label=torch.Tensor(outputs:size(1)):fill(i)
           local temp_do=criterion:backward(outputs, temp_label)
           temp_buffer:repeatTensor(classProbabilities:select(2,i), opt.n_outputs, 1)
           temp_do:cmul(temp_buffer:t())
           dloss_doutput_exp=dloss_doutput_exp+ temp_do
        end
        dloss_doutput_FIM=dloss_doutput_exp
     ----------------------end:calculate expected dloos_doutput-----------------------
    elseif  opt.conditionMode=='AH' then
         print('--------AH--------')
         dloss_doutput_FIM = criterion:backward(outputs, labels)
       end


     for k,v in pairs(model:findModules('nn.Linear_Validation')) do
        v:update_FIM_flag(true) --calculate FIM
     end
     for k,v in pairs(model:findModules('nn.NormLinear_Validation')) do
        v:update_FIM_flag(true) --calculate FIM
     end

       model:backward(trainData.data[{{1,opt.FIM_number},{}}], dloss_doutput_FIM)
     for k,v in pairs(model:findModules('nn.Linear_Validation')) do
        v:update_FIM_flag(false) --calculate FIM
     end
     for k,v in pairs(model:findModules('nn.NormLinear_Validation')) do
        v:update_FIM_flag(false) --calculate FIM
     end
       model:training()
    end
      --------------------------end: calculate the condition number of FIM---------------------------


   
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(start_time)))

  train_acc = confusion.totalValid * 100
  train_accus[#train_accus+1]=train_acc
  confusion:zero()
  
  if epoch % opt.lrD_epoch ==0 then
     opt.optimState.learningRate=opt.optimState.learningRate / 2
     print('new learningRate:'..opt.optimState.learningRate) 
  end
  
  epoch = epoch + 1
end


function test()

 model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 38
  for i=1,testData.data:size(1),bs do
    local outputs = model:forward(testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
   test_accus[#test_accus+1]=confusion.totalValid * 100
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()
   end
  confusion:zero()

end

iteration=0
losses={}
timeCosts={}
train_times={}
test_times={}
train_accus={}
test_accus={}
start_time=torch.tic()
for i=1,opt.max_epoch do
  local function t(f) local s = torch.Timer();f() return  s:time().real end
  local  train_time = t(train)
 train_times[#train_times+1]=train_time
 print('train Time:'..train_time)

  local  test_time = t(test)
  test_times[#test_times+1]=test_time
  print('test Time:'..test_time)

end

conditionNumber_FIMs={}
conditionNumber_FIMs_90PerCent={}
if string.match(opt.model_method,'sgd') or string.match(opt.model_method,'batch') or string.match(opt.model_method,'nnn') or string.match(opt.model_method,'layer')   or string.match(opt.model_method,'DBN') then
  -- print('------------match-----------')
      for k,v in pairs(model:findModules('nn.Linear_Validation')) do
        table.insert(conditionNumber_FIMs,v.conditionNumber_FIM)
        table.insert(conditionNumber_FIMs_90PerCent,v.conditionNumber_FIM_90PerCent)
     end
     for k,v in pairs(model:findModules('nn.NormLinear_Validation')) do
        table.insert(conditionNumber_FIMs,v.conditionNumber_FIM)
        table.insert(conditionNumber_FIMs_90PerCent,v.conditionNumber_FIM_90PerCent)
     end
end

results={}
opt.optimState=nil
results.opt=opt
results.losses=losses
results.train_accus=train_accus
results.test_accus=test_accus
results.conditionNumber_FIMs=conditionNumber_FIMs
results.conditionNumber_FIMs_90PerCent=conditionNumber_FIMs_90PerCent
results.train_times=train_times
results.test_times=test_times

torch.save('set_result/MLP/result_1FIM_YaleB_'..opt.model_method..
'_'..opt.optimization..'_b'..opt.batchSize..'_lr'..opt.learningRate..
'_nl'..opt.mode_nonlinear..'_mm'..opt.momentum..
'_ep'..opt.epcilo..'_T'..opt.T..
'_g'..opt.m_perGroup..'_gWDBN'..opt.m_perGroup_WDBN..'_lrDe'..opt.lrD_epoch..
'_seed'..opt.seed..
'.dat',results)

 
