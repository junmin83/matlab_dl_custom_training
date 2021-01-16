[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;

dsXTrain = arrayDatastore(XTrain,'IterationDimension',4);
dsYTrain = arrayDatastore(YTrain);
dsAnglesTrain = arrayDatastore(anglesTrain);

dsTrain = combine(dsXTrain,dsYTrain,dsAnglesTrain);

classNames = categories(YTrain);
numClasses = numel(classNames);
numResponses = size(anglesTrain,2);
numObservations = numel(YTrain);

idx = randperm(numObservations,64);
I = imtile(XTrain(:,:,:,idx));
figure
imshow(I)

filterSize = [5 5];
numChannels = 1;
numFilters = 16;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.conv1.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv1.Bias = initializeZeros([numFilters 1]);

parameters.batchnorm1.Offset = initializeZeros([numFilters 1]);
parameters.batchnorm1.Scale = initializeOnes([numFilters 1]);
state.batchnorm1.TrainedMean = zeros(numFilters,1,'single');
state.batchnorm1.TrainedVariance = ones(numFilters,1,'single');

filterSize = [3 3];
numChannels = 16;
numFilters = 32;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.conv2.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv2.Bias = initializeZeros([numFilters 1]);

parameters.batchnorm2.Offset = initializeZeros([numFilters 1]);
parameters.batchnorm2.Scale = initializeOnes([numFilters 1]);
state.batchnorm2.TrainedMean = zeros(numFilters,1,'single');
state.batchnorm2.TrainedVariance = ones(numFilters,1,'single');

filterSize = [3 3];
numChannels = 32;
numFilters = 32;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.conv3.Weights = initializeGlorot(sz,numOut,numIn);
parameters.conv3.Bias = initializeZeros([numFilters 1]);

parameters.batchnorm3.Offset = initializeZeros([numFilters 1]);
parameters.batchnorm3.Scale = initializeOnes([numFilters 1]);
state.batchnorm3.TrainedMean = zeros(numFilters,1,'single');
state.batchnorm3.TrainedVariance = ones(numFilters,1,'single');

filterSize = [1 1];
numChannels = 16;
numFilters = 32;

sz = [filterSize numChannels numFilters];
numOut = prod(filterSize) * numFilters;
numIn = prod(filterSize) * numFilters;

parameters.convSkip.Weights = initializeGlorot(sz,numOut,numIn);
parameters.convSkip.Bias = initializeZeros([numFilters 1]);

parameters.batchnormSkip.Offset = initializeZeros([numFilters 1]);
parameters.batchnormSkip.Scale = initializeOnes([numFilters 1]);
state.batchnormSkip.TrainedMean = zeros([numFilters 1],'single');
state.batchnormSkip.TrainedVariance = ones([numFilters 1],'single');

sz = [numClasses 6272];
numOut = numClasses;
numIn = 6272;
parameters.fc1.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc1.Bias = initializeZeros([numClasses 1]);

sz = [numResponses 6272];
numOut = numResponses;
numIn = 6272;
parameters.fc2.Weights = initializeGlorot(sz,numOut,numIn);
parameters.fc2.Bias = initializeZeros([numResponses 1]);

parameters
parameters.conv1
state
state.batchnorm1

numEpochs = 20;
miniBatchSize = 128;

plots = "training-progress";

mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','',''});

trailingAvg = [];
trailingAvgSq = [];

if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq)
    
    % Loop over mini-batches
    while hasdata(mbq)
    
        iteration = iteration + 1;
        
        [dlX,dlY1,dlY2] = next(mbq);
              
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function.
        [gradients,state,loss] = dlfeval(@modelGradients, parameters, dlX, dlY1, dlY2, state);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,iteration);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end

[XTest,YTest,anglesTest] = digitTest4DArrayData;

dsXTest = arrayDatastore(XTest,'IterationDimension',4);
dsYTest = arrayDatastore(YTest);
dsAnglesTest = arrayDatastore(anglesTest);

dsTest = combine(dsXTest,dsYTest,dsAnglesTest);

mbqTest = minibatchqueue(dsTest,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','',''});


doTraining = false;

classesPredictions = [];
anglesPredictions = [];
classCorr = [];
angleDiff = [];

% Loop over mini-batches.
while hasdata(mbqTest)
    
    % Read mini-batch of data.
    [dlXTest,dlY1Test,dlY2Test] = next(mbqTest);
    
    % Make predictions using the predict function.
    [dlY1Pred,dlY2Pred] = model(parameters,dlXTest,doTraining,state);
    
    % Determine predicted classes.
    Y1PredBatch = onehotdecode(dlY1Pred,classNames,1);
    classesPredictions = [classesPredictions Y1PredBatch];
    
    % Dermine predicted angles
    Y2PredBatch = extractdata(dlY2Pred);
    anglesPredictions = [anglesPredictions Y2PredBatch];
    
    % Compare predicted and true classes
    Y1Test = onehotdecode(dlY1Test,classNames,1);
    classCorr = [classCorr Y1PredBatch == Y1Test];
    
    % Compare predicted and true angles
    angleDiffBatch = Y2PredBatch - dlY2Test;
    angleDiff = [angleDiff extractdata(gather(angleDiffBatch))];
    
end


accuracy = mean(classCorr)

angleRMSE = sqrt(mean(angleDiff.^2))

idx = randperm(size(XTest,4),9);
figure
for i = 1:9
    subplot(3,3,i)
    I = XTest(:,:,:,idx(i));
    imshow(I)
    hold on
    
    sz = size(I,1);
    offset = sz/2;
    
    thetaPred = anglesPredictions(idx(i));
    plot(offset*[1-tand(thetaPred) 1+tand(thetaPred)],[sz 0],'r--')
    
    thetaValidation = anglesTest(idx(i));
    plot(offset*[1-tand(thetaValidation) 1+tand(thetaValidation)],[sz 0],'g--')
    
    hold off
    label = string(classesPredictions(idx(i)));
    title("Label: " + label)
end

function [dlY1,dlY2,state] = model(parameters,dlX,doTraining,state)

% Convolution
weights = parameters.conv1.Weights;
bias = parameters.conv1.Bias;
dlY = dlconv(dlX,weights,bias,'Padding','same');

% Batch normalization, ReLU
offset = parameters.batchnorm1.Offset;
scale = parameters.batchnorm1.Scale;
trainedMean = state.batchnorm1.TrainedMean;
trainedVariance = state.batchnorm1.TrainedVariance;

if doTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm1.TrainedMean = trainedMean;
    state.batchnorm1.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
end

dlY = relu(dlY);

% Convolution, batch normalization (Skip connection)
weights = parameters.convSkip.Weights;
bias = parameters.convSkip.Bias;
dlYSkip = dlconv(dlY,weights,bias,'Stride',2);

offset = parameters.batchnormSkip.Offset;
scale = parameters.batchnormSkip.Scale;
trainedMean = state.batchnormSkip.TrainedMean;
trainedVariance = state.batchnormSkip.TrainedVariance;

if doTraining
    [dlYSkip,trainedMean,trainedVariance] = batchnorm(dlYSkip,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnormSkip.TrainedMean = trainedMean;
    state.batchnormSkip.TrainedVariance = trainedVariance;
else
    dlYSkip = batchnorm(dlYSkip,offset,scale,trainedMean,trainedVariance);
end

% Convolution
weights = parameters.conv2.Weights;
bias = parameters.conv2.Bias;
dlY = dlconv(dlY,weights,bias,'Padding','same','Stride',2);

% Batch normalization, ReLU
offset = parameters.batchnorm2.Offset;
scale = parameters.batchnorm2.Scale;
trainedMean = state.batchnorm2.TrainedMean;
trainedVariance = state.batchnorm2.TrainedVariance;

if doTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm2.TrainedMean = trainedMean;
    state.batchnorm2.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
end

dlY = relu(dlY);

% Convolution
weights = parameters.conv3.Weights;
bias = parameters.conv3.Bias;
dlY = dlconv(dlY,weights,bias,'Padding','same');

% Batch normalization
offset = parameters.batchnorm3.Offset;
scale = parameters.batchnorm3.Scale;
trainedMean = state.batchnorm3.TrainedMean;
trainedVariance = state.batchnorm3.TrainedVariance;

if doTraining
    [dlY,trainedMean,trainedVariance] = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
    
    % Update state
    state.batchnorm3.TrainedMean = trainedMean;
    state.batchnorm3.TrainedVariance = trainedVariance;
else
    dlY = batchnorm(dlY,offset,scale,trainedMean,trainedVariance);
end

% Addition, ReLU
dlY = dlYSkip + dlY;
dlY = relu(dlY);

% Fully connect, softmax (labels)
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
dlY1 = fullyconnect(dlY,weights,bias);
dlY1 = softmax(dlY1);

% Fully connect (angles)
weights = parameters.fc2.Weights;
bias = parameters.fc2.Bias;
dlY2 = fullyconnect(dlY,weights,bias);

end

function [gradients,state,loss] = modelGradients(parameters,dlX,T1,T2,state)

doTraining = true;
[dlY1,dlY2,state] = model(parameters,dlX,doTraining,state);

lossLabels = crossentropy(dlY1,T1);
lossAngles = mse(dlY2,T2);

loss = lossLabels + 0.1*lossAngles;
gradients = dlgradient(loss,parameters);

end

function [X,Y,angle] = preprocessMiniBatch(XCell,YCell,angleCell)
    
    % Extract image data from cell and concatenate
    X = cat(4,XCell{:});
    % Extract label data from cell and concatenate
    Y = cat(2,YCell{:});
    % Extract angle data from cell and concatenate
    angle = cat(2,angleCell{:});
        
    % One-hot encode labels
    Y = onehotencode(Y,1);
        
end

function weights = initializeGlorot(sz,numOut,numIn)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end

function parameter = initializeZeros(sz)

parameter = zeros(sz,'single');
parameter = dlarray(parameter);

end

function parameter = initializeOnes(sz)

parameter = ones(sz,'single');
parameter = dlarray(parameter);

end



