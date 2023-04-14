function [trainedNet, trainInfo] = trainVGG19(trainDatastore, train_data, validationDatastore)

    net = vgg19;
    lgraph = layerGraph(net);

    % Modify the output layer for the new classification problem
    numClasses = numel(categories(train_data.Labels));
    new_fc = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
    new_softmax = softmaxLayer('Name', 'new_softmax');
    new_class = classificationLayer('Name', 'new_class');

    % Replace the last layers
    lgraph = replaceLayer(lgraph, 'fc8', new_fc);
    lgraph = replaceLayer(lgraph, 'prob', new_softmax);
    lgraph = replaceLayer(lgraph, 'output', new_class);

    % Set up training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 50, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', validationDatastore, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Train the neural network
    [trainedNet, trainInfo] = trainNetwork(trainDatastore, lgraph, options);
end
