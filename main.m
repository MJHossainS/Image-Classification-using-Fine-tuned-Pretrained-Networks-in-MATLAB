

train_folder = '/MATLAB Drive/train';
test_folder = '/MATLAB Drive/test';
validation_folder = '/MATLAB Drive/val';

%[trainDatastore, validationDatastore, testDatastore] = prepareData(train_folder, validation_folder, test_folder);

% Load and preprocess the data
[trainDatastore, validationDatastore, testDatastore, train_data] = prepareData(train_folder, validation_folder, test_folder);

% Train the neural network
%[trainedNet, trainInfo] = trainResNet50(trainDatastore, train_data, validationDatastore);


% Train the VGG19 neural network
[trainedVGG19Net, trainVGG19Info] = trainVGG19(trainDatastore, train_data, validationDatastore);

% Train the VGG16 neural network
%[trainedVGG16Net, trainVGG16Info] = trainVGG16(trainDatastore, train_data, validationDatastore);



% Evaluate the models
%[resNet50Acc, resNet50Prec, resNet50Recall, resNet50F1] = evaluate(resNet50Model, testDatastore);
[vgg19Acc, vgg19Prec, vgg19Recall, vgg19F1] = evaluate(vgg19Model, testDatastore);
%[vgg16Acc, vgg16Prec, vgg16Recall, vgg16F1] = evaluate(vgg16Model, testDatastore);

% Plot the performance
%plotPerformance(resNet50Info);
plotPerformance(vgg19Info);
%plotPerformance(vgg16Info);

