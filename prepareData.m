function [trainDatastore, validationDatastore, testDatastore, train_data] = prepareData(train_folder, validation_folder, test_folder)
    % Define the image size for input into the neural network
    imageSize = [224, 224, 1];

    % Create an imageDatastore to store the images
    train_data = imageDatastore(fullfile(train_folder), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    test_data = imageDatastore(fullfile(test_folder), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    validation_data = imageDatastore(fullfile(validation_folder), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % Preprocess the images
    train_data.ReadFcn = @(filename) preprocess_image(filename, imageSize);
    test_data.ReadFcn = @(filename) preprocess_image(filename, imageSize);
    validation_data.ReadFcn = @(filename) preprocess_image(filename, imageSize);

    % Define data augmentation
    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandRotation', [-10, 10], ...
        'RandScale', [0.9, 1.1]);

    % Create augmentedImageDatastores
    trainDatastore = augmentedImageDatastore(imageSize, train_data, 'DataAugmentation', augmenter);
    validationDatastore = augmentedImageDatastore(imageSize, validation_data);
    testDatastore = augmentedImageDatastore(imageSize, test_data);
end


function img = preprocess_image(filename, imageSize)
    img = imread(filename);
    % Convert grayscale images to RGB
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end
    img = imresize(img, imageSize(1:2));
    img = rescale(img, 0, 1);
end

