# Image-Classification-using-Fine-tuned-Pretrained-Networks-in-MATLAB
This repository contains MATLAB code for training and evaluating image classification models using pre-trained deep learning networks. We use ResNet-50, VGG-19, and VGG-16 as the pre-trained networks and fine-tune them on a custom dataset. The project includes data preparation, training, evaluation, and performance visualization.

## Getting Started
Clone this repository to your local machine and make sure you have MATLAB with Deep Learning Toolbox installed.

## Dataset Preparation
Organize your dataset into three separate folders: train, val, and test. Each folder should contain subfolders representing different classes, and the images should be placed within their respective class subfolders.

Example folder structure:
```
/MATLAB Drive/train/
    /class1/
        img1.jpg
        img2.jpg
        ...
    /class2/
        img1.jpg
        img2.jpg
        ...
    ...

/MATLAB Drive/val/
    /class1/
        img1.jpg
        img2.jpg
        ...
    /class2/
        img1.jpg
        img2.jpg
        ...
    ...

/MATLAB Drive/test/
    /class1/
        img1.jpg
        img2.jpg
        ...
    /class2/
        img1.jpg
        img2.jpg
        ...
    ...

```
## Data Preparation
The prepareData function creates image datastores for the training, validation, and test data. It also applies data augmentation on the fly during training to improve the model's performance.

Usage:
```matlab
[trainDatastore, validationDatastore, testDatastore] = prepareData(train_folder, validation_folder, test_folder);
```
## Training Models
The following functions train a deep learning model using ResNet-50, VGG-19, and VGG-16 as pre-trained networks:

- trainResNet50
- trainVGG19
- trainVGG16
Usage:
```matlab
[resNet50Model, resNet50Info] = trainResNet50(trainDatastore, validationDatastore);
[vgg19Model, vgg19Info] = trainVGG19(trainDatastore, validationDatastore);
[vgg16Model, vgg16Info] = trainVGG16(trainDatastore, validationDatastore);
```
## Evaluation
The evaluate function calculates accuracy, precision, recall, and F1-score for a trained model on the test dataset.

Usage:
```matlab
[resNet50Acc, resNet50Prec, resNet50Recall, resNet50F1] = evaluate(resNet50Model, testDatastore);
[vgg19Acc, vgg19Prec, vgg19Recall, vgg19F1] = evaluate(vgg19Model, testDatastore);
[vgg16Acc, vgg16Prec, vgg16Recall, vgg16F1] = evaluate(vgg16Model, testDatastore);
```
## Performance Visualization
The plotPerformance function plots the training accuracy, training loss, validation accuracy, and validation loss for a trained model.

Usage:
```matlab
plotPerformance(resNet50Info);
plotPerformance(vgg19Info);
plotPerformance(vgg16Info);
```
