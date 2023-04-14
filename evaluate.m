function [accuracy, precision, recall, f1score] = evaluate(trainedNet, testDatastore)
    % Get predictions on test data
    predictedLabels = classify(trainedNet, testDatastore);
    trueLabels = testDatastore.Labels;

    % Calculate confusion matrix
    confMat = confusionmat(trueLabels, predictedLabels);
    
    % Calculate accuracy
    accuracy = sum(diag(confMat)) / sum(confMat, 'all');
    
    % Calculate precision, recall, and F1-score
    precision = diag(confMat) ./ sum(confMat, 2);
    recall = diag(confMat) ./ sum(confMat, 1)';
    f1score = 2 * (precision .* recall) ./ (precision + recall);
    
    % Average precision, recall, and F1-score
    precision = mean(precision);
    recall = mean(recall);
    f1score = mean(f1score);
end
