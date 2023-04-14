function plotPerformance(trainInfo)
    figure

    % Training loss
    subplot(2, 1, 1)
    plot(trainInfo.TrainingLoss, 'LineWidth', 2)
    hold on
    plot(trainInfo.ValidationLoss, 'LineWidth', 2)
    hold off
    title('Loss')
    xlabel('Iteration')
    ylabel('Loss')
    legend('Training', 'Validation')

    % Training and validation accuracy
    subplot(2, 1, 2)
    plot(trainInfo.TrainingAccuracy, 'LineWidth', 2)
    hold on
    plot(trainInfo.ValidationAccuracy, 'LineWidth', 2)
    hold off
    title('Accuracy')
    xlabel('Iteration')
    ylabel('Accuracy (%)')
    legend('Training', 'Validation')
end
