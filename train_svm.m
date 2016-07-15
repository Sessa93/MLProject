function [best_gamma,best_C,cv_acc] = train_svm(data,target)
    %# grid of parameters
    folds = 5;
    [C,gamma] = meshgrid(-5:5:20, -20:5:-4);
    
    %# grid search, and cross-validation
    cv_acc = zeros(numel(C),1);
    for i=1:numel(C)
        curr = [C(i),gamma(i)]
        cv_acc(i) = libsvmtrain(target, data, sprintf('-t 2 -c %f -g %f -v %d -q', 2^C(i), 2^gamma(i), folds));
    end

    %# pair (C,gamma) with best accuracy
    [~,idx] = max(cv_acc);

    %# contour plot of paramter selection
    contour(C, gamma, reshape(cv_acc,size(C))), colorbar
    hold on
    plot(C(idx), gamma(idx), 'rx')
    text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
    hold off
    xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

    %# now you can train you model using best_C and best_gamma
    best_C = 2^C(idx);
    best_gamma = 2^gamma(idx);
end

