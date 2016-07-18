function [best_gamma,best_C, best_d, cv_acc] = train_svm_poly(data,target)
    %# grid of parameters
    folds = 5;
    [C,gamma] = meshgrid(5:5:20, -10:5:-5);
    acc_d = []
    for dd =2:5
        %# grid search, and cross-validation
        cv_acc = zeros(numel(C),1);
        for i=1:numel(C)
            curr = [C(i),gamma(i)]
            cv_acc(i) = libsvmtrain(target, data, sprintf('-t 1 -c %f -g %f -v %d -d %f -q', 2^C(i), 2^gamma(i), folds, dd));
        end

        %# pair (C,gamma) with best accuracy
        [max_cv,idx] = max(cv_acc);
        acc_d = [acc_d max_cv];
        %# contour plot of paramter selection
        figure()
        contour(C, gamma, reshape(cv_acc,size(C))), colorbar
        hold on
        plot(C(idx), gamma(idx), 'rx')
        text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
        'HorizontalAlign','left', 'VerticalAlign','top')
        hold off
        xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')
    end

    %# now you can train you model using best_C and best_gamma
    best_C = 2^C(idx);
    best_gamma = 2^gamma(idx);
    [~,d] = max(acc_d);
    best_d = d+1;
end

