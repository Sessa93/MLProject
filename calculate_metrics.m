function [precs, recs] = calculate_metrics(predicted, actual, K)
    labels = { 'unknown','star', 'absorption galaxy', 'galaxy', 'emission galaxy', 'narrow-line QSO', 'broad-line QSO', 'sky', 'Hi-z QSO', 'Late-type star'};
    conf = confusionmat(predicted,actual);
    precs = [];
    recs = [];
    for kk=1:K
        tp_k = conf(kk,kk);
        col = conf(:,kk);
        fp_k = sum(col([1:kk-1,kk+1:end]));
        sub = conf([1:kk-1,kk+1:end],:);
        tn_k = sum(sub(:));
        row = conf(kk,:);
        fn_k = sum(row([1:kk-1,kk+1:end]));
        precs = [precs tp_k/(tp_k + fp_k)];
        recs = [recs tp_k/(tp_k + fn_k)];
    end 
    heatmap(conf, [1:K], [1:K], '%0.2f', 'Colormap', 'money', 'ColorLevels', 50);
    title('Confusion Matrix')
    xlabel('Predicted Class')
    ylabel('Actual Class')
end

