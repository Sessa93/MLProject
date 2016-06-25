function err = naivebwrapper(xTrain, xTest, yTrain, yTest)
    model = fitcnb(xTrain,yTrain);
    pred = predict(model,xTest);
    err = sum(pred ~= yTest);
end