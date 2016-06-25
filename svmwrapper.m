function err = svmwrapper(xTrain, yTrain, xTest, yTest)
  model = libsvmtrain(yTrain, xTrain, sprintf('-t 0 -q -c %f',1));
  err = sum(libsvmpredict(yTest, xTest, model,'-q') ~= yTest);
end