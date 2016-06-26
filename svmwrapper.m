function err = svmwrapper(xTrain, yTrain, xTest, yTest)
  model = libsvmtrain(yTrain, xTrain, sprintf('-t 0 -q -c %f -g %f',2^4,2^-14));
  err = sum(libsvmpredict(yTest, xTest, model,'-q') ~= yTest);
end