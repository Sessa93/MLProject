function err = svmwrapper(xTrain, yTrain, xTest, yTest)
  C = [-10,5,10];
  G = [-10,5,10];
  CF = [-10,5,10];
  best_acc = 0;
  
  for cc=1:3
      for gg=1:3
          for ff=1:3
              acc = libsvmtrain(yTrain, xTrain, sprintf('-t 0 -v 5 -q -c %f -g %f -r %f',2^C(cc),2^G(gg),2^CF(ff)));
              if acc >= best_acc
                  best_acc = acc;
                  best_c = C(cc);
                  best_g = G(gg);
                  best_cf = CF(ff);
              end
          end
      end
  end
   
  model = libsvmtrain(yTrain, xTrain, sprintf('-t 0 -q -c %f -g %f -r %f',2^best_c,2^best_g,2^best_cf));
  err = sum(libsvmpredict(yTest, xTest, model,'-q') ~= yTest);
end