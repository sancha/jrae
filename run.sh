
java -jar jar/jrae.jar \
-DataDir data/mov \
-maxIterations 80 \
-ModelFile data/mov/tunedTheta.rae \
-NumCores 4 \
-TrainModel True

java -jar jar/jrae.jar \
-DataDir data/
-MaxIterations 80
-ModelFile data/mov/tunedTheta.rae
-ClassifierFile data/mov/Softmax.clf
-NumCores 2
-TrainModel False
-ProbabilitiesOutputFile data/tiny/prob.out
