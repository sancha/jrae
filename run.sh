java -jar jar/jrae.jar \
-DataDir data/tiny.mov \
-MaxIterations 20 \
-ModelFile data/tiny.mov/tunedTheta.rae \
-ClassifierFile data/tiny.mov/Softmax.clf \
-NumCores 3 \
-TrainModel True \
-ProbabilitiesOutputFile data/tiny.mov/prob.out \
-TreeDumpDir data/tiny.mov/trees 

#java -jar jar/jrae.jar \
#-DataDir data/
#-MaxIterations 80
#-ModelFile data/mov/tunedTheta.rae
#-ClassifierFile data/mov/Softmax.clf
#-NumCores 2
#-TrainModel False
#-ProbabilitiesOutputFile data/tiny/prob.out
