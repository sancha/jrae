
java -jar jar/jrae.jar \
-DataDir data/mov \
-maxIterations 80 \
-ModelFile data/mov/tunedTheta.rae \
-NumCores 4 \
-TrainModel True

java -jar jar/jrae.jar \
-DataDir data/mov \
-maxIterations 80 \
-ModelFile data/mov/tunedTheta.rae \
-NumCores 4 \
-TrainModel False \
-FeaturesOutputFile data/mov/features.out \
-ProbabilitiesOutputFile data/mov/prob.out

