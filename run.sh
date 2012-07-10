#!/bin/bash

## Compile as 'javac -d bin/ -classpath .:libs/* -Xlint `find src | grep java$`'

## Uses 1G of memory and specialized garbage collector for parallel programs. 

java -Xms1g -Xmx1g -XX:+UseTLAB -XX:+UseConcMarkSweepGC -cp .:bin/:libs/* main.RAEBuilder \
-DataDir data/mov \
-MaxIterations 20 \
-ModelFile data/mov/tunedTheta.rae \
-ClassifierFile data/mov/Softmax.clf \
-NumCores 3 \
-TrainModel True \
-ProbabilitiesOutputFile data/mov/prob.out \
-TreeDumpDir data/mov/trees

#java -Xms1g -Xmx1g -XX:+UseTLAB -XX:+UseConcMarkSweepGC -cp .:bin/:libs/* main.RAEBuilder \
#-DataDir data/
#-MaxIterations 80
#-ModelFile data/mov/tunedTheta.rae
#-ClassifierFile data/mov/Softmax.clf
#-NumCores 2
#-TrainModel False
#-ProbabilitiesOutputFile data/tiny/prob.out
