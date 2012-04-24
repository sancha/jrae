package main;

import java.util.*;
import java.io.*;

import math.*;

import org.jblas.*;

import rae.FineTunableTheta;
import rae.RAECost;
import rae.RAEFeatureExtractor;

import classify.*;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class FullRun {
	
	public static void main(final String[] args) throws Exception
	{
		Arguments params = new Arguments();
		params.parseArguments(args);

		if(params.exitOnReturn)
			return;
		
		RAECost RAECost = null;
		FineTunableTheta InitialTheta = null;
		RAEFeatureExtractor FeatureExtractor = null;
		DifferentiableMatrixFunction f = new Norm1Tanh(); 
		
		System.out.printf("%d\n%d\n",params.DictionarySize,params.hiddenSize);
		
		StratifiedCrossValidation<LabeledDatum<Integer,Integer>,Integer,Integer> cv 
			= new StratifiedCrossValidation<LabeledDatum<Integer,Integer>,Integer,Integer>(params.NumFolds, params.Dataset);
		FineTunableTheta tunedTheta = null;

		for (int foldNumber = 0; foldNumber < params.NumFolds; foldNumber++) 
		{
			long startTime = System.nanoTime();
			InitialTheta = new FineTunableTheta(params.EmbeddingSize,params.EmbeddingSize,
														params.CatSize,params.DictionarySize,true);

			List<LabeledDatum<Integer,Integer>> trainingData = cv.getTrainingData(foldNumber); //,numFolds);
			List<LabeledDatum<Integer,Integer>> testData = cv.getValidationData(foldNumber);
			
			if( params.TrainModel )
			{
				RAECost = new RAECost(params.AlphaCat,params.CatSize,params.Beta,params.DictionarySize,
						params.hiddenSize,params.visibleSize, params.Lambda,InitialTheta.We,trainingData,null,f);
				Minimizer<DifferentiableFunction> minFunc = new QNMinimizer(10, params.MaxIterations);	
				double[] minTheta = minFunc.minimize(RAECost, 1e-6, InitialTheta.Theta, params.MaxIterations);
				tunedTheta = new FineTunableTheta(minTheta, params.hiddenSize, 
												params.visibleSize, params.CatSize, params.DictionarySize);
			}
			else 
			{
				System.out.println("Reading in the pre-computed RAE ...");
				
				FileInputStream fis = new FileInputStream(params.dir + "/opttheta.dat");
				ObjectInputStream ois = new ObjectInputStream(fis);
				tunedTheta = (FineTunableTheta) ois.readObject();
				ois.close();
				
				InitialTheta = new FineTunableTheta(params.EmbeddingSize, params.EmbeddingSize, 
														params.CatSize, params.DictionarySize,true);
				InitialTheta.setWe( DoubleMatrix.zeros(params.hiddenSize, params.DictionarySize) );
			}
			
			// Important step
			tunedTheta.setWe(tunedTheta.We.add(InitialTheta.We));
			tunedTheta.Dump(params.dir + "/" + params.ModelFile + params.AlphaCat + "." + params.Beta + ".rae");
			
			System.out.println("Extracting features ...");
	
			FeatureExtractor = new RAEFeatureExtractor(params.EmbeddingSize, tunedTheta, 
								params.AlphaCat, params.Beta, params.CatSize, params.DictionarySize, f);
			
			List<LabeledDatum<Double, Integer>> classifierTrainingData 
				= FeatureExtractor.extractFeaturesIntoArray(trainingData);
			
			List<LabeledDatum<Double, Integer>> classifierTestingData 
				= FeatureExtractor.extractFeaturesIntoArray(testData);
			
			SoftmaxClassifier<Double,Integer> classifier = new SoftmaxClassifier<Double,Integer>( );

			Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
			Accuracy TestAccuracy = classifier.test(classifierTestingData);
			System.out.println( "Train Accuracy :" + TrainAccuracy.toString() );
			System.out.println( "Test Accuracy :" + TestAccuracy.toString() );
			long endTime = System.nanoTime();
			long duration = endTime - startTime;
			System.out.println( "Fold " + foldNumber + " took " + duration / (1000*1000) + "ms " );
		}
	}
	
	public static DoubleMatrix ReadMatrix(String file, String var) throws IOException
	{
		MatFileReader mfr = new MatFileReader(file);
        MLArray mlArrayRetrived = mfr.getMLArray(var);
        return new DoubleMatrix(((MLDouble)mlArrayRetrived ).getArray()); 
    }	
	
}
