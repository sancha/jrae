package rae;

import io.*;

import java.util.*;
import java.io.*;

import math.*;

import org.jblas.*;

import util.*;

import classify.*;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class Main {
	
	public static void main(final String[] args) throws Exception
	{
		
		Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);
		RAECost RAECost = null;
		// Set the parameters from argMap or use defaults;
		String dir = "data/parsed/"; //ssanjeev
		String SaveFile = "save1.dat";
		int K = 10,	MaxIterations = 80,	EmbeddingSize = 50, CatSize = 1; 
		double AlphaCat = 0.2, Beta = 0.5;
		boolean TrainModel = true;
		double[] Lambda = new double[]{1e-05, 0.0001, 1e-05, 0.01};
		DifferentiableMatrixFunction f = new Norm1Tanh(); 
		LabeledDataSet<LabeledDatum<Integer,Integer>,Integer,Integer> Dataset = null;
		RAEFeatureExtractor FeatureExtractor = null;
		StratifiedCrossValidation<LabeledDatum<Integer,Integer>,Integer,Integer> cv = null;
		
		if (argMap.containsKey("-K")) 
			K = Integer.parseInt( argMap.get("-K") );
		
		if (argMap.containsKey("-maxIterations")) 
			MaxIterations = Integer.parseInt( argMap.get("-maxIterations") );
		
		if (argMap.containsKey("-embeddingSize")) 
			EmbeddingSize = Integer.parseInt( argMap.get("-embeddingSize") );
		
		if (argMap.containsKey("-alphaCat")) 
			AlphaCat = Double.parseDouble( argMap.get("-alphaCat") );
		
		if (argMap.containsKey("-lambdaW")) 
			Lambda[0] = Double.parseDouble( argMap.get("-lambdaW") );
		
		if (argMap.containsKey("-lambdaL")) 
			Lambda[1] = Double.parseDouble( argMap.get("-lambdaL") );
		
		if (argMap.containsKey("-lambdaCat")) 
			Lambda[2] = Double.parseDouble( argMap.get("-lambdaCat") );
		
		if (argMap.containsKey("-lambdaLRAE")) 
			Lambda[3] = Double.parseDouble( argMap.get("-lambdaLRAE") );
		
		if (argMap.containsKey("-Beta")) 
			Beta = Integer.parseInt( argMap.get("-Beta") );

		if (argMap.containsKey("-trainModel")) 
			TrainModel = Boolean.parseBoolean( argMap.get("-trainModel") );		
		
		if (argMap.containsKey("-ProcessedDataDir"))
		{
			dir = argMap.get("-ProcessedDataDir");
			Dataset = new MatProcessData(dir);
		}
		else if (argMap.containsKey("-DataDir"))
		{
			dir = argMap.get("-DataDir");
			Dataset = new ParsedReviewData(dir);
		}else
			Dataset = new MatProcessData(dir);
		
		if (argMap.containsKey("-SaveFile"))
			SaveFile = argMap.get("-SaveFile");
		
		CatSize = Dataset.getCatSize()-1;
		int DictionarySize = Dataset.Vocab.size();
		int hiddenSize = EmbeddingSize, visibleSize = EmbeddingSize;
		
		LBFGSMinimizer minFunc = new LBFGSMinimizer(MaxIterations);
		FineTunableTheta InitialTheta = new FineTunableTheta(EmbeddingSize,EmbeddingSize,CatSize,DictionarySize,true);
		
		System.out.printf("%d\n%d\n%d\n%d\n",DictionarySize,hiddenSize,InitialTheta.Theta.length, InitialTheta.getThetaSize());
		
		cv = new StratifiedCrossValidation<LabeledDatum<Integer,Integer>,Integer,Integer>(K, Dataset);
		FineTunableTheta tunedTheta = null;

		for (int foldNumber = 0; foldNumber < K; foldNumber++) 
		{
			List<LabeledDatum<Integer,Integer>> trainingData = cv.getTrainingData(foldNumber); //,numFolds);
			List<LabeledDatum<Integer,Integer>> testData = cv.getValidationData(foldNumber);
			
			if( TrainModel )
			{
				RAECost = new RAECost(AlphaCat,CatSize,Beta,DictionarySize,hiddenSize,visibleSize,
												Lambda,InitialTheta.We,trainingData,null,f);
				double[] minTheta = minFunc.minimize(RAECost, InitialTheta.Theta, 1e-6);
				tunedTheta = new FineTunableTheta(minTheta, hiddenSize,visibleSize, CatSize, DictionarySize);
			}
			else 
			{
				System.out.println("Reading the mat files ...");
				
	// 			load tunedTheta from disk using serialization
	//			MatFile DataLoader = new MatFile(dir + "/jopttheta.mat"); 
	//			double[] OptTheta = DataLoader.readThetaVector("opttheta");
	//			tunedTheta = new FineTunableTheta(OptTheta, hiddenSize,visibleSize, CatSize, DictionarySize);
				
				FileInputStream fis = new FileInputStream(dir + "/opttheta.dat");
				ObjectInputStream ois = new ObjectInputStream(fis);
				tunedTheta = (FineTunableTheta) ois.readObject();
				ois.close();
				
				InitialTheta.setWe( DoubleMatrix.zeros(hiddenSize, DictionarySize) );
			}
			
			// Important step
			tunedTheta.setWe(tunedTheta.We.add(InitialTheta.We));
			tunedTheta.Dump(dir + "/" + SaveFile);
			
			System.out.println("Extracting features ...");
	
			FeatureExtractor = new RAEFeatureExtractor(EmbeddingSize, tunedTheta, AlphaCat, Beta, CatSize, DictionarySize, f);
			List<LabeledDatum<Double,Integer>> classifierTrainingData = getFeatures(FeatureExtractor, trainingData);
			List<LabeledDatum<Double,Integer>> classifierTestingData = getFeatures(FeatureExtractor, testData);
			
			SoftmaxClassifier<Double,Integer> classifier = new SoftmaxClassifier<Double,Integer>( );

			Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
			Accuracy TestAccuracy = classifier.test(classifierTestingData);
			System.out.println( "Train Accuracy :" + TrainAccuracy.toString() );
			System.out.println( "Test Accuracy :" + TestAccuracy.toString() );
		}
	}
	
	private static List<LabeledDatum<Double,Integer>> 
			getFeatures(RAEFeatureExtractor FeatureExtractor, List<LabeledDatum<Integer,Integer>> Data)
	{
		int dataItem = 0;
		List<LabeledDatum<Double,Integer>> DataFeatures = new ArrayList<LabeledDatum<Double,Integer>>( Data.size() );
		for(LabeledDatum<Integer,Integer> Datum : Data)
		{
			double[] feature = FeatureExtractor.extractFeatures(Datum);
			DataFeatures.add( new ReviewFeatures(Datum.toString(), Datum.getLabel(), dataItem, feature) );
			dataItem++;
		}
		return DataFeatures;
	}
	
	public static DoubleMatrix ReadMatrix(String file, String var) throws IOException
	{
		MatFileReader mfr = new MatFileReader(file);
        MLArray mlArrayRetrived = mfr.getMLArray(var);
        return new DoubleMatrix(((MLDouble)mlArrayRetrived ).getArray()); 
    }	
	
}
