package io;

import java.io.IOException;
import java.util.*;

import org.jblas.*;

import classify.*;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class ReadFeatureMatrices {

	public static DoubleMatrix ReadMatrix(String file, String var) throws IOException
	{
		MatFileReader mfr = new MatFileReader(file);
        MLArray mlArrayRetrived = mfr.getMLArray(var);
        return new DoubleMatrix(((MLDouble)mlArrayRetrived ).getArray()); 
    }
	
	public static void main(String[] args) throws IOException
	{
		String dir = "data/original/";
		String[] vars = {"training_instances","testing_instances","training_labels","testing_labels"};
		SoftmaxClassifier<Double,Integer> classifier = new SoftmaxClassifier<Double,Integer>( );
		
		DoubleMatrix trainFeatures = ReadMatrix(dir + vars[0] + ".mat", vars[0]);
		DoubleMatrix testFeatures = ReadMatrix(dir + vars[1] + ".mat", vars[1]);
		DoubleMatrix trainLabels = ReadMatrix(dir + vars[2] + ".mat", vars[2]);
		DoubleMatrix testLabels = ReadMatrix(dir + vars[3] + ".mat", vars[3]);
		
//		List<LabeledDatum<Double,Integer>> trainingData = new ArrayList<LabeledDatum<Double,Integer>>();
//		List<LabeledDatum<Double,Integer>> testData = new ArrayList<LabeledDatum<Double,Integer>>();
 		LabeledDataSet<LabeledDatum<Double,Integer>,Double,Integer> allData = 
 				new LabeledDataSet<LabeledDatum<Double,Integer>,Double,Integer>(10662);
		int nTrain = 0, cl = 0;
		
		for(int i=0; i<trainFeatures.rows; i++)
		{
			double[] f = (trainFeatures.getRow(i)).data;
			int l = (int) trainLabels.get(i,0);
			allData.add( new ReviewFeatures(i + " " + l,l,i,f) );
//			trainingData.add( new ReviewFeatures(i + " " + l,l,i,f) );
			nTrain++;
			cl += l;
		}	
		
		int nTest = 0, vl = 0; 
		for(int i=0; i<testFeatures.rows; i++)
		{
			double[] f = (testFeatures.getRow(i)).data;
			int l = (int) testLabels.get(i,0);
			allData.add( new ReviewFeatures((nTrain+i) + " " + l,l,nTrain+i,f) );
//			testData.add( new ReviewFeatures((nTrain+i) + " " + l,l,nTrain+i,f) );
			nTest++;
			vl += l;
		}
		
		CrossValidation<LabeledDatum<Double,Integer>,Double> cv = 
				new StratifiedCrossValidation<LabeledDatum<Double,Integer>,Double,Integer>(10, allData);
		
		int foldNumber = 0;

		List<LabeledDatum<Double,Integer>> trainingData = cv.getTrainingData(foldNumber); //,numFolds);
		List<LabeledDatum<Double,Integer>> testData = cv.getValidationData(foldNumber);
			
		System.out.println(nTrain + " " + nTest);
		System.out.println(cl + " " + vl);
		
		Accuracy TrainAccuracy = classifier.train(trainingData);
		Accuracy TestAccuracy = classifier.test(testData);
		System.out.println( "Train Accuracy :" + TrainAccuracy.toString() );
		System.out.println( "Test Accuracy :" + TestAccuracy.toString() );
	}
}
