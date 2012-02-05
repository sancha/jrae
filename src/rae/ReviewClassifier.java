package rae;

import io.DataSet;
import io.LabeledDataSet;
import io.MatProcessData;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.Map;

import util.CommandLineUtils;
import util.Counter;

import math.DifferentiableMatrixFunction;
import math.Norm1Tanh;

import classify.Accuracy;
import classify.LabeledDatum;
import classify.ReviewDatum;
import classify.ReviewFeatures;
import classify.SoftmaxClassifier;


public class ReviewClassifier 
{
	public static void main(String[] args)
	{
		String dir = "data/parsed";
		String inputDataFile = null, featuresOutputFile = null;
		MatProcessData Dataset = null;
		LabeledDataSet<LabeledDatum<Double,Integer>,Double,Integer> DataFeatures = null;
		
		Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);
		
		if (argMap.containsKey("-input") && argMap.containsKey("-output")) 
		{
			inputDataFile = argMap.get("-input");
			featuresOutputFile = argMap.get("-output");
		}
		else
		{
			System.err.println("Both -input and -output parameters are required are required.");
			System.err.println("Please input a file path with text whose features you want to extract in the -input parameter,");
			System.err.println("and a file path with text where you want the features dumped.");
			return;
		}
		
		try{
			PrintStream out = new PrintStream(featuresOutputFile);
			
			Dataset = new MatProcessData(dir);	
			int EmbeddingSize = 50; 
			double AlphaCat = 0.2, Beta = 0.5;
			DifferentiableMatrixFunction f = new Norm1Tanh(); 
			
			System.out.println("loading the pre-learnt RAE ...");
			FileInputStream fis = new FileInputStream(dir + "/save1.dat");
			ObjectInputStream ois = new ObjectInputStream(fis);
			FineTunableTheta tunedTheta = (FineTunableTheta) ois.readObject();
			ois.close();
			
			System.out.println("Extracting features ...");
			RAEFeatureExtractor fe = new RAEFeatureExtractor(EmbeddingSize, tunedTheta, AlphaCat, Beta, Dataset.getCatSize(), Dataset.Vocab.size(), f);
			DataFeatures = new LabeledDataSet<LabeledDatum<Double,Integer>,Double,Integer>( Dataset.Data.size() );
			int dataIndex = 0;
			for(LabeledDatum<Integer,Integer> Datum : Dataset.Data)
			{
				double[] feature = fe.extractFeatures(Datum);
				DataFeatures.add( new ReviewFeatures(Datum.toString(), Datum.getLabel(), dataIndex, feature) );
				dataIndex++;
			}
		
			SoftmaxClassifier<Double,Integer> classifier = new SoftmaxClassifier<Double,Integer>();
			Accuracy trainingAccuracy = classifier.train(DataFeatures.Data);
			System.out.println("Training Accuracy : " + trainingAccuracy);
			
	        String strLine;
	        FileInputStream fstream = new FileInputStream(inputDataFile);
	        DataInputStream in = new DataInputStream(fstream);
	        BufferedReader br = new BufferedReader(new InputStreamReader(in));

	        int itemNo = 0;
	        while ((strLine = br.readLine()) != null)
	        {
	        	strLine = strLine.trim();
	        	String[] words = strLine.split(" ");
	        	int[] indices = new int[ words.length ];
	        	for(int i=0; i<words.length; i++)
	        	{
	        		if( Dataset.Vocab.contains(words[i].toLowerCase()))
	        			indices[i] = Dataset.getWordIndex(words[i].toLowerCase());
	        		else
	        			indices[i] = Dataset.getWordIndex(DataSet.UNK);
	        	}
	        	ReviewDatum d = new ReviewDatum(strLine, -1, itemNo, indices);
	        	double[] features = fe.extractFeatures(d);
	        	
	        	ReviewFeatures rf = new ReviewFeatures(strLine, -1, itemNo, features); 
	        	Counter<Integer> prob = classifier.getProbabilities(rf);
	        	
	        	for(Integer l : prob.keySet() )
	        		out.printf("%d : %.8f, ", l.intValue(), prob.getCount(l));
	        	
	        	out.println();
	        	itemNo++;
	        }
	        in.close();
	    }
		catch(IOException e)
		{
			System.err.println(e.getMessage());
		}
		catch(ClassNotFoundException e)
		{
			System.err.println(e.getMessage());
		}
	}
	
}
