package main;

import io.DataSet;
import io.MatProcessData;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.Map;

import math.DifferentiableMatrixFunction;
import math.Norm1Tanh;
import rae.FineTunableTheta;
import rae.RAEFeatureExtractor;
import util.CommandLineUtils;
import classify.ReviewDatum;

public class MovieReviewFeatureExtractor 
{
	public static void main(String[] args)
	{
		String dir = "data/parsed";
		String inputDataFile = null, featuresOutputFile = null;
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
			MatProcessData Dataset = new MatProcessData(dir);	
			int EmbeddingSize = 50; 
			double AlphaCat = 0.2, Beta = 0.5;
			DifferentiableMatrixFunction f = new Norm1Tanh(); 
			
			FileInputStream fis = new FileInputStream(dir + "/save1.dat");
			ObjectInputStream ois = new ObjectInputStream(fis);
			FineTunableTheta tunedTheta = (FineTunableTheta) ois.readObject();
			ois.close();
			
			RAEFeatureExtractor fe = new RAEFeatureExtractor(EmbeddingSize, tunedTheta, AlphaCat, Beta, Dataset.getCatSize(), Dataset.Vocab.size(), f);
			
	        String strLine;
	        FileInputStream fstream = new FileInputStream(inputDataFile);
	        DataInputStream in = new DataInputStream(fstream);
	        BufferedReader br = new BufferedReader(new InputStreamReader(in));

	        PrintStream out = new PrintStream(featuresOutputFile);
	        
	        int itemNo = 0;
	        while ((strLine = br.readLine()) != null)
	        {
	        	strLine = strLine.trim();
	        	String[] words = strLine.split(" ");
	        	int[] indices = new int[ words.length ];
	        	for(int i=0; i<words.length; i++)
	        	{
	        		if( Dataset.Vocab.contains(words[i]))
	        			indices[i] = Dataset.getWordIndex(words[i]);
	        		else
	        			indices[i] = Dataset.getWordIndex(DataSet.UNK);
	        	}
	        	ReviewDatum d = new ReviewDatum(strLine, -1, itemNo, indices);
	        	double[] features = fe.extractFeatures(d);
	        	
	        	for(int i=0; i<features.length; i++)
	        		out.printf("%.8f ", features[i]);
	        	
	        	out.println();
	        	itemNo++;
	        }
	        in.close();
	    }
		catch(IOException e)
		{
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		catch(ClassNotFoundException e)
		{
			System.err.println(e.getMessage());
		}
	}
}
