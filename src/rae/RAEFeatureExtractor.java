package rae;

import io.DataSet;
import io.MatProcessData;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.*;

import math.DifferentiableMatrixFunction;
import math.Norm1Tanh;

import org.jblas.*;

import util.ArraysHelper;
import util.CommandLineUtils;

import classify.*;

public class RAEFeatureExtractor {
	int HiddenSize;
	FineTunableTheta Theta;
	RAEPropagation Propagator;
	
	public RAEFeatureExtractor(int HiddenSize, FineTunableTheta Theta, double AlphaCat, double Beta, 
								int CatSize, int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this.HiddenSize = HiddenSize;
		this.Theta = Theta;
		Propagator = new RAEPropagation(AlphaCat, Beta, HiddenSize, CatSize, DictionaryLength, f);
	}
	
	public DoubleMatrix extractFeatures(List<LabeledDatum<Integer,Integer>> Data)
	{
		int numExamples = Data.size();
		DoubleMatrix features = DoubleMatrix.zeros(2*HiddenSize,numExamples);
		
		int CurrentDataIndex = 0;
		for(LabeledDatum<Integer,Integer> data : Data)
		{
			double[] feature = extractFeatures(data);
			features.putColumn(CurrentDataIndex, new DoubleMatrix(feature));
			CurrentDataIndex++;
			
			if(CurrentDataIndex % 1000 == 0)
				System.out.println("Finished extracting features for " + CurrentDataIndex + " items.");
		}
		return features;
	}
	
	public double[] extractFeatures(LabeledDatum<Integer,Integer> data)
	{
		int SentenceLength = data.getFeatures().size();
		int TreeSize = 2 * SentenceLength - 1;
		
		if(SentenceLength == 0)
			System.err.println("Zero length data");
		
		double[] feature = new double[ HiddenSize * 2 ];
		
		int[] wordIndices = ArraysHelper.getIntArray( data.getFeatures() );
		DoubleMatrix WordsEmbedded = Theta.We.getColumns(wordIndices);
		int CurrentLabel = data.getLabel();
		
		Tree tree = Propagator.ForwardPropagate(Theta, WordsEmbedded, null, CurrentLabel, SentenceLength);
		DoubleMatrix tf = new DoubleMatrix(HiddenSize,TreeSize);
		if(SentenceLength > 1)
		{
			for(int i=0; i<TreeSize; i++)
				tf.putColumn(i, tree.T[i].Features);
			tf.muli(1.0/(double)TreeSize);
			
			System.arraycopy(tree.T[ 2 * SentenceLength - 2 ].Features.data, 0, feature, 0, HiddenSize);
			System.arraycopy(tf.rowSums().data, 0, feature, HiddenSize, HiddenSize);
		}
		else
		{
			//features1(ii,:) = Tree.nodeFeatures(:,1);
	        //features2(ii,:) = Tree.nodeFeatures(:,1);
			System.arraycopy(tree.T[ 2 * SentenceLength - 2].Features.data, 0, feature, 0, HiddenSize);
			System.arraycopy(tree.T[ 2 * SentenceLength - 2].Features.data, 0, feature, HiddenSize, HiddenSize);
		}
		return feature;
	}
	
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
			PrintStream out = new PrintStream(featuresOutputFile);
			
			MatProcessData Dataset = new MatProcessData(dir);	
			int EmbeddingSize = 50; 
			double AlphaCat = 0.2, Beta = 0.5;
			DifferentiableMatrixFunction f = new Norm1Tanh(); 
			
			FileInputStream fis = new FileInputStream(dir + "/opttheta.dat");
			ObjectInputStream ois = new ObjectInputStream(fis);
			FineTunableTheta tunedTheta = (FineTunableTheta) ois.readObject();
			ois.close();
			
			RAEFeatureExtractor fe = new RAEFeatureExtractor(EmbeddingSize, tunedTheta, AlphaCat, Beta, Dataset.getCatSize(), Dataset.Vocab.size(), f);
			
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
		}
		catch(ClassNotFoundException e)
		{
			System.err.println(e.getMessage());
		}
	}

}
