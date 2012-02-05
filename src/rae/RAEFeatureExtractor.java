package rae;

import io.DataSet;
import io.MatProcessData;

import java.io.*;
import java.util.*;
import java.util.concurrent.locks.*;

import math.DifferentiableMatrixFunction;
import math.Norm1Tanh;

import org.jblas.*;

import util.ArraysHelper;
import util.CommandLineUtils;

import classify.*;
import parallel.*;

public class RAEFeatureExtractor {
	int HiddenSize;
	FineTunableTheta Theta;
	RAEPropagation Propagator;
	DoubleMatrix features;
	Lock lock;
	
	public RAEFeatureExtractor(int HiddenSize, FineTunableTheta Theta, double AlphaCat, double Beta, 
								int CatSize, int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this.HiddenSize = HiddenSize;
		this.Theta = Theta;
		Propagator = new RAEPropagation(AlphaCat, Beta, HiddenSize, CatSize, DictionaryLength, f);
		lock = new ReentrantLock();
	}
	
	public DoubleMatrix extractFeatures(List<LabeledDatum<Integer,Integer>> Data)
	{
		int numExamples = Data.size();
		features = DoubleMatrix.zeros(2*HiddenSize,numExamples);

		Parallel.For(Data, new Parallel.Operation<LabeledDatum<Integer,Integer>>(){
			@Override
			public void perform(int index, LabeledDatum<Integer, Integer> data) {
				double[] feature = extractFeatures(data);
				lock.lock();
				{
					features.putColumn(index, new DoubleMatrix(feature));				
				}
				lock.unlock();
			}
		});	
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
			tf.muli(1.0/TreeSize);
			
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
