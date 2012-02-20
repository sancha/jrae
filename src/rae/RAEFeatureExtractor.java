package rae;

import java.util.*;
import java.util.concurrent.locks.*;
import math.DifferentiableMatrixFunction;
import org.jblas.*;
import util.ArraysHelper;
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
		
//		int min = 1000000, max = -1;
//		for (int i=0; i<wordIndices.length; i++)
//		{
//			min = Math.min(wordIndices[i], min);
//			max = Math.max(wordIndices[i], max);
//		}
//		//if(min < 0 || max > Theta.We.columns)
//			System.err.println(Theta.We.columns + " " + min + " " + max);
		
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
}
