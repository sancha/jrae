package rae;

import java.util.*;
import java.util.concurrent.locks.*;
import math.DifferentiableMatrixFunction;
import org.jblas.*;
import util.ArraysHelper;
import classify.*;
import parallel.*;

public class RAEFeatureExtractor {
	int HiddenSize, CatSize, DictionaryLength;
	double AlphaCat, Beta;
	FineTunableTheta Theta;
	RAEPropagation Propagator;
	DoubleMatrix features;
	Lock lock;
	DifferentiableMatrixFunction f;
	
	public RAEFeatureExtractor(int HiddenSize, FineTunableTheta Theta, double AlphaCat, double Beta, 
								int CatSize, int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this.HiddenSize = HiddenSize;
		this.Theta = Theta;
		this.AlphaCat = AlphaCat; 
		this.Beta = Beta;
		this.HiddenSize = HiddenSize;
		this.CatSize = CatSize;
		this.DictionaryLength = DictionaryLength;
		this.f = f;
		lock = new ReentrantLock();
		Propagator = new RAEPropagation(AlphaCat, Beta, HiddenSize, CatSize, DictionaryLength, f);
	}
	
	public List<LabeledDatum<Double,Integer>>
	extractFeaturesIntoArray(List<LabeledRAETree> trees)
	{	
		int numExamples = trees.size();
		final LabeledDatum<Double,Integer>[] DataFeatures = new ReviewFeatures[numExamples];
		
		Parallel.For(trees, new Parallel.Operation<LabeledRAETree>(){
			@Override
			public void perform(int index, LabeledRAETree tree) {
				double[] feature = tree.getFeaturesVector();
				lock.lock();
				{
					ReviewFeatures r = 
						new ReviewFeatures (null, tree.getLabel(), index, feature);
					DataFeatures[index] = r;
				}
				lock.unlock();
			}
		});	
		return Arrays.asList(DataFeatures);
	}
	
	public DoubleMatrix extractFeatures(List<LabeledDatum<Integer,Integer>> Data)
	{
		int numExamples = Data.size();
		features = DoubleMatrix.zeros(2*HiddenSize,numExamples);		
		ThreadPool.map (Data, Propagator, 
			new ThreadPool.Operation<RAEPropagation, LabeledDatum<Integer,Integer>>() {
				public void perform(RAEPropagation locPropagator, int index, 
						LabeledDatum<Integer,Integer> data)		
				{
					double[] feature = extractFeatures(locPropagator, data);
					lock.lock();
					{
						features.putColumn(index, new DoubleMatrix(feature));				
					}
					lock.unlock();
				}
		});		
		return features;
	}
	
	public double[] extractFeatures (LabeledDatum<Integer,Integer> Data)
	{
		return getRAETree (Propagator, Data).getFeaturesVector();
	}	
	
	public double[] extractFeatures (RAEPropagation Propagator, LabeledDatum<Integer,Integer> Data)
	{
		return getRAETree (Propagator, Data).getFeaturesVector();
	}
		
	public List<LabeledRAETree> getRAETrees(List<LabeledDatum<Integer,Integer>> Data)
	{
		int numExamples = Data.size();
		final LabeledRAETree[] ExtractedTrees = new LabeledRAETree[numExamples];

		ThreadPool.map (Data, Propagator, 
			new ThreadPool.Operation<RAEPropagation, LabeledDatum<Integer,Integer>>() {
				public void perform(RAEPropagation locPropagator, int index, 
						LabeledDatum<Integer,Integer> data)		
				{
					LabeledRAETree tree = getRAETree(locPropagator, data);
					lock.lock();
					{
						ExtractedTrees[index] = tree;				
					}
					lock.unlock();
				}
		});
		return Arrays.asList(ExtractedTrees);
	}
	
	public LabeledRAETree getRAETree(RAEPropagation Propagator, LabeledDatum<Integer,Integer> data)
	{
		int SentenceLength = data.getFeatures().size();
		
		if(SentenceLength == 0)
			System.err.println("Zero length data");
		
		int[] wordIndices = ArraysHelper.getIntArray( data.getFeatures() );
				
		DoubleMatrix WordsEmbedded = Theta.We.getColumns(wordIndices);
		int CurrentLabel = data.getLabel();
		
		LabeledRAETree tree = Propagator
			.ForwardPropagate(Theta, WordsEmbedded, null, CurrentLabel, SentenceLength);
		
		tree = Propagator
			.ForwardPropagate(Theta, WordsEmbedded, null, CurrentLabel, SentenceLength, tree);
		
		return tree;
	}	
}
