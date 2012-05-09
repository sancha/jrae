package rae;

import math.*;
import java.util.*;
import org.jblas.*;

import parallel.Parallel;
import classify.LabeledDatum;

public class RAECost extends MemoizedDifferentiableFunction {

	double[] Lambda, FTLambda;
	double value, AlphaCat, Beta;
	DoubleMatrix WeOrig;
	int hiddenSize, visibleSize, catSize, dictionaryLength;
	DifferentiableMatrixFunction f;
	List<LabeledDatum<Integer,Integer>> DataCell;
	
	public RAECost(double AlphaCat, int CatSize, double Beta, int DictionaryLength, 
			int hiddenSize, int visibleSize, double[] Lambda, DoubleMatrix WeOrig, 
			List<LabeledDatum<Integer,Integer>> DataCell, FloatMatrix FreqOrig, DifferentiableMatrixFunction f) {
		
		evalCount = 0;
		this.f = f;
		this.Beta = Beta;
		this.hiddenSize = hiddenSize;
		this.visibleSize = visibleSize;
		this.dictionaryLength = DictionaryLength;
		this.catSize = CatSize;
		this.WeOrig = WeOrig;
		this.AlphaCat = AlphaCat;
		this.DataCell = DataCell;
		
		this.Lambda = new double[]{ Lambda[0], Lambda[1], Lambda[3]};
		this.FTLambda = new double[]{ Lambda[0], Lambda[1], Lambda[2]};
		DoubleArrays.scale(this.Lambda, AlphaCat);
		DoubleArrays.scale(this.FTLambda, 1-AlphaCat);
		
		initPrevQuery();
	}

	@Override
	public int dimension() {
		return 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength 
                + hiddenSize + 2 * visibleSize  +  catSize * hiddenSize + catSize;
	}

	@Override
	public double valueAt(double[] x)
	{
		if(!requiresEvaluation(x))
			return value;
		
		Theta Theta1 = new Theta(x,hiddenSize,visibleSize,dictionaryLength);
		FineTunableTheta Theta2 = new FineTunableTheta(x,hiddenSize,visibleSize,catSize,dictionaryLength);
		Theta2.setWe( Theta2.We.add(WeOrig) );
		
		final RAEClassificationCost classificationCost = new RAEClassificationCost(
				catSize, AlphaCat, Beta, dictionaryLength, hiddenSize, Lambda, f, Theta2);
		final RAEFeatureCost featureCost = new RAEFeatureCost(
				AlphaCat, Beta, dictionaryLength, hiddenSize, Lambda, f, WeOrig, Theta1);
	
		Parallel.For(DataCell, 
			new Parallel.Operation<LabeledDatum<Integer,Integer>>() {
				public void perform(int index, LabeledDatum<Integer,Integer> Data)
				{
					try {
						LabeledRAETree Tree = featureCost.Compute(Data);
						classificationCost.Compute(Data, Tree);					
					} catch (Exception e) {
						System.err.println(e.getMessage());
					}
				}
		});
		
		double costRAE = featureCost.getCost();
		double[] gradRAE = featureCost.getGradient().clone();
			
		double costSUP = classificationCost.getCost();
		gradient = classificationCost.getGradient();
			
		value = costRAE + costSUP;
		for(int i=0; i<gradRAE.length; i++)
			gradient[i] += gradRAE[i];
		
		System.gc();	System.gc();
		System.gc();	System.gc();
		System.gc();	System.gc();
		System.gc();	System.gc();
		
		return value;
	}
}
