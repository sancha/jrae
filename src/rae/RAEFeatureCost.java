package rae;

import math.*;
import org.jblas.*;

import util.*;
import classify.*;
public class RAEFeatureCost extends OnePassCost
{
	DoubleMatrix WeOrig;
	Theta Theta;
	
	public RAEFeatureCost(double AlphaCat, double Beta, 
				int DictionaryLength, int HiddenSize, double[] Lambda,
				DifferentiableMatrixFunction f, DoubleMatrix WeOrig, Theta Theta){
		super(AlphaCat, Beta, DictionaryLength, HiddenSize, Lambda, f);
		this.WeOrig = WeOrig;
		this.Theta = Theta;
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
	}
	
	public LabeledRAETree Compute(LabeledDatum<Integer,Integer> Data) 
		throws Exception
	{
		int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
		DoubleMatrix L = Theta.We.getColumns(WordIndices);
		DoubleMatrix WordsEmbedded = (WeOrig.getColumns(WordIndices)).addi(L);
		
		int CurrentLabel = Data.getLabel();
		int SentenceLength = WordsEmbedded.columns;
		
		if(SentenceLength == 1) 
			return null;
		
		LabeledRAETree Tree = Propagator.ForwardPropagate
					(Theta, WordsEmbedded, null, CurrentLabel, SentenceLength);
		Propagator.BackPropagate(Tree, Theta, WordIndices);
		
		lock.lock();
		{
			cost += Tree.TotalScore; 
	        num_nodes += SentenceLength - 1;
        }
		lock.unlock();
		return Tree;
	}
	
	@Override
	public double getCost()
	{
		CalculateCosts(Theta);
		return cost;
	}
	
	protected void CalculateCosts(Theta Theta)
	{
		double WNormSquared = DoubleMatrixFunctions.SquaredNorm(Theta.W1) + DoubleMatrixFunctions.SquaredNorm(Theta.W2) +
				DoubleMatrixFunctions.SquaredNorm(Theta.W3) + DoubleMatrixFunctions.SquaredNorm(Theta.W4);

		cost = (1.0f/num_nodes)*cost + 0.5 * LambdaW * WNormSquared 
							 + 0.5 * LambdaL * DoubleMatrixFunctions.SquaredNorm(Theta.We);
		
		double[] CalcGrad = (new Theta(Propagator.GW1, Propagator.GW2,
				Propagator.GW3, Propagator.GW4, Propagator.GWe_total,
				Propagator.Gb1, Propagator.Gb2, Propagator.Gb3)).Theta;
		
		DoubleMatrix b0 = DoubleMatrix.zeros(HiddenSize,1);
		double[] WeightedGrad = (new Theta(Theta.W1.mul(LambdaW),Theta.W2.mul(LambdaW),Theta.W3.mul(LambdaW),
		Theta.W4.mul(LambdaW),Theta.We.mul(LambdaL),b0,b0,b0)).Theta; 
		
		DoubleArrays.scale(CalcGrad, (1.0f/num_nodes));
		Gradient = DoubleArrays.add(CalcGrad, WeightedGrad);		
	}
}
