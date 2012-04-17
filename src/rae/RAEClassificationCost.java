package rae;

import math.*;
import org.jblas.*;

import util.*;
import classify.*;

public class RAEClassificationCost extends OnePassCost
{
	int CatSize;
	FineTunableTheta Theta;
	
	public RAEClassificationCost(int CatSize, double AlphaCat, double Beta, 
				int DictionaryLength, int HiddenSize, double[] Lambdas,
				DifferentiableMatrixFunction f, FineTunableTheta Theta){
		super(AlphaCat, Beta, DictionaryLength, HiddenSize, Lambdas, f);
		this.CatSize = CatSize;
		this.Theta = Theta;
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
	}
	
	public void Compute(LabeledDatum<Integer,Integer> Data, LabeledRAETree Tree) throws Exception
	{
		int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
		DoubleMatrix WordsEmbedded = Theta.We.getColumns(WordIndices);

		int CurrentLabel = Data.getLabel();
		int SentenceLength = WordsEmbedded.columns;

		if(SentenceLength == 1)
			return;

		Tree = Propagator.ForwardPropagate
					(Theta, WordsEmbedded, null, CurrentLabel, 
					SentenceLength, Tree);

		Propagator.BackPropagate(Tree, Theta, WordIndices);

		lock.lock();
		{
			cost += Tree.TotalScore; 
			num_nodes += Tree.TreeSize;
		}
		lock.unlock();
	}
	
	@Override
	public double getCost()
	{
		CalculateCosts(Theta);
		return cost;
	}

	@Override
	protected void CalculateCosts(Theta Param)
	{
		FineTunableTheta Theta = (FineTunableTheta) Param;
		double WNormSquared = DoubleMatrixFunctions.SquaredNorm(Theta.W1) + DoubleMatrixFunctions.SquaredNorm(Theta.W2) +
				DoubleMatrixFunctions.SquaredNorm(Theta.W3) + DoubleMatrixFunctions.SquaredNorm(Theta.W4);
		
		cost = (1.0f/num_nodes)*cost + 0.5 * LambdaW * WNormSquared
						+ 0.5 * LambdaL * DoubleMatrixFunctions.SquaredNorm(Theta.We)
						+ 0.5 * LambdaCat * DoubleMatrixFunctions.SquaredNorm(Theta.Wcat);
		
		/** WNormSquared, DoubleMatrixFunctions.SquaredNorm(Theta.We), DoubleMatrixFunctions.SquaredNorm(Theta.Wcat)); **/
		DoubleMatrix bcat0 = DoubleMatrix.zeros(CatSize,1);
		DoubleMatrix b0 = DoubleMatrix.zeros(HiddenSize,1);
		double[] WeightedGrad = (new FineTunableTheta(Theta.W1.mul(LambdaW),Theta.W2.mul(LambdaW),Theta.W3.mul(LambdaW),
		Theta.W4.mul(LambdaW),Theta.Wcat.mul(LambdaCat),Theta.We.mul(LambdaL),b0,b0,b0,bcat0)).Theta; 
		
		double[] CalcGrad = (new FineTunableTheta(Propagator.GW1,
				Propagator.GW2, Propagator.GW3, Propagator.GW4,
				Propagator.GWCat, Propagator.GWe_total, Propagator.Gb1,
				Propagator.Gb2, Propagator.Gb3, Propagator.Gbcat)).Theta;
 
		DoubleArrays.scale(CalcGrad, (1.0f/num_nodes));
		Gradient = DoubleArrays.add(CalcGrad, WeightedGrad);		
	}
}
