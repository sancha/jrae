package rae;

import java.util.*;
import java.util.concurrent.locks.*;

import math.*;
import org.jblas.*;
import util.*;
import classify.*;
import parallel.*;

public class RAECostComputer
{
	double AlphaCat, Beta;
	double LambdaW, LambdaL, LambdaCat;
	int CatSize, DictionaryLength, HiddenSize, NumExamples, num_nodes;
	DoubleMatrix WeOrig;
	List<LabeledDatum<Integer,Integer>> DataCell;
	RAEPropagation Propagator;
	FloatMatrix FreqOrig;
	DifferentiableMatrixFunction f;
	double[] Gradient;
	double cost;
	Lock lock;
	LabeledRAETree[] AllTrees;
	
	public RAECostComputer(int CatSize, double AlphaCat, double Beta, int DictionaryLength, int HiddenSize
				, List<LabeledDatum<Integer,Integer>> DataCell, FloatMatrix FreqOrig, DifferentiableMatrixFunction f){
		NumExamples = DataCell.size();
		this.Beta = Beta;
		this.CatSize = CatSize;
		this.DataCell = DataCell;
		this.AlphaCat = AlphaCat;
		this.DictionaryLength = DictionaryLength;
		this.HiddenSize = HiddenSize;
		this.FreqOrig = FreqOrig;
		this.f = f;
		num_nodes = 0;
		AllTrees = new LabeledRAETree[NumExamples];
		lock = new ReentrantLock();
	}
	
	/**
	 * This method is called by RAECost after building the Kids data
	 * It is equivalent to calling computeCostAndGradRAE(allKids, theta2, 1 ....
	 */
	public double Compute(final FineTunableTheta Theta, final double[] Lambda) throws Exception
	{
		setLambdas(Lambda);
		
		cost = 0;
		num_nodes = 0;
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
		
		Propagator = ThreadPool.mapReduce (DataCell, Propagator, 
				new ThreadPool.Operation<RAEPropagation, LabeledDatum<Integer,Integer>>() {
					public void perform(RAEPropagation locPropagator, int index, 
							LabeledDatum<Integer,Integer> Data)
					{
						int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
						DoubleMatrix WordsEmbedded = Theta.We.getColumns(WordIndices);
						//FloatMatrix Freq = FreqOrig.getColumns(Data);
				
						int CurrentLabel = Data.getLabel();
						int SentenceLength = WordsEmbedded.columns;
				
						if(SentenceLength == 1)
							return;
				
						LabeledRAETree tree = locPropagator.ForwardPropagate
									(Theta, WordsEmbedded, FreqOrig, CurrentLabel, 
									SentenceLength, AllTrees[index]);
				
						locPropagator.BackPropagate(tree, Theta, WordIndices);
				
						lock.lock();
						{
							cost += tree.TotalScore; 
							num_nodes += tree.TreeSize;
						}
						lock.unlock();
						
						tree = null;
						WordIndices = null;
						WordsEmbedded = null;
						
					}
			});
		
		CalculateFineCosts(Theta);
		
		return cost;	
	}
	
	/**
	 * This method is called first by RAECost to build the Kids data
	 * It is equivalent to calling computeCostAndGradRAE([], theta1, 0 ....
	 */
	public double Compute(final Theta Theta, final double[] Lambda, final DoubleMatrix WeOrig) throws Exception
	{
		setLambdas(Lambda);
		
		cost = 0;
		num_nodes = 0;
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
		
		Propagator = ThreadPool.mapReduce (DataCell, Propagator, 
				new ThreadPool.Operation<RAEPropagation, LabeledDatum<Integer,Integer>>() {
					public void perform(RAEPropagation locPropagator, int index, 
							LabeledDatum<Integer,Integer> Data)
					{
						int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
						DoubleMatrix L = Theta.We.getColumns(WordIndices);
						DoubleMatrix WordsEmbedded = (WeOrig.getColumns(WordIndices)).addi(L);
						
						int CurrentLabel = Data.getLabel();
						int SentenceLength = WordsEmbedded.columns;
						
						if(SentenceLength == 1) 
							return;
						
						LabeledRAETree tree = locPropagator.ForwardPropagate
									(Theta, WordsEmbedded, FreqOrig, CurrentLabel, SentenceLength);
						locPropagator.BackPropagate(tree, Theta, WordIndices);
						
						lock.lock();
						{
							AllTrees[ index ] = tree;
							cost += tree.TotalScore; 
					        num_nodes += SentenceLength;
				        }
						lock.unlock();
					}
			});
		
		num_nodes -= NumExamples;
		CalculateCosts(Theta);
		return cost;
	}
	
	public double[] getGradient()
	{
		return Gradient;
	}
	
	public double getCost()
	{
		return cost;
	}
	
	public LabeledRAETree[] getKids()
	{ 
		return AllTrees;
	} 
	
	private void CalculateCosts(Theta Theta)
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

	private void CalculateFineCosts(FineTunableTheta Theta)
	{
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

	private void setLambdas(double[] Lambda)
	{
		LambdaW = Lambda[0];	
		LambdaL = Lambda[1]; 	
		LambdaCat = Lambda[2];		
	}
}
