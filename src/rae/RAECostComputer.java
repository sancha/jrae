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
	Lock lock;
	
	double[] CalcGrad, Gradient;
	double cost;
	Structure[] AllKids;
	
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
		AllKids = new Structure[NumExamples];
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
		CalcGrad = DoubleArrays.constantArray(0,Theta.Theta.length);
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
		
		Parallel.For(DataCell, new Parallel.Operation<LabeledDatum<Integer,Integer>>() {
			public void perform(int index, LabeledDatum<Integer,Integer> Data)
			{
				int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
				DoubleMatrix WordsEmbedded = Theta.We.getColumns(WordIndices);
				//FloatMatrix Freq = FreqOrig.getColumns(Data);
				
				int CurrentLabel = Data.getLabel();
				int SentenceLength = WordsEmbedded.columns;
				
				if(SentenceLength == 1)
					return;
				
				
				Tree tree = Propagator.ForwardPropagate(Theta, WordsEmbedded, FreqOrig, 
										CurrentLabel, SentenceLength, AllKids[index]);
				
				double[] Gradient = Propagator.BackPropagate(tree, Theta, WordIndices).Theta;
				
				lock.lock();
				{
					cost += tree.TotalScore; 
					num_nodes++;
					DoubleArrays.addi(CalcGrad, Gradient);
				}
				lock.unlock();
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
		CalcGrad = DoubleArrays.constantArray(0,Theta.Theta.length);
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
		
		Parallel.For(DataCell, new Parallel.Operation<LabeledDatum<Integer,Integer>>() {
			public void perform(int index, LabeledDatum<Integer,Integer> Data)
			{
				int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
				DoubleMatrix L = Theta.We.getColumns(WordIndices);
				DoubleMatrix WordsEmbedded = (WeOrig.getColumns(WordIndices)).addi(L);
				
				int CurrentLabel = Data.getLabel();
				int SentenceLength = WordsEmbedded.columns;
				
				if(SentenceLength == 1)
					return;
				
				Tree tree = Propagator.ForwardPropagate(Theta, WordsEmbedded, FreqOrig, CurrentLabel, SentenceLength);
				AllKids[ index ] = tree.structure;
				
				double[] Gradient = Propagator.BackPropagate(tree, Theta, WordIndices).Theta;
				
				lock.lock();
				{	
					cost += tree.TotalScore; 
			        num_nodes += SentenceLength;
			        DoubleArrays.addi(CalcGrad, Gradient);
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
	
	public Structure[] getKids()
	{ 
		return AllKids;
	} 
	
	private void CalculateCosts(Theta Theta)
	{
		double WNormSquared = DoubleMatrixFunctions.SquaredNorm(Theta.W1) + DoubleMatrixFunctions.SquaredNorm(Theta.W2) +
				DoubleMatrixFunctions.SquaredNorm(Theta.W3) + DoubleMatrixFunctions.SquaredNorm(Theta.W4);

		cost = (1.0f/num_nodes)*cost + 0.5 * LambdaW * WNormSquared 
							 + 0.5 * LambdaL * DoubleMatrixFunctions.SquaredNorm(Theta.We);
		
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
