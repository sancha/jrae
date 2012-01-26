package rae;

import java.util.*;
import math.*;
import org.jblas.*;
import util.*;
import parallel.*;
import classify.LabeledDatum;

public class RAECostComputer {
	double AlphaCat, Beta;
	double LambdaW, LambdaL, LambdaCat;
	int CatSize, DictionaryLength, HiddenSize, NumExamples, num_nodes;
	DoubleMatrix WeOrig;
	List<LabeledDatum<Integer,Integer>> DataCell;
	RAEPropagation Propagator;
	FloatMatrix FreqOrig;
	DifferentiableMatrixFunction f;
	
	double[] CalcGrad, Gradient;
	double cost;
	
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
	}
	
	/**
	 * This method is called first by RAECost to build the Kids data
	 * It is equivalent to calling computeCostAndGradRAE([], theta1, 0 ....
	 */
	public double Compute(Theta Theta, FineTunableTheta FTheta, double[] Lambda, DoubleMatrix WeOrig) throws Exception
	{
		setLambdas(Lambda);
		cost = 0;
		num_nodes = 0;
		int CurrentDataIndex = 0;
		CalcGrad = new double[ Theta.Theta.length ];
		Propagator = new RAEPropagation(Theta,AlphaCat,Beta,HiddenSize,DictionaryLength,f);
		
		for(LabeledDatum<Integer,Integer> Data : DataCell)
		{
			int[] WordIndices = ArraysHelper.getIntArray( Data.getFeatures() );
			DoubleMatrix L = Theta.We.getColumns(WordIndices);
			DoubleMatrix WordsEmbedded = (WeOrig.getColumns(WordIndices)).addi(L);
			//FloatMatrix Freq = FreqOrig.getColumns(Data);
			
			int CurrentLabel = Data.getLabel();
			int SentenceLength = WordsEmbedded.columns;
		
			if(SentenceLength == 1)
				continue;
			
			Tree tree = Propagator.ForwardPropagate(Theta, WordsEmbedded, FreqOrig, CurrentLabel, SentenceLength);
			
			double[] Gradient = Propagator.BackPropagate(tree, Theta, WordIndices).Theta;
			
			cost += tree.TotalScore; 
            num_nodes += SentenceLength;
            DoubleArrays.addi(CalcGrad, Gradient);
			
            
			WordsEmbedded = Theta.We.getColumns(WordIndices);
			
			tree = Propagator.ForwardPropagate(FTheta, WordsEmbedded, FreqOrig, 
									CurrentLabel, SentenceLength, tree.structure);
			
			double[] Gradient = Propagator.BackPropagate(tree, Theta, WordIndices).Theta;
			
			cost += tree.TotalScore; 
            num_nodes += SentenceLength;
            DoubleArrays.addi(CalcGrad, Gradient);
			
            CurrentDataIndex++;
            
            
            
		}
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
