package rae;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import math.DifferentiableMatrixFunction;

public abstract class OnePassCost {
	double AlphaCat, Beta;
	double LambdaW, LambdaL, LambdaCat;
	int DictionaryLength, HiddenSize, num_nodes;
	
	RAEPropagation Propagator;
	DifferentiableMatrixFunction f;
	double[] Gradient;
	double cost;
	Lock lock;
	Theta Theta;
	
	public OnePassCost(double AlphaCat, double Beta, 
				int DictionaryLength, int HiddenSize, double[] Lambda, 
				DifferentiableMatrixFunction f){
		setLambdas(Lambda);
		this.Beta = Beta;
		this.AlphaCat = AlphaCat;
		this.DictionaryLength = DictionaryLength;
		this.HiddenSize = HiddenSize;
		this.f = f;
		num_nodes = 0;
		cost = 0;
		lock = new ReentrantLock();
	}
	
	protected abstract void CalculateCosts(Theta Theta);
	
	public double[] getGradient()
	{
		return Gradient;
	}
	
	public abstract double getCost();
	
	protected void setLambdas(double[] Lambda)
	{
		LambdaW = Lambda[0];	
		LambdaL = Lambda[1]; 	
		LambdaCat = Lambda[2];		
	}
}
