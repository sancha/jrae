package classify;

import math.*;

import org.jblas.*;

import util.*;

public class SoftmaxCost extends MemoizedDifferentiableFunction
{
	DoubleMatrix Features, Labels;
	DifferentiableMatrixFunction Activation;
	ClassifierTheta Gradient;
	double Lambda;
	int CatSize, numDataItems, FeatureLength;
	
	public SoftmaxCost(DoubleMatrix Features, int[] Labels, int CatSize, double Lambda)
	{
		this.CatSize = CatSize;
		FeatureLength = Features.rows;
		numDataItems = Features.columns;
		assert numDataItems == Labels.length;
		
		this.Labels = DoubleMatrix.zeros(CatSize,numDataItems);
		for(int i=0; i<Labels.length; i++)
		{
			if( Labels[i] < 0 || Labels[i] > CatSize )
				System.err.println("Bad Data : " + Labels[i] + " | " + i);
			else
				this.Labels.put(Labels[i],i,1);
		}
		this.Features = Features;
		this.Lambda = Lambda;
		
		Activation = CatSize > 1 ? new Softmax() :new Sigmoid();
		Gradient = null;
		initPrevQuery();
	}
	
	public SoftmaxCost(DoubleMatrix Features, DoubleMatrix Labels, double Lambda)
	{
		this.Features = Features;
		this.Lambda = Lambda;
		this.Labels = Labels;
		CatSize = Labels.rows;
		FeatureLength = Features.rows;
		numDataItems = Features.columns;
		Activation = CatSize > 1 ? new Softmax() :new Sigmoid();
		Gradient = null;
		initPrevQuery();
	}
	
	public SoftmaxCost (int CatSize, int FeatureLength, double Lambda)
	{
		Features = null;
		Labels = null;
		Gradient = null;
		numDataItems = -1;
		
		this.Lambda = Lambda;
		this.CatSize = CatSize;
		this.FeatureLength = FeatureLength;
		
		Activation = CatSize > 1 ? new Softmax() :new Sigmoid();
		
		initPrevQuery();
	}
	
	@Override
	public int dimension() 
	{
		return (CatSize - 1) * (FeatureLength + 1);
	}
	
	public DoubleMatrix getPredictions (ClassifierTheta Theta, DoubleMatrix Features)
	{
		int numDataItems = Features.columns;
		DoubleMatrix Input = ((Theta.W.transpose()).mmul(Features)).addColumnVector(Theta.b);
		Input = DoubleMatrix.concatVertically(Input, DoubleMatrix.zeros(1,numDataItems));
		return Activation.valueAt(Input);		
	}
	
	private double getNetLogLoss (DoubleMatrix Prediction, DoubleMatrix Labels)
	{
		return -MatrixFunctions.log((Labels.mul(Prediction)).columnSums()).sum();
	}

	@Override
	public double valueAt(double[] x) 
	{
		if( !requiresEvaluation(x) )
			return value;
		
		int[] requiredRows = ArraysHelper.makeArray(0, CatSize-2);
		ClassifierTheta Theta = new ClassifierTheta(x,FeatureLength,CatSize);
		
		DoubleMatrix Input = ((Theta.W.transpose()).mmul(Features)).addColumnVector(Theta.b);
		Input = DoubleMatrix.concatVertically(Input, DoubleMatrix.zeros(1,numDataItems));
		DoubleMatrix Prediction = Activation.valueAt(Input);
		
		double MeanTerm = 1.0 / (double) numDataItems;
		double Cost = getNetLogLoss (Prediction, Labels) * MeanTerm; 
		double RegularisationTerm = 0.5 * Lambda * DoubleMatrixFunctions.SquaredNorm(Theta.W);
		
		DoubleMatrix Diff = Prediction.sub(Labels).muli(MeanTerm);
	    DoubleMatrix Delta = Features.mmul(Diff.transpose());
	
	    DoubleMatrix gradW = Delta.getColumns(requiredRows);
	    DoubleMatrix gradb = ((Diff.rowSums()).getRows(requiredRows));
	    
	    if (gradW.rows != Theta.W.rows)
	    	System.err.println ("W FAIL 1");
	    
	    if (gradW.columns != Theta.W.columns)
	    	System.err.println ("W FAIL 2");
	    
	    if (gradb.rows != Theta.b.rows || gradb.columns != Theta.b.columns)
	    	System.err.println ("b FAIL");
	    
	    //Regularizing. Bias does not have one.
	    gradW = gradW.addi(Theta.W.mul(Lambda));
	    
	    Gradient = new ClassifierTheta(gradW,gradb);
	    value = Cost + RegularisationTerm;
	    gradient = Gradient.Theta;
		return value; 
	}
}