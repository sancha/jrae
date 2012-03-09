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
	int CatSize, FeatureLength;
	int[] requiredEntries; 
	
	public SoftmaxCost(DoubleMatrix Features, int[] Labels, int CatSize, double Lambda)
	{
		this.CatSize = CatSize;
		FeatureLength = Features.rows;
		assert Features.columns == Labels.length;
		
		this.Labels = getLabelsRepresentation(Labels,Features.columns);
		this.Features = Features;
		this.Lambda = Lambda;
		requiredEntries = ArraysHelper.makeArray(0, CatSize-2);
		
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
		Activation = CatSize > 1 ? new Softmax() :new Sigmoid();
		requiredEntries = ArraysHelper.makeArray(0, CatSize-2);
		Gradient = null;
		initPrevQuery();
	}
	
	public SoftmaxCost (int CatSize, int FeatureLength, double Lambda)
	{
		Features = null;
		Labels = null;
		Gradient = null;
		
		this.Lambda = Lambda;
		this.CatSize = CatSize;
		this.FeatureLength = FeatureLength;
		requiredEntries = ArraysHelper.makeArray(0, CatSize-2);
		
		Activation = CatSize > 1 ? new Softmax() :new Sigmoid();
		initPrevQuery();
	}
	
	@Override
	public int dimension() 
	{
		return (CatSize - 1) * (FeatureLength + 1);
	}

	@Override
	public double valueAt(double[] x) 
	{
		if( !requiresEvaluation(x) )
			return value;
		
		int numDataItems = Features.columns;
		ClassifierTheta Theta = new ClassifierTheta(x,FeatureLength,CatSize);
		
		DoubleMatrix Input = ((Theta.getW().transpose()).mmul(Features)).addColumnVector(Theta.b);
		Input = DoubleMatrix.concatVertically(Input, DoubleMatrix.zeros(1,numDataItems));
		DoubleMatrix Prediction = Activation.valueAt(Input);
		
		double MeanTerm = 1.0 / (double) numDataItems;
		double Cost = getLogLoss (Prediction, Labels).sum() * MeanTerm; 
		double RegularisationTerm = 0.5 * Lambda * DoubleMatrixFunctions.SquaredNorm(Theta.getW());
		
		DoubleMatrix Diff = Prediction.sub(Labels).muli(MeanTerm);
	    DoubleMatrix Delta = Features.mmul(Diff.transpose());
	
	    DoubleMatrix gradW = Delta.getColumns(requiredEntries);
	    DoubleMatrix gradb = ((Diff.rowSums()).getRows(requiredEntries));
	    
	    if (gradW.rows != Theta.getW().rows)
	    	System.err.println ("W FAIL 1");
	    
	    if (gradW.columns != Theta.getW().columns)
	    	System.err.println ("W FAIL 2");
	    
	    if (gradb.rows != Theta.b.rows || gradb.columns != Theta.b.columns)
	    	System.err.println ("b FAIL");
	    
	    //Regularizing. Bias does not have one.
	    gradW = gradW.addi(Theta.getW().mul(Lambda));
	    
	    Gradient = new ClassifierTheta(gradW,gradb);
	    value = Cost + RegularisationTerm;
	    gradient = Gradient.Theta;
		return value; 
	}
	
	/**
	 * Finds the predicted labels for input data as parameterized by ClassifierTheta
	 * @param Theta Classifier parameters
	 * @param Features Input Data of dimensions featureLenght by numDataItems
	 * @return A Prediction matrix of numCategories by numDataItems Matrix of predictions
	 */
	public DoubleMatrix getPredictions (ClassifierTheta Theta, DoubleMatrix Features)
	{
		int numDataItems = Features.columns;
		DoubleMatrix Input = ((Theta.getW().transpose()).mmul(Features)).addColumnVector(Theta.b);
		Input = DoubleMatrix.concatVertically(Input, DoubleMatrix.zeros(1,numDataItems));
		return Activation.valueAt(Input);		
	}
		
	/**
	 * Calculates the classifier's loss metric for each data item.
	 * @param classifierParams Classifier parameters
	 * @param Features Input Data of dimensions featureLenght by numDataItems
	 * @param Labels A vector of numDataItems labels each {0,1,...,K-1}
	 * @return Softmax loss / the negative-log-loss as a row vector with numDataItems entries
	 */
	public DoubleMatrix getLogLoss (ClassifierTheta classifierParams, 
									DoubleMatrix Features, int[] Labels)
	{
		int numDataItems = Labels.length;
		assert CatSize == classifierParams.CatSize+1;
		this.Labels = getLabelsRepresentation(Labels,numDataItems);
		DoubleMatrix Predictions = getPredictions(classifierParams, Features);
		return getLogLoss(Predictions, this.Labels);
	}
	
	public double[] derivativeFor (ClassifierTheta Theta, DoubleMatrix dataItem, int[] Label)
	{
		if (dataItem.columns > 1 || Label.length > 1)
			System.err.println ("Data item is malformed. Please pass a column vector");
		DoubleMatrix Labels = getLabelsRepresentation(Label,1);
		DoubleMatrix Input = ((Theta.getW().transpose()).mmul(dataItem)).addColumnVector(Theta.b);
		Input = DoubleMatrix.concatVertically(Input, DoubleMatrix.zeros(1,1));
		DoubleMatrix Prediction = Activation.valueAt(Input);
		DoubleMatrix Diff = Prediction.sub(Labels);
	    DoubleMatrix Delta = Features.mmul(Diff.transpose());
	
	    DoubleMatrix gradW = Delta.getColumns(requiredEntries);
	    DoubleMatrix gradb = ((Diff.rowSums()).getRows(requiredEntries));
	    
	    //Regularizing. Bias does not have one.
	    gradW = gradW.addi(Theta.getW().mul(Lambda));
	    
	    return new ClassifierTheta(gradW, gradb).Theta;
	}
	
	private DoubleMatrix getLogLoss (DoubleMatrix Prediction, DoubleMatrix Labels)
	{
		return MatrixFunctions.log((Labels.mul(Prediction)).columnSums()).muli(-1);
	}
	
	private DoubleMatrix getLabelsRepresentation (int[] Labels, int numDataItems)
	{
		DoubleMatrix LabelRepresentation = DoubleMatrix.zeros(CatSize,numDataItems);
		for(int i=0; i<numDataItems; i++)
		{
			if( Labels[i] < 0 || Labels[i] > CatSize )
				System.err.println("Bad Data : " + Labels[i] + " | " + i);
			else
				LabelRepresentation.put(Labels[i],i,1);
		}
		return LabelRepresentation;
	}
}