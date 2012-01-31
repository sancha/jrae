package rae;

import math.*;
import java.util.*;
import org.jblas.*;
import classify.LabeledDatum;

public class RAECost implements DifferentiableFunction {

	RAECostComputer Computer;
	double[] gradient, Lambda, FTLambda;
	double value, AlphaCat;
	DoubleMatrix WeOrig;
	int hiddenSize, visibleSize, catSize, dictionaryLength;
	
	public RAECost(double AlphaCat, int CatSize, double Beta, int DictionaryLength, 
			int hiddenSize, int visibleSize, double[] Lambda, DoubleMatrix WeOrig, 
			List<LabeledDatum<Integer,Integer>> DataCell, FloatMatrix FreqOrig, DifferentiableMatrixFunction f) {
		
		Computer = new RAECostComputer(CatSize, AlphaCat, Beta, DictionaryLength, hiddenSize, DataCell, FreqOrig, f);

		this.hiddenSize = hiddenSize;
		this.visibleSize = visibleSize;
		this.dictionaryLength = DictionaryLength;
		this.catSize = CatSize;
		this.WeOrig = WeOrig;
		this.AlphaCat = AlphaCat;
		
		this.Lambda = new double[]{ Lambda[0], Lambda[1], Lambda[3]};
		this.FTLambda = new double[]{ Lambda[0], Lambda[1], Lambda[2]};
		DoubleArrays.scale(this.Lambda, AlphaCat);
		DoubleArrays.scale(this.FTLambda, 1-AlphaCat);
	}

	@Override
	public int dimension() {
		return 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength 
                + hiddenSize + 2 * visibleSize  +  catSize * hiddenSize + catSize;
	}

	@Override
	public double valueAt(double[] x) {
		Theta Theta1 = new Theta(x,hiddenSize,visibleSize,dictionaryLength);
		FineTunableTheta Theta2 = new FineTunableTheta(x,hiddenSize,visibleSize,catSize,dictionaryLength);
		Theta2.setWe( Theta2.We.add(WeOrig) );
		try{
			Computer.Compute(Theta1, Lambda, WeOrig);
			double costRAE = Computer.getCost();
			double[] gradRAE = Computer.getGradient().clone();
			
			Computer.Compute(Theta2, FTLambda);
			double costSUP = Computer.getCost();
			gradient = Computer.getGradient();
			
			value = costRAE + costSUP;
			for(int i=0; i<gradRAE.length; i++)
				gradient[i] += gradRAE[i];
		}
		catch(Exception e)
		{
			System.err.println("Error while calculating graident : " + e.getMessage());
			e.printStackTrace();
		}
		return value;
	}

	@Override
	public double[] derivativeAt(double[] x){
		if( gradient == null )
			valueAt(x);
		double[] retGrad = gradient;
		gradient = null;
		return retGrad;
	}

}
