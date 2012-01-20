package rae;

import static org.junit.Assert.*;
import java.util.*;

import math.*;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import classify.LabeledDatum;
import classify.ReviewDatum;

public class RAECostTest {

	@Test
	public void test() throws Exception 
	{
		double alphaCat = 0.2, beta = 0.5; 
		int hiddenSize = 5, DictionarySize = 5;
		DifferentiableMatrixFunction f = new Norm1Tanh();
		
		int[] data = new int[]{1, 0, 4, 3 };
		
		List<LabeledDatum<Integer,Integer>> dataset = new ArrayList<LabeledDatum<Integer,Integer>>(1);
		dataset.add( new ReviewDatum("tmp", 0, 0, data));
		double[] lambda = new double[]{1e-05, 0.0001, 1e-05, 0.01};
		
		RAECost cost = new RAECost(alphaCat, 1, beta, DictionarySize, hiddenSize, hiddenSize, 
				lambda, DoubleMatrix.zeros(hiddenSize, DictionarySize), dataset, null, f);
	
		assertTrue( GradientChecker.check(cost));
		
	}

}
