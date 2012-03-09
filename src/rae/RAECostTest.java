package rae;

import java.util.Random;
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
		
		int[] data1 = new int[]{1, 0, 4, 3 };
		int[] data2 = new int[]{3, 0};
		
		List<LabeledDatum<Integer,Integer>> dataset = new ArrayList<LabeledDatum<Integer,Integer>>(1);
		dataset.add( new ReviewDatum(new String[]{"tmp1"}, 0, 0, data1));
		dataset.add( new ReviewDatum(new String[]{"tmp2"}, 1, 1, data2));
		double[] lambda = new double[]{1e-05, 0.0001, 1e-05, 0.01};
		
		RAECost cost = new RAECost(alphaCat, 2, beta, DictionarySize, hiddenSize, hiddenSize, 
				lambda, DoubleMatrix.zeros(hiddenSize, DictionarySize), dataset, null, f);
		
		assertTrue( GradientChecker.check(cost) );
		
		DoubleMatrix xMat = DoubleMatrix.ones(cost.dimension(),1).mul(0.1);
		
		assertTrue( Math.abs(cost.valueAt(xMat.data)-0.5257024003476533) < 1e-10 );
		
		Random rgen = new Random(0);
		xMat = DoubleMatrix.ones(cost.dimension(),1);
		for (int i=0; i<xMat.rows; i++)
			xMat.put(i, 0, rgen.nextDouble());
		DoubleArrays.prettyPrint(cost.derivativeAt(xMat.data));
	}

}
