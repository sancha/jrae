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
		int hiddenSize = 5, DictionarySize = 5, CatSize = 2;
		DifferentiableMatrixFunction f = new Norm1Tanh();
		List<LabeledDatum<Integer,Integer>> dataset = new ArrayList<LabeledDatum<Integer,Integer>>(1);
		
		Random rgen = new Random(0);
		int numDataItems = 50, maxSentenceLength = 39;
		int[][] data = new int[numDataItems][];
		for (int i=0; i<numDataItems; i++)
		{
			data[i] = new int[ 1+rgen.nextInt(maxSentenceLength) ];
			for (int j=0; j<data[i].length; j++)
				data[i][j] = rgen.nextInt(DictionarySize);
			dataset.add(new ReviewDatum(new String[]{"tmp"+i}, rgen.nextInt(CatSize), i, data[i]));
		}
		
		double[] lambda = new double[]{1e-05, 0.0001, 1e-05, 0.01};
		
		RAECost cost = new RAECost(alphaCat, CatSize, beta, DictionarySize, hiddenSize, hiddenSize, 
				lambda, DoubleMatrix.zeros(hiddenSize, DictionarySize), dataset, null, f);
		System.out.println (cost.dimension());
		assertTrue( GradientChecker.check(cost) );
		
//		DoubleMatrix xMat = DoubleMatrix.ones(cost.dimension(),1).mul(0.1);
//		assertTrue( Math.abs(cost.valueAt(xMat.data)-0.5257024003476533) < 1e-10 );
		
		
		DoubleMatrix yMat = DoubleMatrix.zeros(cost.dimension(),1);
		for (int i=0; i<140; i++)
			yMat.put(i, 0, rgen.nextDouble());
		
		for (int i=140; i<yMat.rows; i++)
			yMat.put(i, 0, (double)(i-139)/10.0);
		cost.valueAt (yMat.data);
		
//		DoubleArrays.prettyPrint(cost.derivativeAt(yMat.data));
	}

}
