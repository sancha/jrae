package classify;

import static org.junit.Assert.*;

import io.DataSet;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import math.DoubleArrays;

import org.jblas.DoubleMatrix;
import org.junit.Test;


public class SoftmaxClassifierTest {

	@Test
	/**
	 * Dummy linearly classifiable binary data, should get a 100% training accuracy,
	 * unless poorly initialized. So it is run for 10 times with different initializations.
	 */
	public void dummyBinaryTest() {
		String dir = "data/parsed";
		DataSet<LabeledDatum<Double, Integer>,Double> Dataset 
				= new DataSet<LabeledDatum<Double, Integer>,Double>(100);

		int l;
		double[] x = new double[2];

		try {
			FileInputStream fstream = new FileInputStream(dir + "/binary_test.txt");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			for (int i = 0; i < 100; i++) {
				String[] parts = br.readLine().split(" ");
				x[0] = Double.parseDouble(parts[0]);
				x[1] = Double.parseDouble(parts[1]);
				l = Integer.parseInt(parts[2]);

				Dataset.add(new ReviewFeatures(l + " ", l, i, x));
			}

			fstream.close();
			in.close();
			br.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}

		SoftmaxClassifier<Double, Integer> c = new SoftmaxClassifier<Double, Integer>( );

		for(int i=0; i<10; i++)
		{
			Accuracy a = c.train(Dataset.Data);
			System.out.println(a);
			assertTrue(a.Accuracy == 1.0);
		}
	}
	
	@Test
	public void dummyMultiClassTest (){
		int numItems = 400;
		String dir = "data/parsed";
		DataSet<LabeledDatum<Double, Integer>,Double> Dataset 
		= new DataSet<LabeledDatum<Double, Integer>,Double>(numItems);

		int l = 0;
		double[] x = new double[2];
		
		try {
			FileInputStream fstream = new FileInputStream(dir + "/fournary_test.txt");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			for (int i = 0; i < numItems; i++) {
				String[] parts = br.readLine().split(" ");
				x[0] = Double.parseDouble(parts[0]);
				x[1] = Double.parseDouble(parts[1]);
				l = Integer.parseInt(parts[2]);

				Dataset.add(new ReviewFeatures(l + " ", l, i, x));
			}

			fstream.close();
			in.close();
			br.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		
		SoftmaxClassifier<Double, Integer> c = new SoftmaxClassifier<Double, Integer>( );

		for(int i=0; i<10; i++)
		{
			Accuracy a = c.train(Dataset.Data);
			System.out.println(a);
			assertTrue( a.Accuracy == 1.0 );
		}
	}
	
	@Test
	public void testClassifierTheta() throws Exception
	{
		int FeatureLength = 10;
		int CatSize = 3;
		DoubleMatrix W = DoubleMatrix.rand(FeatureLength,CatSize-1);
		DoubleMatrix b = DoubleMatrix.rand(CatSize-1,1);
		
		ClassifierTheta C = new ClassifierTheta(W, b);
		ClassifierTheta D = new ClassifierTheta(C.Theta, FeatureLength, CatSize);
		ClassifierTheta E = new ClassifierTheta(D.W, D.b);
		
		assertTrue (new DoubleMatrix(DoubleArrays.subtract(C.Theta, D.Theta)).sum() == 0);
		assertTrue (new DoubleMatrix(DoubleArrays.subtract(C.Theta, E.Theta)).sum() == 0);
	}
}
