package classify;

import static org.junit.Assert.*;
import io.*;

import java.io.*;
import math.DoubleArrays;
import math.GradientChecker;

import org.jblas.DoubleMatrix;
import org.junit.Test;

import util.ArraysHelper;


public class SoftmaxCostTest {
	SoftmaxCost costfn;
	
	@Test
	public void testDummyData() {
		String dir = "data/parsed";
		int numCat = 2;
		DoubleMatrix features = DoubleMatrix.zeros(2, 100);
		int[] l = new int[100];

		try 
		{
			FileInputStream fstream = new FileInputStream(dir + "/binary_test.txt");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			for (int i = 0; i < 100; i++) {
				String[] parts = br.readLine().split(" ");
				double x = Double.parseDouble(parts[0]);
				double y = Double.parseDouble(parts[1]);
				l[i] = Integer.parseInt(parts[2]);

				features.put(0, i, x);
				features.put(1, i, y);
			}

			fstream.close();
			in.close();
			br.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		costfn = new SoftmaxCost(features, l, numCat, 1e-6);
		assertTrue(GradientChecker.check(costfn));
	}
	
	@Test
	public void testRealData() throws IOException, ClassNotFoundException
	{
		String dir = "data/parsed";
		LabeledDataSet<LabeledDatum<Integer,Integer>,Integer,Integer> Dataset = new MatProcessData(dir);
		
		int CatSize = Dataset.getCatSize();
	
		FileInputStream fis = new FileInputStream("data/parsed/features.dat");
		ObjectInputStream ois = new ObjectInputStream(fis);
		DoubleMatrix features = ((DoubleMatrix) ois.readObject()).getColumns( ArraysHelper.makeArray(0,999));
		ois.close();
		
		System.out.println(features.rows + " " + features.columns );
		
		int[] Labels = new int[ 1000 ];
		for(int i=0; i< 1000; i++)
			Labels[i] = Dataset.Data.get(i).getLabel();
		
		SoftmaxCost TrainingCostFunction = new SoftmaxCost(features,Labels,CatSize,0.2);
		
		System.out.println("Checking...");
		
		assertTrue(GradientChecker.check(TrainingCostFunction));
	}
	

	@Test
	public void testFournaryData ()
	{
		int numItems = 1;
		int numCat = 4;
		String dir = "data/parsed";
		DoubleMatrix features = DoubleMatrix.zeros(2, numItems);
		int[] l = new int[numItems];

		try 
		{
			FileInputStream fstream = new FileInputStream(dir + "/fournary_test.txt");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			for (int i = 0; i < numItems; i++) {
				String[] parts = br.readLine().split(" ");
				double x = Double.parseDouble(parts[0]);
				double y = Double.parseDouble(parts[1]);
				l[i] = Integer.parseInt(parts[2]);

				features.put(0, i, x);
				features.put(1, i, y);
			}

			fstream.close();
			in.close();
			br.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		costfn = new SoftmaxCost(features, l, numCat, 0);
		assertTrue(GradientChecker.check(costfn));
	}
	
	@Test
	public void testAnother ()
	{
		double[][] f = 
				{{0.701039,0.366361,0.671759,0.297568,0.449285,0.453154,0.449638,0.297568,0.366361,0.452110},
				{0.745353,0.477637,0.740417,0.428904,0.440570,0.436850,0.440844,0.428904,0.477637,0.437994},
				{0.607288,0.703970,0.894435,0.687552,0.449640,0.455082,0.450274,0.687552,0.703970,0.454983},
				{0.235082,0.322768,0.579958,0.408496,0.447331,0.442386,0.446858,0.408496,0.322768,0.440377},
				{0.389043,0.011655,0.510755,0.353052,0.449177,0.448340,0.448390,0.353052,0.011655,0.450352}};
		DoubleMatrix features = new DoubleMatrix (f);
		int[] labels = {0,0,0,0,0,0,0,1,1,1};
		
		costfn = new SoftmaxCost(features, labels, 3, 0);
		System.out.println (costfn.dimension());
		
		double[] shu = {0.1,0.3,0.5,0.7,0.9,0.2,0.4,0.6,0.8,1.0,1.1,1.2};
		System.out.println (costfn.valueAt (shu));
		
		DoubleArrays.prettyPrint(costfn.derivativeAt (shu));
	}
}
