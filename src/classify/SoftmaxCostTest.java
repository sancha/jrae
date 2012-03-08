package classify;

import static org.junit.Assert.*;
import io.*;

import java.io.*;

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
		int numItems = 400;
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
}
