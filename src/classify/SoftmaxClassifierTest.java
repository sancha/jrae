package classify;

import static org.junit.Assert.*;

import io.DataSet;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;

import org.junit.Test;


public class SoftmaxClassifierTest {

	@Test
	/**
	 * Dummy linearly classifiable data, should get a 100% training accuracy,
	 * unless poorly initialized. So it is run for 10 times with different initializations.
	 */
	public void dummyDataTest() {
		String dir = "data/parsed";
		DataSet<LabeledDatum<Double, Integer>,Double> Dataset 
				= new DataSet<LabeledDatum<Double, Integer>,Double>(100);

		int l;
		double[] x = new double[2];

		try {
			FileInputStream fstream = new FileInputStream(dir + "/test.txt");
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
			if( a.Accuracy == 1.0 )
			{
				assertTrue(Boolean.TRUE);
				return;
			}
		}
		assertTrue(Boolean.FALSE);
	}
}
