package rae;

import static org.junit.Assert.*;
import java.util.Random;
import math.DoubleArrays;
import org.junit.Test;

/**
 * @author ssanjeev
 */
public class FineTunableThetaTest {

	@Test
	public void test() throws Exception {
		int h = 10, k = 4, d = 50;
		FineTunableTheta t0 = new FineTunableTheta(h, h, k, d, false);
		int size = t0.Theta.length;
		Random rgen = new Random(10);

		double[] base = new double[size];
		for (int i = 0; i < size; i++)
			base[i] = rgen.nextDouble();

		FineTunableTheta t2 = new FineTunableTheta(base, h, h, k, d);
		FineTunableTheta t3 = new FineTunableTheta(t2.W1, t2.W2, t2.W3, t2.W4,
				t2.Wcat, t2.We, t2.b1, t2.b2, t2.b3, t2.bcat);
		FineTunableTheta t4 = new FineTunableTheta (t3.Theta, h, h, k, d); 
		
		assertTrue (t4.W1.sub(t2.W1).sum() == 0);
		assertTrue (t4.W2.sub(t2.W2).sum() == 0);
		assertTrue (t4.W3.sub(t2.W3).sum() == 0);
		assertTrue (t4.W4.sub(t2.W4).sum() == 0);
		assertTrue (t4.Wcat.sub(t2.Wcat).sum() == 0);
		assertTrue (t4.We.sub(t2.We).sum() == 0);
		assertTrue (t4.b1.sub(t2.b1).sum() == 0);
		assertTrue (t4.b2.sub(t2.b2).sum() == 0);
		assertTrue (t4.b3.sub(t2.b3).sum() == 0);
		
		assertTrue (DoubleArrays.total(DoubleArrays.subtract(t4.Theta,t2.Theta)) == 0);
	}

}
