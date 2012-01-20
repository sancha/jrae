package rae;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import junit.framework.Assert;

import math.DoubleArrays;

import org.junit.Test;


public class FineTunableThetaTest 
{

	@Test
	public void test() throws Exception
	{
		String dir = "data/parsed";
		FileInputStream fis = new FileInputStream(dir + "/opttheta.dat");
		ObjectInputStream ois = new ObjectInputStream(fis);
		FineTunableTheta tunedTheta = (FineTunableTheta) ois.readObject();
		ois.close();
		
		FineTunableTheta base = new FineTunableTheta(tunedTheta.W1, tunedTheta.W2, tunedTheta.W3, tunedTheta.W4, 
				tunedTheta.Wcat, tunedTheta.We, tunedTheta.b1, tunedTheta.b2, tunedTheta.b3, tunedTheta.bcat);
		
		double[] diff = DoubleArrays.subtract(base.Theta, tunedTheta.Theta);
		double totalDiff = 0;
		for(int i=0; i<diff.length;i++)
			totalDiff += Math.abs(diff[i]);
		
		System.out.println(totalDiff);
		Assert.assertTrue( totalDiff == 0 );
		
		FineTunableTheta built = new FineTunableTheta( base.Theta, 50, 50, 1, 14043 );
		
		Assert.assertTrue( (built.W1.sub( tunedTheta.W1 )).sum() == 0 );
		Assert.assertTrue( (built.W2.sub( tunedTheta.W2 )).sum() == 0 );
		Assert.assertTrue( (built.W3.sub( tunedTheta.W3 )).sum() == 0 );
		Assert.assertTrue( (built.W4.sub( tunedTheta.W4 )).sum() == 0 );
		Assert.assertTrue( (built.We.sub( tunedTheta.We )).sum() == 0 );
		Assert.assertTrue( (built.Wcat.sub( tunedTheta.Wcat )).sum() == 0 );
		
		Assert.assertTrue( (built.b1.sub( tunedTheta.b1 )).sum() == 0 );
		Assert.assertTrue( (built.b2.sub( tunedTheta.b2 )).sum() == 0 );
		Assert.assertTrue( (built.b3.sub( tunedTheta.b3 )).sum() == 0 );
		Assert.assertTrue( (built.bcat.sub( tunedTheta.bcat )).sum() == 0 );
	}

}
