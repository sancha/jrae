package math;

import org.jblas.*;

public class GradientChecker {
	
	public static boolean check(DifferentiableFunction Func)
	{
		int size = Func.dimension();
		double p = size;
		int attempts = 10;
		while( attempts > 0 )
		{
			DoubleMatrix xMat = DoubleMatrix.rand(size);
//			DoubleMatrix xMat = attempts != 10 ? DoubleMatrix.rand(size) : DoubleMatrix.ones(size).mul(0.1);
			double[] x = xMat.data;
			double ReturnedCost = Func.valueAt(x);
			double[] ReturnedGradient = Func.derivativeAt(x);
			double[] NumericalGradient = new double[ size ]; 
			double PartCosts;
			
			double Mean = 2e-6 * ((1 + xMat.norm2()) / p);
			for(int i=0; i<size; i++)
			{
				double[] e = DoubleMatrix.zeros(size).data;
				e[i] = 1;
				DoubleArrays.scale(e, Mean);
				double[] y = DoubleArrays.add(x,e);
				PartCosts = Func.valueAt(y);
				NumericalGradient[i] = ( PartCosts - ReturnedCost )/ Mean;
			}
			
			double diff = 0, totalDiff = 0, maxDiff = -1.0;
			for(int i=0; i<size; i++)
			{
				diff = Math.abs( NumericalGradient[i] - ReturnedGradient[i] );
				if (diff > 1e-5)
					System.out.println (i + "-" + diff +" "+ NumericalGradient[i] + " " + ReturnedGradient[i]);
				maxDiff = Math.max(diff, maxDiff);
				totalDiff += diff;
			}
			
			System.out.println(attempts + " " + totalDiff + " " + maxDiff);
			
			if( maxDiff > 1e-4 )
			{
				System.err.println("Gradient calc fails! Max Diff is too damn high");
				return false;				
			}
//			if( totalDiff > 1e-4 )
//			{
//				System.err.println("Gradient calc fails! Total Diff is too damn high");
//				return false;
//			}
			attempts--;
		}
		return true;
	}
}
