package math;

import org.jblas.*;

public class Norm1Tanh extends DifferentiableMatrixFunction {
	
	/**
	 * TODO Currently jblas supports only Frobenius norm,
	 * tanh' should actually use the Euclidean norm.
	 * @param M input double matrix
	 * @return     
	 * nrm = norm(x)
	 * y = (x-x.^3)
	 * diag(1-x.^2)./nrm - y*x'./nrm^3
	 */	
	@Override
	public DoubleMatrix derivativeAt(DoubleMatrix M)
	{
		double norm = M.norm2();
		DoubleMatrix Squared = M.mul(M);
		DoubleMatrix y = M.sub( Squared.mul(M) );
		DoubleMatrix p1 = DoubleMatrix.diag((Squared.mul(-1)).add(1)).divi(norm);
		DoubleMatrix p2 = (y.mmul(M.transpose())).divi(Math.pow(norm,3));
		return p1.subi(p2);    	
	}
	
	
	/**
	 * @param M input double matrix
	 * @return Return the plain tanh of the function, NOT norm
	 * as in the original matlab file.
	 */
	@Override
	public DoubleMatrix valueAt(DoubleMatrix M)
	{
		return MatrixFunctions.tanh(M);
	}

	/**
	public static void main(String[] args)
	{
		DoubleMatrix p = new DoubleMatrix(new double[]{1,2,3,4});
		Norm1Tanh t = new Norm1Tanh();
		System.out.println(p);
		System.out.println(t.derivativeAt(p));
		 
//		  norm1tanh_prime([1 2 3 4]')
//				0         0         0         0
//		   0.0365   -0.4747    0.1095    0.1461
//		   0.1461    0.2921   -1.0224    0.5842
//		   0.3651    0.7303    1.0954   -1.2780
		
	}
	**/
}
