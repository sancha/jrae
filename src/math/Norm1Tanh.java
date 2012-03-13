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
}
