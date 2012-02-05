package math;

import org.jblas.*;

public class Tanh extends DifferentiableMatrixFunction {

	/**
	 * Return the plain tanh, available from the library
	 */
	@Override
	public DoubleMatrix valueAt(DoubleMatrix M) {
		return MatrixFunctions.tanh(M);
	}

	/**
	 * @param M input double matrix 
	 * @return tanh_prime = (1-M.^2);
	 */
	@Override
	public DoubleMatrix derivativeAt(DoubleMatrix M) {
		DoubleMatrix Squared = M.mul(M);
		return (Squared.muli(-1)).addi(1);
	}

}
