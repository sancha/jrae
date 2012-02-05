package math;

import org.jblas.*;

public class Sigmoid extends DifferentiableMatrixFunction {

	
	/**
	 * @param M
	 * @return sigmoid = 1 ./ (1 + exp(-x));
	 */
	@Override
	public DoubleMatrix valueAt(DoubleMatrix M) {
		DoubleMatrix Denom = (MatrixFunctions.exp(M.mul(-1))).addi(1);
		return Denom.rdivi(1);
	}

	/**
	 * @param M input double matrix
	 * @return sigmoid_prime = M.*(1-M);
	 */
	@Override
	public DoubleMatrix derivativeAt(DoubleMatrix M) {
		return M.mul( (M.mul(-1)).addi(1) );
	}
	
}
