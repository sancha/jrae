package math;

import org.jblas.*;

public class Softmax extends DifferentiableMatrixFunction
{
	/**
	 * @param M
	 * @return e^{\eta_j} / \sum_i e^{\eta_i} 
	 */
	@Override
	public DoubleMatrix valueAt(DoubleMatrix M) {
		DoubleMatrix exp = MatrixFunctions.exp(M);
		DoubleMatrix sums = exp.columnSums();
		return exp.diviRowVector(sums);
	}

	/**
	 * @param X input double matrix
	 * @return derivative of softmax, has the same formula as sigmoid :) 
	 */
	@Override
	public DoubleMatrix derivativeAt(DoubleMatrix X) {
		DoubleMatrix M = valueAt(X);
		return M.mul( (M.mul(-1)).addi(1) );
	}
}
