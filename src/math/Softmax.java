package math;

import org.jblas.*;
import util.ArraysHelper;

public class Softmax extends DifferentiableMatrixFunction
{
	/**
	 * @param M
	 * @return e^{\eta_j} / \sum_i e^{\eta_i} 
	 */
	@Override
	public DoubleMatrix valueAt(DoubleMatrix M) {
		int[] rows = ArraysHelper.makeArray(0,M.rows-1);
		DoubleMatrix exp = MatrixFunctions.exp(M);
		exp = DoubleMatrix.concatVertically(DoubleMatrix.ones(1, exp.columns),exp);
		DoubleMatrix sums = exp.columnSums();
		return ((exp.diviRowVector(sums)).getRows(rows));
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
