package math;

import org.jblas.*;

public abstract class DifferentiableMatrixFunction {
	
	public DoubleMatrix valueAt(DoubleMatrix M){ return null; }
	
	public DoubleMatrix derivativeAt(DoubleMatrix M){ return null; }
}
