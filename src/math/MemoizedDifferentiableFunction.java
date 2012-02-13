package math;

public abstract class MemoizedDifferentiableFunction implements DifferentiableFunction {
	protected double[] prevQuery, gradient;
	protected double value;
	protected int evalCount;
	
	protected void initPrevQuery()
	{
		prevQuery = new double[ dimension() ];
	}
	
	protected boolean requiresEvaluation(double[] x)
	{
		if(DoubleArrays.equals(x,prevQuery))
			return false;
		
		System.arraycopy(x, 0, prevQuery, 0, x.length);
		evalCount++;	
		return true;
	}
	
	@Override
	public double[] derivativeAt(double[] x){
		if(DoubleArrays.equals(x,prevQuery))
			return gradient;
		valueAt(x);
		return gradient;
	}
}
