package classify;

import java.io.Serializable;

import org.jblas.*;

public class ClassifierTheta implements Serializable{
	private static final long serialVersionUID = -687919927781459921L;
	
	DoubleMatrix W,b;
	double[] Theta;
	int FeatureLength;
	/**
	 * CatSize is always k-1 where k is the number of classes 
	 * We do not want to over parameterize, so we always have
	 * the last set of weights to be zero
	 */
	int CatSize;
	
	public ClassifierTheta(int FeatureLength, int CatSize)
	{
		CatSize--;
		this.FeatureLength = FeatureLength;
		this.CatSize = CatSize;
		W = DoubleMatrix.rand(FeatureLength,CatSize).subi(0.5);
		b = DoubleMatrix.rand(CatSize,1).subi(0.5);
		Theta = new double[ FeatureLength * CatSize + CatSize ];
		flatten();
	}
	
	public ClassifierTheta(DoubleMatrix W, DoubleMatrix b)
	{
		CatSize = W.columns;
		FeatureLength = W.rows;
		this.W = W;
		this.b = b;
		Theta = new double[ FeatureLength * CatSize + CatSize];
		flatten();
	}
	
	public ClassifierTheta(double[] t, int FeatureLength, int CatSize)
	{
		CatSize--;
		Theta = t.clone();
		this.FeatureLength = FeatureLength;
		this.CatSize = CatSize;
		W = DoubleMatrix.zeros(FeatureLength,CatSize);
		b = DoubleMatrix.zeros(CatSize,1);
		
		if(Theta.length != CatSize * ( 1 + FeatureLength ) )
			System.err.println("ClassifierTheta : Size Mismatch : " 
							+ Theta.length + " != " + (CatSize * ( 1 + FeatureLength )));
		build ();
	}
	
	public void flatten ()
	{
		System.arraycopy(W.data, 0, Theta, 0, FeatureLength * CatSize);
		System.arraycopy(b.data, 0, Theta, FeatureLength * CatSize, CatSize);
	}
	
	private void build ()
	{
		System.arraycopy(Theta, 0, W.data, 0, FeatureLength * CatSize);
		System.arraycopy(Theta, FeatureLength * CatSize, b.data, 0, CatSize);
	}
	
	public DoubleMatrix getW(){
		return W;
	}
	
	public DoubleMatrix getb(){
		return b;
	}
}
