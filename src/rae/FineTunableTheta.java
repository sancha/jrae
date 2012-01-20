package rae;

import org.jblas.*;
import java.io.*;

public class FineTunableTheta extends Theta{
	
	DoubleMatrix Wcat, bcat;
	int CatSize;
	private static final long serialVersionUID = 752647956355547L;
	
	/**
	 * @param iTheta
	 * @param hiddenSize
	 * @param visibleSize
	 * @param catSize
	 * @param dictionaryLength
	 */
	public FineTunableTheta(double[] iTheta, int hiddenSize, int visibleSize, int CatSize, int dictionaryLength)
	{
		super(iTheta, hiddenSize, visibleSize, dictionaryLength);
		this.CatSize = CatSize;
		fixIndices();
		
		DoubleMatrix Full = new DoubleMatrix(iTheta);
		Wcat = Full.getRowRange(Wbegins[5], Wends[5]+1, 0).reshape(CatSize, hiddenSize);
		bcat = Full.getRowRange(bbegins[5], bends[5]+1, 0).reshape(CatSize, 1);
	}
	
	public FineTunableTheta(FineTunableTheta orig)
	{
		super(orig);
		CatSize = orig.CatSize;
		Wcat = orig.Wcat.dup();
		bcat = orig.bcat.dup();
	}
	
	/**
	 * @param hiddenSize
	 * @param visibleSize
	 * @param catSize
	 * @param dictionaryLength
	 */
	public FineTunableTheta(int hiddenSize, int visibleSize, int catSize, int dictionaryLength)
	{
		super(hiddenSize,visibleSize,catSize,dictionaryLength);

		// Initialize the matrices to small random values and then
		// convert weights and bias matrices to the vector form.
		// This step will "unroll" (flatten and concatenate together) all
		// your parameters into a vector, which can then be used with minFunc.
		this.CatSize = catSize;
		InitializeMatrices();
		Theta = new double[ this.getThetaSize() ];
		flatten(Theta);		
		
	}
	
	/**
	 * Set the Ws and bs and populate theta
	 * @param W1
	 * @param W2
	 * @param W3
	 * @param W4
	 * @param We
	 * @param b1
	 * @param b2
	 * @param b3
	 */
	public FineTunableTheta(DoubleMatrix W1, DoubleMatrix W2, 
					DoubleMatrix W3, DoubleMatrix W4, DoubleMatrix Wcat,
					DoubleMatrix We, DoubleMatrix b1, DoubleMatrix b2, 
					DoubleMatrix b3, DoubleMatrix bcat)
	{
		this.W1 = W1;
		this.W2 = W2;
		this.W3 = W3;
		this.W4 = W4;
		this.We = We;
		this.b1 = b1;
		this.b2 = b2;
		this.b3 = b3;
		this.Wcat = Wcat;
		this.bcat = bcat;

		hiddenSize = W1.rows;
		visibleSize = W1.columns;
		dictionaryLength = We.columns;
		CatSize = bcat.rows;
		
		Theta = new double[ getThetaSize()  ];
		flatten(Theta);		
	}
	
	public void Dump(String FileName) throws IOException
	{
		FileOutputStream fos = new FileOutputStream(FileName);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(this);
		oos.flush();
		oos.close();
	}
	
	public int getThetaSize()
	{
		return 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength 
                + hiddenSize + 2 * visibleSize + CatSize * hiddenSize + CatSize;
	}

	protected void InitializeMatrices()
	{
		super.InitializeMatrices();
		Wcat = (DoubleMatrix.rand(CatSize, hiddenSize).muli(2*r1)).subi(r1);
		bcat = DoubleMatrix.zeros(CatSize, 1);
	}
	
	protected void flatten(double[] Theta)
	{
		fixIndices();
		super.flatten(Theta);
		System.arraycopy(Wcat.toArray(), 0, Theta, Wbegins[5], CatSize * hiddenSize);
		System.arraycopy(bcat.toArray(), 0, Theta, bbegins[5], CatSize);			
	}
	
	protected void fixIndices()
	{
		super.fixIndices();
		Wbegins[5] = bends[2] + 1;		Wends[5] = Wbegins[5] + CatSize * hiddenSize -1;	//Wcat
		bbegins[5] = Wends[5] + 1;		bends[5] = bbegins[5] + CatSize - 1;				//bcat		
	}
}
