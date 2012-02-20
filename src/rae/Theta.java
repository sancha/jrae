package rae;

import java.io.*;
import org.jblas.*;

/**
 * These classes are made only of the constructors.
 * When either constructor is called, all members are initialized and populated
 */
public class Theta implements Serializable{
	
	DoubleMatrix W1, W2, W3, W4;
	DoubleMatrix b1, b2, b3;
	
	//TODO Remove public visibility
	public double[] Theta;
	public DoubleMatrix We;
	
	protected double r1, r2;

	int[] Wbegins, Wends, bbegins, bends;
	
	int hiddenSize, visibleSize, dictionaryLength;
	private static final long serialVersionUID = 752647392162776147L;
	
	/** Dummy constructor, because it is required by Java for subclass **/
	public Theta() {
		Wbegins = new int[6];
		Wends = new int[6]; 
		bbegins = new int[6]; 
		bends = new int[6];
	}
	
	public Theta(int hiddenSize, int visibleSize, int catSize, int dictionaryLength)
	{
		this();
		this.hiddenSize = hiddenSize;
		this.visibleSize = visibleSize;
		this.dictionaryLength = dictionaryLength;
		
		r1 = Math.sqrt(6) / Math.sqrt(hiddenSize+visibleSize+1);
		r2 = 0.001f;
	}
	
	/**
	 * Creates a bunch of DoubleMatrices and returns a flattened concat of 
	 * all matrices as in /initializeParamters.m
	 */
	public Theta(int hiddenSize, int visibleSize, int dictionaryLength, boolean random)
	{
		this(hiddenSize, visibleSize, 1, dictionaryLength);
		// Initialize the matrices to small random values and then
		// convert weights and bias matrices to the vector form.
		// This step will "unroll" (flatten and concatenate together) all
		// your parameters into a vector, which can then be used with minFunc.

		if(random)
			InitializeMatrices();
		else
			InitializeMatricesToZeros();
		Theta = new double[ getThetaSize() ];
		flatten(Theta);		
	}
	
	/**
	 * Copy constructor. Makes deepCopy of all matrices 
	 * and the double[] vector. 
	 * @param orig
	 */
	public Theta(Theta orig)
	{
		this();
		hiddenSize = orig.hiddenSize;
		visibleSize = orig.visibleSize;
		dictionaryLength = orig.dictionaryLength;
		
		Wbegins = orig.Wbegins.clone();
		Wends = orig.Wends.clone();
		bbegins = orig.bbegins.clone();
		bends = orig.bends.clone();
		
		r1 = orig.r1;
		r2 = orig.r2;
		
		Theta = orig.Theta.clone();
		W1 = orig.W1.dup();
		W2 = orig.W2.dup();
		W3 = orig.W3.dup();
		W4 = orig.W4.dup();
		We = orig.We.dup();
		b1 = orig.b1.dup();
		b2 = orig.b2.dup();
		b3 = orig.b3.dup();
	}	
	
	/**
	 * Reconstruct the Theta from theta vector and populate all the W matrices.
	 */
	public Theta(double[] iTheta, int hiddenSize, int visibleSize, int dictionaryLength)
	{
		this();
		this.hiddenSize = hiddenSize;
		this.visibleSize = visibleSize;
		this.dictionaryLength = dictionaryLength;
		this.fixIndices();
		DoubleMatrix Full = new DoubleMatrix(iTheta);
		
		W1 = Full.getRowRange(Wbegins[0], Wends[0]+1, 0).reshape(hiddenSize, visibleSize);
		W2 = Full.getRowRange(Wbegins[1], Wends[1]+1, 0).reshape(hiddenSize, visibleSize);
		W3 = Full.getRowRange(Wbegins[2], Wends[2]+1, 0).reshape(visibleSize, hiddenSize);
		W4 = Full.getRowRange(Wbegins[3], Wends[3]+1, 0).reshape(visibleSize, hiddenSize);
		We = Full.getRowRange(Wbegins[4], Wends[4]+1, 0).reshape(hiddenSize, dictionaryLength);
		
		b1 = Full.getRowRange(bbegins[0], bends[0]+1, 0).reshape(hiddenSize, 1);
		b2 = Full.getRowRange(bbegins[1], bends[1]+1, 0).reshape(visibleSize, 1);
		b3 = Full.getRowRange(bbegins[2], bends[2]+1, 0).reshape(visibleSize, 1);		

		Theta = new double[ getThetaSize() ];
		flatten(Theta);
	}
	
	/**
	 * Set the Ws and bs and populate theta
	 */
	public Theta(DoubleMatrix W1, DoubleMatrix W2, 
					DoubleMatrix W3, DoubleMatrix W4, 
					DoubleMatrix We, DoubleMatrix b1, 
					DoubleMatrix b2, DoubleMatrix b3)
	{
		this();
		this.W1 = W1;
		this.W2 = W2;
		this.W3 = W3;
		this.W4 = W4;
		this.We = We;
		this.b1 = b1;
		this.b2 = b2;
		this.b3 = b3;

		hiddenSize = W1.rows;
		visibleSize = W1.columns;
		dictionaryLength = We.columns;
		
		Theta = new double[ 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength + hiddenSize + 2 * visibleSize  ];
		this.flatten(Theta);
	}
	
	/**
	 * Set We, and update the Theta vector,
	 * @param We
	 */
	public void setWe(DoubleMatrix We)
	{
		this.We = We;
		System.arraycopy(We.toArray(), 0, Theta, Wbegins[4], hiddenSize * dictionaryLength);
	}
	
	public int getThetaSize()
	{
		return 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength + hiddenSize + 2 * visibleSize;
	}
	
	protected void InitializeMatrices()
	{
		Wbegins = new int[6];
		Wends = new int[6];
		bbegins = new int[6];
		bends = new int[6];
		
		W1 = (DoubleMatrix.rand(hiddenSize, visibleSize).muli(2*r1)).subi(r1);
		W2 = (DoubleMatrix.rand(hiddenSize, visibleSize).muli(2*r1)).subi(r1);
		W3 = (DoubleMatrix.rand(visibleSize, hiddenSize).muli(2*r1)).subi(r1);
		W4 = (DoubleMatrix.rand(visibleSize, hiddenSize).muli(2*r1)).subi(r1);
		
		b1 = DoubleMatrix.zeros(hiddenSize, 1);
		b2 = DoubleMatrix.zeros(visibleSize, 1);
		b3 = DoubleMatrix.zeros(visibleSize, 1);		
		
		We = ((DoubleMatrix.rand(hiddenSize, dictionaryLength).muli(2*r1)).subi(r1)).muli(r2);
	}

	protected void InitializeMatricesToZeros()
	{
		Wbegins = new int[6];
		Wends = new int[6];
		bbegins = new int[6];
		bends = new int[6];
		
		W1 = DoubleMatrix.zeros(hiddenSize, visibleSize);
		W2 = DoubleMatrix.zeros(hiddenSize, visibleSize);
		W3 = DoubleMatrix.zeros(visibleSize, hiddenSize);
		W4 = DoubleMatrix.zeros(visibleSize, hiddenSize);
		
		b1 = DoubleMatrix.zeros(hiddenSize, 1);
		b2 = DoubleMatrix.zeros(visibleSize, 1);
		b3 = DoubleMatrix.zeros(visibleSize, 1);		
		
		We = DoubleMatrix.zeros(hiddenSize, dictionaryLength);
	}
	
	protected void flatten(double[] Theta)
	{
		this.fixIndices();
		
		System.arraycopy(W1.toArray(), 0, Theta, Wbegins[0], hiddenSize * visibleSize);
		System.arraycopy(W2.toArray(), 0, Theta, Wbegins[1], hiddenSize * visibleSize);
		System.arraycopy(W3.toArray(), 0, Theta, Wbegins[2], hiddenSize * visibleSize);
		System.arraycopy(W4.toArray(), 0, Theta, Wbegins[3], hiddenSize * visibleSize);
		System.arraycopy(We.toArray(), 0, Theta, Wbegins[4], hiddenSize * dictionaryLength);
		
		System.arraycopy(b1.toArray(), 0, Theta, bbegins[0], hiddenSize);
		System.arraycopy(b2.toArray(), 0, Theta, bbegins[1], visibleSize);
		System.arraycopy(b3.toArray(), 0, Theta, bbegins[2], visibleSize);		
	}
	
	protected void fixIndices()
	{
		Wbegins[0] = 0;				Wends[0] = hiddenSize * visibleSize - 1;					//W1
		Wbegins[1] = Wends[0]+1;	Wends[1] = Wbegins[1] + hiddenSize * visibleSize - 1;		//W2
		Wbegins[2] = Wends[1]+1;	Wends[2] = Wbegins[2] + hiddenSize * visibleSize - 1;		//W3
		Wbegins[3] = Wends[2]+1;	Wends[3] = Wbegins[3] + hiddenSize * visibleSize - 1;		//W4
		Wbegins[4] = Wends[3]+1;	Wends[4] = Wbegins[4] + hiddenSize * dictionaryLength - 1;	//We
		Wbegins[5] = -1;			Wends[5] = -1;												//Wcat
		
		bbegins[0] = Wends[4]+1;	bends[0] = bbegins[0] + hiddenSize - 1;		//b1
		bbegins[1] = bends[0]+1;	bends[1] = bbegins[1] + visibleSize -1;		//b2
		bbegins[2] = bends[1]+1;	bends[2] = bbegins[2] + visibleSize -1;		//b3
		bbegins[3] = -1;			bends[3] = -1;	//b4
		bbegins[4] = -1;			bends[4] = -1;	//be
		bbegins[5] = -1;			bends[5] = -1;	//bcat
	}
}
