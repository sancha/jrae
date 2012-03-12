package rae;

import org.jblas.*;

import java.io.*;
import classify.ClassifierTheta;

public class FineTunableTheta extends Theta{
	
	DoubleMatrix Wcat, bcat;
	int CatSize;
	private static final long serialVersionUID = 752647956355547L;
	
	public FineTunableTheta(double[] iTheta, int hiddenSize, int visibleSize, int CatSize, int dictionaryLength)
	{
		super();
		this.hiddenSize = hiddenSize;
		this.visibleSize = visibleSize;
		this.dictionaryLength = dictionaryLength;
		this.CatSize = CatSize;
		fixIndices();
		DoubleMatrix Full = new DoubleMatrix(iTheta);
		
		W1 = Full.getRowRange(Wbegins[0], Wends[0]+1, 0).reshape(hiddenSize, visibleSize);
		W2 = Full.getRowRange(Wbegins[1], Wends[1]+1, 0).reshape(hiddenSize, visibleSize);
		W3 = Full.getRowRange(Wbegins[2], Wends[2]+1, 0).reshape(visibleSize, hiddenSize);
		W4 = Full.getRowRange(Wbegins[3], Wends[3]+1, 0).reshape(visibleSize, hiddenSize);
		We = Full.getRowRange(Wbegins[4], Wends[4]+1, 0).reshape(hiddenSize, dictionaryLength);
		
		b1 = Full.getRowRange(bbegins[0], bends[0]+1, 0).reshape(hiddenSize, 1);
		b2 = Full.getRowRange(bbegins[1], bends[1]+1, 0).reshape(visibleSize, 1);
		b3 = Full.getRowRange(bbegins[2], bends[2]+1, 0).reshape(visibleSize, 1);		
		
		
		Wcat = Full.getRowRange(Wbegins[5], Wends[5]+1, 0).reshape(CatSize, hiddenSize);
		bcat = Full.getRowRange(bbegins[5], bends[5]+1, 0).reshape(CatSize, 1);
		
		Theta = new double[ getThetaSize() ];
		flatten(Theta);
	}
	
	public FineTunableTheta(FineTunableTheta orig)
	{
		super(orig);
		CatSize = orig.CatSize;
		Wcat = orig.Wcat.dup();
		bcat = orig.bcat.dup();
	}
	
	public FineTunableTheta(int hiddenSize, int visibleSize, int catSize, int dictionaryLength, boolean random)
	{
		super(hiddenSize,visibleSize,catSize,dictionaryLength);

		this.CatSize = catSize;
		if(random)
			InitializeMatrices();
		else
			InitializeMatricesToZeros();
		Theta = new double[ this.getThetaSize() ];
		flatten(Theta);		
	}
	
	/**
	 * Set the Ws and bs and populate theta
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
	
	@Override
	public int getThetaSize()
	{
		return 4 * hiddenSize * visibleSize + hiddenSize * dictionaryLength 
                + hiddenSize + 2 * visibleSize + CatSize * hiddenSize + CatSize;
	}
	
	public int getNumCategories()
	{
		return CatSize+1;
	}
	
	public ClassifierTheta getClassifierParameters()
	{
		return new ClassifierTheta(Wcat.transpose(),bcat);
	}

	@Override
	protected void InitializeMatrices()
	{
		super.InitializeMatrices();
		Wcat = (DoubleMatrix.rand(CatSize, hiddenSize).muli(2*r1)).subi(r1);
		bcat = DoubleMatrix.zeros(CatSize, 1);
	}
	
	@Override
	protected void InitializeMatricesToZeros()
	{
		super.InitializeMatricesToZeros();
		Wcat = DoubleMatrix.zeros(CatSize, hiddenSize);
		bcat = DoubleMatrix.zeros(CatSize, 1);
	}
	
	@Override
	protected void flatten(double[] Theta)
	{
		fixIndices();
		super.flatten(Theta);
		System.arraycopy(Wcat.toArray(), 0, Theta, Wbegins[5], CatSize * hiddenSize);
		System.arraycopy(bcat.toArray(), 0, Theta, bbegins[5], CatSize);			
	}
	
	@Override
	protected void fixIndices()
	{
		super.fixIndices();
		Wbegins[5] = bends[2] + 1;		Wends[5] = Wbegins[5] + CatSize * hiddenSize -1;	//Wcat
		bbegins[5] = Wends[5] + 1;		bends[5] = bbegins[5] + CatSize - 1;				//bcat		
	
//		for(int i=0; i<=5; i++)
//			System.out.println (Wbegins[i] + " " + Wends[i]);
//		System.out.println ("----");
//		for(int i=0; i<=5; i++)
//			System.out.println (bbegins[i] + " " + bends[i]);		
//		System.out.println ("----");
//		System.out.println ("----");
	}
}
