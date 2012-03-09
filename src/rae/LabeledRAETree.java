package rae;

import org.jblas.*;

import classify.LabeledDatum;

import util.*;

import java.util.*;

public class LabeledRAETree implements LabeledDatum<Double, Integer>{
	Node[] T;
	double[] feature;
	Structure structure;
	int SentenceLength, TreeSize, Label;
	double TotalScore;
	
	public LabeledRAETree(int SentenceLength, int Label)
	{
		this.SentenceLength = SentenceLength;
		TreeSize = 2 * SentenceLength - 1;
		T = new Node[TreeSize];
		structure = new Structure( TreeSize );
		this.Label = Label;
	}
	
	public LabeledRAETree(int SentenceLength, int Label, int HiddenSize, DoubleMatrix WordsEmbedded)
	{	
		this(SentenceLength, Label);
		for(int i=0; i<TreeSize; i++)
		{
			T[i] = new Node(i,SentenceLength,HiddenSize,WordsEmbedded);
			structure.add(new Pair<Integer,Integer>(-1,-1));
		}
	}
	
	public LabeledRAETree(int SentenceLength, int Label, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded)
	{
		this(SentenceLength, Label);
		for(int i=0; i<TreeSize; i++)
		{
			T[i] = new Node(i,SentenceLength,HiddenSize,CatSize,WordsEmbedded);
			structure.add(new Pair<Integer,Integer>(-1,-1));
		}		
	}
	
	public String getStructureString()
	{
		int[] parents = new int[ TreeSize ];
		for (int i=TreeSize-1; i>=0; i--)
		{
			parents[ structure.get(i).getFirst() ] = i;
			parents[ structure.get(i).getSecond() ] = i;
		}
		return ArraysHelper.makeStringFromIntArray(parents);
	}
	
	@Override
	public String toString()
	{
		return null;
	}

	@Override
	public Integer getLabel() {
		return Label;
	}

	
	public double[] getFeaturesVector()
	{
		if (feature != null)
			return feature;
		
		int HiddenSize = T[0].Features.rows;
		feature = new double[ HiddenSize * 2 ];
		DoubleMatrix tf = new DoubleMatrix(HiddenSize,TreeSize);
		if(SentenceLength > 1)
		{
			for(int i=0; i<TreeSize; i++)
				tf.putColumn(i, T[i].Features);
			tf.muli(1.0/TreeSize);
			
			System.arraycopy(T[ 2 * SentenceLength - 2 ].Features.data, 0, feature, 0, HiddenSize);
			System.arraycopy(tf.rowSums().data, 0, feature, HiddenSize, HiddenSize);
		}
		else
		{
			System.arraycopy(T[ 2 * SentenceLength - 2].Features.data, 0, feature, 0, HiddenSize);
			System.arraycopy(T[ 2 * SentenceLength - 2].Features.data, 0, feature, HiddenSize, HiddenSize);
		}
		return feature;	
	}
	
	@Deprecated
	public Collection<Double> getFeatures() {
		System.err.println ("There's no way I am returning a Collection."
				+ "\nPlease use the getFeatureVector method instead.");
		
		return null;
	}
}

class Structure extends ArrayList<Pair<Integer,Integer>>
{
	private static final long serialVersionUID = -1616780629111786862L;
	public Structure(int Capacity)
	{
		super(Capacity);
	}
}

class Node {
	Node parent, LeftChild, RightChild;
	int NodeName, SubtreeSize;
	double[] catDelta, scores; //, Freq;
	DoubleMatrix UnnormalizedFeatures, 
		Features, LeafFeatures, Z, 
		DeltaOut1, DeltaOut2, ParentDelta, 
		dW1, dW2, dW3, dW4, dL, Y1C1, Y2C2;
	
	/**
	 * Specialized Constructor for fitting in that list
	 * @param NodeIndex
	 * @param SentenceLength
	 * @param HiddenSize
	 * @param WordsEmbedded
	 */
	public Node(int NodeIndex, int SentenceLength, int HiddenSize, DoubleMatrix WordsEmbedded)
	{
		NodeName = NodeIndex;
		parent = LeftChild = RightChild = null;
		scores = null;
//		Freq = 0;
		SubtreeSize = 0;
		if( NodeIndex < SentenceLength )
		{
			Features = WordsEmbedded.getColumn(NodeIndex);
			UnnormalizedFeatures = WordsEmbedded.getColumn(NodeIndex);
		}		
	}
	
	public Node(int NodeIndex, int SentenceLength, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded)
	{
		this(NodeIndex,SentenceLength,HiddenSize,WordsEmbedded);
		DeltaOut1 = DoubleMatrix.zeros(HiddenSize,1);
		DeltaOut2 = DoubleMatrix.zeros(HiddenSize,1);
		ParentDelta = DoubleMatrix.zeros(HiddenSize,1);
		Y1C1 = DoubleMatrix.zeros(HiddenSize,1);
		Y2C2 = DoubleMatrix.zeros(HiddenSize,1);
		if( NodeIndex >= SentenceLength )
		{
			Features = DoubleMatrix.zeros(HiddenSize, 1);
			UnnormalizedFeatures = DoubleMatrix.zeros(HiddenSize, 1);
		}	
	}
	
	public boolean isLeaf()
	{
		if( LeftChild == null && RightChild == null )
			return true;
		else if( LeftChild != null && RightChild != null )
			return false;
		System.err.println("Broken tree, node has one child " + NodeName);
		return false;
	}
}
