package rae;

import org.jblas.*;

import util.*;

import java.util.*;

public class Tree {
	Node[] T;
	Structure structure;
	int SentenceLength, TreeSize;
	double TotalScore;
	
	public Tree(int SentenceLength)
	{
		this.SentenceLength = SentenceLength;
		TreeSize = 2 * SentenceLength - 1;
		T = new Node[TreeSize];
		structure = new Structure( TreeSize );
	}
	
	public Tree(int SentenceLength, int HiddenSize, DoubleMatrix WordsEmbedded)
	{	
		this(SentenceLength);
		for(int i=0; i<TreeSize; i++)
		{
			T[i] = new Node(i,SentenceLength,HiddenSize,WordsEmbedded);
			structure.add(new Pair<Integer,Integer>(-1,-1));
		}
	}
	
	public Tree(int SentenceLength, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded)
	{
		this(SentenceLength);
		for(int i=0; i<TreeSize; i++)
		{
			T[i] = new Node(i,SentenceLength,HiddenSize,CatSize,WordsEmbedded);
			structure.add(new Pair<Integer,Integer>(-1,-1));
		}		
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
	double score, Freq;
	DoubleMatrix UnnormalizedFeatures, 
		Features, LeafFeatures, Z, 
		DeltaOut1, DeltaOut2, ParentDelta, 
		catDelta, dW1, dW2, dW3, dW4, dL, Y1C1, Y2C2;
	
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
		score = 0;
		Freq = 0;
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
