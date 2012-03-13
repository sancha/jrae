package rae;
import java.util.LinkedList;

import org.jblas.DoubleMatrix;

public class RAENode {
	RAENode parent, LeftChild, RightChild;
	int NodeName, SubtreeSize;
	double[] scores; //, Freq;
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
	public RAENode(int NodeIndex, int SentenceLength, int HiddenSize, DoubleMatrix WordsEmbedded)
	{
		NodeName = NodeIndex;
		parent = LeftChild = RightChild = null;
		scores = null;
//			Freq = 0;
		SubtreeSize = 0;
		if( NodeIndex < SentenceLength )
		{
			Features = WordsEmbedded.getColumn(NodeIndex);
			UnnormalizedFeatures = WordsEmbedded.getColumn(NodeIndex);
		}		
	}
	
	public double[] getScores ()
	{
		return scores;
	}
	
	public double[] getFeatures ()
	{
		return Features.data;
	}
	
	public LinkedList<Integer> getSubtreeWordIndices ()
	{
		LinkedList<Integer> list = new LinkedList<Integer>();
		if (isLeaf ())
			list.add(NodeName);
		else{
			list.addAll(LeftChild.getSubtreeWordIndices());
			list.addAll(RightChild.getSubtreeWordIndices());
		}
		return list;
	}
	
	public RAENode(int NodeIndex, int SentenceLength, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded)
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
