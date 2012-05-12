package rae;

import math.DoubleArrays;

import org.jblas.*;

import classify.LabeledDatum;

import util.*;

import java.util.*;

public class LabeledRAETree implements LabeledDatum<Double, Integer>{
	RAENode[] T;
	double[] feature;
	Structure structure;
	int SentenceLength, TreeSize, Label;
	double TotalScore;
	
	public LabeledRAETree(int SentenceLength, int Label)
	{
		this.SentenceLength = SentenceLength;
		TreeSize = 2 * SentenceLength - 1;
		T = new RAENode[TreeSize];
		structure = new Structure( TreeSize );
		this.Label = Label;
	}
	
	public LabeledRAETree(int SentenceLength, int Label, int HiddenSize, DoubleMatrix WordsEmbedded)
	{	
		this(SentenceLength, Label);
		for(int i=0; i<TreeSize; i++)
		{
			T[i] = new RAENode(i,SentenceLength,HiddenSize,WordsEmbedded);
			structure.add(new Pair<Integer,Integer>(-1,-1));
		}
	}
	
	public RAENode[] getNodes ()
	{
		return T;
	}
	
	public LabeledRAETree(int SentenceLength, int Label, int HiddenSize, int CatSize, DoubleMatrix WordsEmbedded)
	{
		this(SentenceLength, Label);
		for(int i=0; i<TreeSize; i++)
		{
			T[i] = new RAENode(i,SentenceLength,HiddenSize,CatSize,WordsEmbedded);
			structure.add(new Pair<Integer,Integer>(-1,-1));
		}		
	}
	
	public int[] getStructureString()
	{
		int[] parents = new int[ TreeSize ];
		Arrays.fill(parents, -1);
		
		for (int i=TreeSize-1; i>=0; i--)
		{
			int leftChild = structure.get(i).getFirst();
			int rightChild = structure.get(i).getSecond();
			if (leftChild != -1 && rightChild != -1)
			{
				if (parents[ leftChild ] != -1
						|| parents[ rightChild ] != -1)
					System.err.println ("TreeStructure is messed up!");
				parents[ leftChild ] = i;
				parents[ rightChild ] = i;
			}
		}
		return parents;
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
		int scoreLength = T[0].scores.length;
		
		feature = new double[ HiddenSize * 4 + scoreLength * 3 ];
		DoubleMatrix tf = new DoubleMatrix(HiddenSize,TreeSize);
		DoubleMatrix leafFeatures = new DoubleMatrix(HiddenSize,1);
		DoubleMatrix interFeatures = new DoubleMatrix(HiddenSize,1);
		
		
		double[] leafScores = new double[scoreLength];
		double[] interScores = new double[scoreLength]; 
		double[] meanScores = new double[scoreLength];
		
		if(SentenceLength > 1)
		{
			for(int i=0; i<TreeSize; i++)
			{
				tf.putColumn(i, T[i].Features);
				meanScores = DoubleArrays.addi(meanScores, T[i].scores);
        if (T[i].isLeaf()) {
          leafFeatures.addi(T[i].Features);
          leafScores = DoubleArrays.addi(leafScores, T[i].scores);
        } else {
          interFeatures.addi(T[i].Features);
          interScores = DoubleArrays.addi(interScores, T[i].scores);
        }
			}
			tf.muli(1.0/TreeSize);
			leafFeatures.muli(1.0/SentenceLength);
			interFeatures.muli(1.0/(SentenceLength-1));
			
			meanScores = DoubleArrays.multiply(meanScores,1.0/TreeSize);
			leafScores = DoubleArrays.multiply(leafScores,1.0/SentenceLength);
			interScores = DoubleArrays.multiply(interScores,1.0/(SentenceLength-1));
      
			System.arraycopy(T[ 2 * SentenceLength - 2 ].Features.data, 0, feature, 0, HiddenSize); //root node response
			System.arraycopy(tf.rowSums().data, 0, feature,  1*HiddenSize, HiddenSize); //average node responses
			System.arraycopy(interFeatures.data, 0, feature, 2*HiddenSize, HiddenSize); //average leaf node responses
			System.arraycopy(leafFeatures.data, 0, feature,  3*HiddenSize, HiddenSize); //average non-leaf node responses
			
			System.arraycopy(meanScores, 0, feature,  4*HiddenSize, scoreLength); //average classifier responses
			System.arraycopy(interScores, 0, feature,  4*HiddenSize + 1*scoreLength, scoreLength); //average non-leaf classifier responses
			System.arraycopy(leafScores, 0, feature,  4*HiddenSize + 2*scoreLength, scoreLength); //average leaf classifier responses
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
	
	public String toString ()
	{
		String retString = "";
		for (Pair<Integer,Integer> pii : this)
			retString += "<"+pii.getFirst()+","+pii.getSecond()+">";
		return retString;
	}
}