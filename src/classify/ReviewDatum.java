package classify;

import io.DataSet;

import java.util.*;

public class ReviewDatum implements LabeledDatum<Integer,Integer> {
	public static final int UnknownLabel = -1;
	
	private String ReviewStr;
	private int[] Indices;
	private int Label, Index;
	
	public ReviewDatum(String review, int label, int index)
	{
		this.Label = label;
		this.ReviewStr = review;
		this.Index = index;
	}
	
	public ReviewDatum(String review, int label, int itemNo, int[] indices)
	{
		this(review,label,itemNo);
		this.Indices = new int[ indices.length ];
		for(int i=0; i<indices.length; i++)
			this.Indices[i] = indices[i];
	}
	
	public void indexWords(Map<String,Integer> WordsIndexer)
	{
		String[] parts = ReviewStr.split(" ");
		Indices = new int[ parts.length ];
		for(int i=0; i<parts.length; i++)
			if (WordsIndexer.containsKey(parts[i]))
				Indices[i] = WordsIndexer.get(parts[i]);
			else
				Indices[i] = WordsIndexer.get(DataSet.UNK);
	}
	
	@Override
	public Collection<Integer> getFeatures() 
	{
		List<Integer> Features = new ArrayList<Integer>( Indices.length );
	    for(int index : Indices)
	    	Features.add(index);
	    return Features;
	}

	@Override
	public Integer getLabel() 
	{
		return Label;
	}
	
	@Override
	public String toString()
	{
		return Index + " // " + ReviewStr;
	}
	
	public int[] getIndices()
	{
		return Indices;
	}
	
	public int getDataIndex()
	{
		return Index;
	}
}
