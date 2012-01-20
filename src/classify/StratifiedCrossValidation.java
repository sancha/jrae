package classify;


import io.*;

import java.util.*;

public class StratifiedCrossValidation<T extends LabeledDatum<F,L>, F,L> extends CrossValidation<T,F> {

	protected int LabelSetSize;
	protected Set<L> LabelSet;	
	protected LabeledDataSet<T,F,L> Data;
	
	public StratifiedCrossValidation(int k, LabeledDataSet<T,F,L> Data) 
	{
		super(k, Data);
		this.Data = Data;
		LabelSetSize = Data.getCatSize();
		permutation = getStratifiedSplits();
	}
	
	protected int[] getStratifiedSplits()
	{
		int[] permutation = new int[ k * foldSize ];
		ArrayList<ArrayList<Integer>> listByClass = new ArrayList<ArrayList<Integer>>(LabelSetSize);
		for(int i=0; i<LabelSetSize; i++)
			listByClass.add( new ArrayList<Integer>(totalSize/LabelSetSize) );
		
		for(int i=0; i<Data.Data.size();i++)
		{
			LabeledDatum<F,L> d = Data.Data.get(i);
			int index = Data.getLabelIndex( d.getLabel() );
			listByClass.get(index).add(i);
		}
		
		int currentIndex = 0;
		while(currentIndex != k * foldSize )
			currentIndex = populateOneOfEachClass(permutation,currentIndex,listByClass);
		
		for(int i=0; i<permutation.length; i++)
			System.out.printf("%d ",permutation[i]);
		System.out.println();
		
		return permutation;
	}

	private int populateOneOfEachClass(int[] permutation, int currentIndex, List<ArrayList<Integer>> listByClass) {
		
		Random rgen = new Random();
		List<Integer> currentClassList;
		for(int i=0; i<listByClass.size();i++)
		{
			int ci = i;
			do{
				currentClassList = listByClass.get(ci);
				ci = rgen.nextInt(listByClass.size());
			}while( currentClassList.size() == 0 );
			
			int randomIndex = rgen.nextInt(currentClassList.size());
			permutation[ currentIndex++ ] = currentClassList.get(randomIndex);
			currentClassList.remove( randomIndex );
		}
		return currentIndex;
	}

}
