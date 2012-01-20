package classify;

import io.*;

import java.util.*;

import util.*;

/**
 * The class splits the data passed to it.
 * It is constructed with k and the data.
 */
public class CrossValidation<T extends Datum<F>,F> {
	protected int k, foldSize, totalSize;
	protected DataSet<T,F> Data;
	protected int[] permutation;
	
	/**
	 * Randomly permutes the data set and splits it into k folds
	 **/
	public CrossValidation(int k, DataSet<T,F> Data)
	{
		this.k = k;
		this.Data = Data;
		this.totalSize = Data.Data.size();
		this.foldSize = (int)Math.floor(this.totalSize / k);
		permutation = ArraysHelper.makeArray(0, this.totalSize-1);
		permutation = ArraysHelper.shuffle(permutation);
	}
	
	public CrossValidation(int k, DataSet<T,F> Data, int[] permutation)
	{
		this.k = k;
		this.Data = Data;
		this.totalSize = Data.Data.size();
		this.foldSize = (int)Math.floor(this.totalSize / k);
		this.permutation = permutation; 
	}	
	
	protected void printSetBitmap(Collection<Integer> indices)
	{
		int[] bitmap = new int[ totalSize ];
		for(int ti : indices )
			bitmap[ti] = 1;
				
		for(int bti : bitmap)
			System.out.printf("%d ",bti);
		System.out.println();
	}
	
	protected List<Integer> getTrainingIndices(int foldNumber)
	{
		return getTrainingIndices(foldNumber, k-1);
	}
	
	protected List<Integer> getTrainingIndices(int foldNumber, int numFolds)
	{
		List<Integer> trainingIndices = new ArrayList<Integer>( foldSize * numFolds );
		int i=0;
		while ( trainingIndices.size() < foldSize *numFolds )
		{
			if(i == foldNumber * foldSize )
			{
				i += foldSize;
				continue;
			}
			trainingIndices.add( permutation[i] );
			i++;
		}
		printSetBitmap(trainingIndices);
		return trainingIndices;
	}

	protected List<Integer> getValidationIndices(int foldNumber)
	{
		List<Integer> validationIndices = new ArrayList<Integer>( foldSize );
		int i = foldNumber * foldSize;
		while ( validationIndices.size() < foldSize )
		{
			validationIndices.add( permutation[i] );
			i++;
		}
		printSetBitmap(validationIndices);
		return validationIndices;
	}

	public List<T> getTrainingData(int foldNumber)
	{
		return getTrainingData(foldNumber, k-1);
	}
	
	public List<T> getTrainingData(int foldNumber, int numFolds)
	{
		List<Integer> trainingIndices = getTrainingIndices(foldNumber, numFolds);
		List<T> trainingData = new ArrayList<T>( trainingIndices.size() );
		for(int i : trainingIndices)
			trainingData.add( Data.Data.get(i) );
		return trainingData;
	}

	public List<T> getValidationData(int foldNumber)
	{
		List<Integer> validationIndices = getValidationIndices(foldNumber);
		List<T> validationData = new ArrayList<T>( validationIndices.size() );
		for(int i : validationIndices)
			validationData.add( Data.Data.get(i) );
		return validationData;
	}	
}
