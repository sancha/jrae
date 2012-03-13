package util;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

public abstract class ArraysHelper{
	
	/**
	 * @param from first element of the returned array
	 * @param to last element of the returned array
	 * @return an array containing elements from, from+1 ... to  == [from:1:to]
	 */
	public static int[] makeArray(int from, int to)
	{		
		int size = to-from+1;
		if( size <= 0 ){
			//System.err.println("from > to : " + from + " " + to);
			return null;
		}
		int[] array = new int[size];
		for(int i=0, j=from; i<size && j <=to; j++, i++)
			array[i] = j;
		return array;
	}
	
	/**
	 * In-place shuffler for integer array.
	 */
	public static int[] shuffle(int[] inp)
	{
		Random rgen = new Random();
		for (int i=0; i<inp.length; i++) 
		{
		    int randomPosition = rgen.nextInt(inp.length);
		    int temp = inp[i];
		    inp[i] = inp[randomPosition];
		    inp[randomPosition] = temp;
		}
		return inp;
	}
	
	public static int[] getIntArray(Collection<Integer> inp)
	{
		int[] baseArray = new int[ inp.size() ];
		int i=0;
		for(Integer item : inp )
		{
			baseArray[i] = item.intValue();
			i++;
		}
		return baseArray;
	}
	
	public static double[] getDoubleArray(Collection<Double> inp)
	{
		double[] baseArray = new double[ inp.size() ];
		int i=0;
		for(Double item : inp )
		{
			baseArray[i] = item.doubleValue();
			i++;
		}
		return baseArray;		
	}
	
	public static String makeStringFromIntArray (int[] inp)
	{
		String str = "";
		for (int i=0; i<inp.length; i++)
			str += inp[i] + " ";
		return str;
	}
	
	public static String makeStringFromDoubleArray (double[] inp)
	{
		NumberFormat formatter = new DecimalFormat("#0.0000");
		String str = "";
		for (int i=0; i<inp.length; i++)
			str += " " + formatter.format(inp[i]);
		return str;
	}
}
