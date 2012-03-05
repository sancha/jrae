package io;

import java.util.*;
import util.Counter;
import classify.Datum;

public class DataSet<T extends Datum<F>,F> {
	public static final int GOOD = 1, BAD = 0;
	public static final String UNK = "*UNKNOWN*";
	
	public static int MINCOUNT = 5;

	public List<T> Data;
	public Counter<String> Vocab;
	
	public DataSet()
	{
		Data = new ArrayList<T>();
		Vocab = new Counter<String>();
	}
	
	public DataSet(int capacity)
	{
		Data = new ArrayList<T>(capacity);
		Vocab = new Counter<String>();
	}
	
	public boolean add(T Datum)
	{
		Data.add(Datum);
		return true;
	}
}
