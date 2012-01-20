package io;

import java.util.*;

import classify.Datum;

public class DataSet<T extends Datum<F>,F> {
	public static final int GOOD = 1, BAD = 0;
	public static final String UNK = "*UNKNOWN*";

	public List<T> Data;
	public Set<String> Vocab;
	
	public DataSet()
	{
		Data = new ArrayList<T>();
		Vocab = new HashSet<String>();
	}
	
	public DataSet(int capacity)
	{
		Data = new ArrayList<T>(capacity);
		Vocab = new HashSet<String>();
	}
	
	public DataSet(int capacity, Set<String> Vocab)
	{
		this(capacity);
		this.Vocab = Vocab;
	}
	
	public boolean add(T Datum)
	{
		Data.add(Datum);
		return true;
	}
}
