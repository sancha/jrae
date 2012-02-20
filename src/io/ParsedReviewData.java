package io;

import java.io.*;
import java.util.*;
import classify.Datum;
import classify.LabeledDatum;
import classify.ReviewDatum;

/*
 * Routines for reading in the review data and processing them into Vocab and 
 * vectors into Vocab (Words_Index) for words in each sentence (Sentence_Words)
 * 
 * @author Sanjeev Satheesh, Daniel Seng
 */
public class ParsedReviewData extends LabeledDataSet<LabeledDatum<Integer,Integer>,Integer,Integer>
{ 
	private int NumExamples, NumTestExamples, counter, testCounter;
	private int[] Sentence_lengths;
	private Map<String,Integer> WordsIndexer;
	private Map<Integer,String> SecretLabelMapping;
	
	public ParsedReviewData()
	{
		counter = 0;
		testCounter = 0;
		WordsIndexer = new HashMap<String, Integer>();
		SecretLabelMapping = new HashMap<Integer, String>();
	}
	
	/**
	 * Initialize and populate all data-structures in this class by reading each file 
	 * in the given directory, using each line as a data item and labelled per file.
	 * The labels are created 0-(n-1) in alphabetical order of the strings.
	 * Do not split for cross validation in this class
	 * @param DirectoryName Reads all the files under this directory
	 */
	public ParsedReviewData(String DirectoryName) throws IOException
	{	
		this();
		
		File Dir = new File(DirectoryName);
		String[] files = Dir.list( );
		
		NumExamples = 0;
		NumTestExamples = 0;
		for(String fileName : files )
		{
			String FullFileName = (new File(Dir,fileName)).getAbsolutePath();
			if(isFile(FullFileName))
				if(isTestDataFile(FullFileName))
					NumTestExamples += IOUtils.countLines(FullFileName);
				else
					NumExamples += IOUtils.countLines(FullFileName);
		}
		
		System.out.println("ProcessData : " + NumExamples + " " + NumTestExamples);
		
		Sentence_lengths = new int[ NumExamples ];
		Data = new ArrayList<LabeledDatum<Integer,Integer>>( NumExamples );
		TestData = new ArrayList<LabeledDatum<Integer,Integer>>( NumTestExamples );
		
		int CurrentLabel = 0;
		Arrays.sort(files);
		for(String fileName : files )
		{
			String FullFileName = (new File(Dir,fileName)).getAbsolutePath();
			if (isTestDataFile(FullFileName))
			{
				LoadFile(FullFileName,ReviewDatum.UnknownLabel);
			}
			else if (isFile(FullFileName))
			{
				LoadFile(FullFileName,CurrentLabel);
				labelSet.put(CurrentLabel,CurrentLabel);
				SecretLabelMapping.put(CurrentLabel,fileName);
				CurrentLabel++;
			}
		}
		
		EmbedWords();
	}
	
	private void LoadFile(String FileName,int Label) throws IOException
	{
		BufferedReader inBr = new BufferedReader(new FileReader(FileName));
		int iCountLines = 0;
		String sLine;
		while ((sLine = inBr.readLine())!=null) {
			sLine = sLine.trim();
			String sWords[] = sLine.split(" ");
			if (sWords == null || sWords.length == 0)
				continue;
			if(Label != ReviewDatum.UnknownLabel){ // Test data
				Data.add( new ReviewDatum(sLine, Label, counter) );
				Vocab.addAll( Arrays.asList(sWords) );
				Sentence_lengths[counter] = sWords.length;
				counter++;
			}
			else{
				TestData.add( new ReviewDatum(sLine, Label, testCounter) );
				testCounter++;
			}
			iCountLines++;
		}
		inBr.close();
		System.out.println("A total of "+iCountLines+" read from " + FileName);
	}
	
	protected void EmbedWords()
	{
		Vocab.add(DataSet.UNK);
		System.out.println("There are a total of "+Vocab.size()+" words in Vocab");
		
		// Iterate through Vocab to set up WordsIndex
		Iterator<String> itr = Vocab.iterator();
		int i = 0;
		while( itr.hasNext() ) {
			String s = itr.next();
			WordsIndexer.put(s, i);
			i++;
		}
		
		// Update Words_Index
		for (LabeledDatum<Integer,Integer> DataItem : Data) 
			((ReviewDatum)DataItem).indexWords(WordsIndexer);

		for (Datum<Integer> DataItem : TestData) 
			((ReviewDatum)DataItem).indexWords(WordsIndexer);
	}
	
	protected boolean isFile(String fileName){
		return (new File(fileName).isFile() && fileName.endsWith(".txt"));
	}
	
	protected boolean isTestDataFile(String fileName){
		String baseName = new File(fileName).getName();
		return fileName.endsWith(".txt") && baseName.indexOf("test") == 0;
	}
	
	public String getLabelString (int label)
	{
		return SecretLabelMapping.get(label);
	}
	
	public int getWordIndex(String Word)
	{
		if( Vocab.contains(Word) )
			return WordsIndexer.get(Word);
		return WordsIndexer.get( DataSet.UNK );
	}
}
