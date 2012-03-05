package io;

import java.io.*;
import java.util.*;

import util.Counter;
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
	private boolean testLabelsKnown;
	
	public ParsedReviewData()
	{
		counter = 0;
		testCounter = 0;
		WordsIndexer = new HashMap<String, Integer>();
		SecretLabelMapping = new HashMap<Integer, String>();
	}
	
	public ParsedReviewData(int minCount)
	{
		this();
		MINCOUNT = minCount;
	}
	
	public ParsedReviewData(String DirectoryName, int minCount, String wordmapFile) 
						throws IOException{
		this(minCount);
		
		File Dir = new File(DirectoryName);
		String[] files = Dir.list( );
		
		if (wordmapFile == null)
			wordmapFile = new File(DirectoryName,"wordmap.map").getAbsolutePath();
		
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
		
		Sentence_lengths = new int[ NumExamples ];
		Data = new ArrayList<LabeledDatum<Integer,Integer>>( NumExamples );
		TestData = new ArrayList<LabeledDatum<Integer,Integer>>( NumTestExamples );
		
		System.out.println(NumTestExamples + " " + NumExamples);
		
		int CurrentLabel = 0;
		Arrays.sort(files);
		for(String fileName : files )
		{
			String FullFileName = (new File(Dir,fileName)).getAbsolutePath();
			if (isFile(FullFileName) && !isTestDataFile(FullFileName))
			{
				LoadFile(FullFileName,CurrentLabel);
				labelSet.put(CurrentLabel,CurrentLabel);
				SecretLabelMapping.put(CurrentLabel,getLabelString(fileName));
				CurrentLabel++;
			}
		}
		
		
		Set<Integer> testLabelsSeen = new HashSet<Integer>();
		for(String fileName : files )
		{
			String FullFileName = (new File(Dir,fileName)).getAbsolutePath();
			if (isTestDataFile(FullFileName))
			{
				CurrentLabel = getTestLabel(FullFileName);
				testLabelsSeen.add(CurrentLabel);
				LoadTestFile(FullFileName,CurrentLabel);
			}
		}
		
		if (testLabelsSeen.size () > 1)
			testLabelsKnown = true;
		
		EmbedWords();
		
		try {
			// Create file
			FileWriter fstream = new FileWriter(wordmapFile);
			BufferedWriter out = new BufferedWriter(fstream);
			
			for(String word : WordsIndexer.keySet())
				out.write(word + " " + WordsIndexer.get(word));
			
			out.close();
		} catch (Exception e) {// Catch exception if any
			System.err.println("Could not write the wordmap file.");
			System.err.println("Error: " + e.getMessage());
		}
		
		printComplete();
	}
	
	private void printComplete()
	{
		System.out.println("Data has been completely loaded.\n"
			+"There are " + counter + " training examples and "+testCounter+ " test examples.\n"
			+"There are a total of "+WordsIndexer.size()+" elements in the vocabulary."
			+"Note that we only count words that occur atleast "+ MINCOUNT +" number of times. \n"
			+"We maintain an internal mapping of String labels to integers. " 
			+"Class probabilities output are in increasing order of this mapping, and it is as follows\n\n{");
		
		for(int label=0; label < labelSet.size(); label++)
			System.out.println("\t" + label + " : " + getLabelString(label));
		
		System.out.println("}");
	}
	
	private void LoadFile(String FileName, int Label) throws IOException
	{
		assert Label != ReviewDatum.UnknownLabel;
		
		BufferedReader inBr = new BufferedReader(new FileReader(FileName));
		int iCountLines = 0;
		String sLine;
		while ((sLine = inBr.readLine())!=null) {
			sLine = sLine.trim();
			String sWords[] = sLine.split(" ");
			if (sWords == null || sWords.length == 0)
				continue;
			Data.add( new ReviewDatum(sLine, Label, counter) );
			Vocab.addAll( sWords );
			Sentence_lengths[counter] = sWords.length;
			counter++;	
			iCountLines++;
		}
		inBr.close();
		System.out.println(iCountLines+" train examples read from " + FileName);
	}
	
	private void LoadTestFile(String FileName, int Label) throws IOException
	{
		BufferedReader inBr = new BufferedReader(new FileReader(FileName));
		int iCountLines = 0;
		String sLine;
		while ((sLine = inBr.readLine())!=null) {
			sLine = sLine.trim();
			String sWords[] = sLine.split(" ");
			if (sWords == null || sWords.length == 0)
				continue;
			
			TestData.add( new ReviewDatum(sLine, Label, testCounter) );
			testCounter++;
			iCountLines++;
		}
		inBr.close();
		System.out.println(iCountLines+" test examples read from " + FileName);		
	}
	
	protected void EmbedWords()
	{
		Counter<String> trimmedVocab = new Counter<String>();
		Vocab.setCount(DataSet.UNK, MINCOUNT + 10);

		// Iterate through Vocab to set up WordsIndex
		int i = 0;
		for(String s : Vocab.keySet())
			if (Vocab.getCount(s) >= MINCOUNT)
			{	
				WordsIndexer.put(s, i);
				trimmedVocab.setCount(s, Vocab.getCount(s));
				i++;
			}
	
		Vocab = trimmedVocab;
		
		// Update Words_Index
		for (LabeledDatum<Integer,Integer> DataItem : Data) 
			((ReviewDatum)DataItem).indexWords(WordsIndexer);

		for (Datum<Integer> DataItem : TestData) 
			((ReviewDatum)DataItem).indexWords(WordsIndexer);
	}
	
	protected int getTestLabel(String fileName)
	{
		assert fileName.length() > 8; //test/_.+/.txt 
		assert isTestDataFile(fileName);
		
		String labelName = getLabelString (fileName);
		return getLabel (labelName);
	}
	
	protected boolean isFile(String fileName){
		return (new File(fileName).isFile() && fileName.endsWith(".txt"));
	}
	
	protected boolean isTestDataFile(String fileName){
		String baseName = new File(fileName).getName();
		return baseName.endsWith(".txt") && baseName.indexOf("test") == 0;
	}
	
	
	protected String getLabelString (String str)
	{
		String baseName = str;
		
		// Could be a filename
		baseName = new File(str).getName();
		int extensionIndex = baseName.indexOf(".txt");

		// better not be an empty label name
		if ( extensionIndex > 0) 
			baseName = baseName.substring(0, extensionIndex);
		
		if(baseName.indexOf("test") == 0)
			baseName = baseName.substring(4);
		
		if(baseName.length() > 0 && baseName.charAt(0) == '_')
			baseName = baseName.substring(1);
		
		return baseName;
	}
	
	public int getLabel(String label)
	{
		for(Integer key : SecretLabelMapping.keySet())
			if (SecretLabelMapping.get(key).equals(label))
				return key;
		return -1;
	}
	
	public boolean isTestLablesKnown()
	{
		return testLabelsKnown;
	}
	
	public String getLabelString (int label)
	{
		return SecretLabelMapping.get(label);
	}
	
	public int getWordIndex(String Word)
	{
		if( Vocab.containsKey(Word) )
			return WordsIndexer.get(Word);
		return WordsIndexer.get( DataSet.UNK );
	}
}
