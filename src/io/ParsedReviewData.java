package io;

import java.io.*;
import java.util.*;
import edu.stanford.nlp.ie.machinereading.domains.ace.reader.*;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordTokenFactory;

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
	private WordMap WordsIndexer;
	private StringLabelSet labelMapping;
	private boolean testLabelsKnown;
	private RobustTokenizer<Word> tokenizer;
	
	public ParsedReviewData()
	{
		counter = 0;
		testCounter = 0;
		WordsIndexer = new WordMap();
		labelMapping = new StringLabelSet();
	}
	
	public ParsedReviewData(int minCount)
	{
		this();
		MINCOUNT = minCount;
	}
	
	public ParsedReviewData(String DirectoryName, int minCount, 
			String wordmapFileName, String labelsetFileName) throws IOException{
		this(minCount);
		
		File Dir = new File(DirectoryName);
		File labelSetFile = null, wordMapFile = null;
		String[] files = Dir.list( );
		
		wordMapFile = (wordmapFileName == null)?
				new File (DirectoryName,"wordmap.map"): new File (wordmapFileName);

		labelSetFile = (labelsetFileName == null)?
				new File (DirectoryName,"labels.map"): new File (labelsetFileName);
		
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
		
		if (NumExamples == 0 && (!wordMapFile.exists () || !labelSetFile.exists ()) )
			throw new IOException ("If you are not training, then please provide both " +
									"the labels.map and wordmap.map files as input. ");
		
		Sentence_lengths = new int[ NumExamples ];
		Data = new ArrayList<LabeledDatum<Integer,Integer>>( NumExamples );
		TestData = new ArrayList<LabeledDatum<Integer,Integer>>( NumTestExamples );
		
		System.out.println(NumTestExamples + " " + NumExamples);
		
		Arrays.sort(files);
		LoadTrainingData(Dir, files);
		
		if (labelSetFile.exists())
			labelMapping = new StringLabelSet (labelSetFile.getAbsolutePath());
		else
		{
			labelMapping.saveToFile(labelSetFile.getAbsolutePath());
		}	
		
		LoadTestData(Dir, files);
		
		if (wordMapFile.exists())
			WordsIndexer = new WordMap (wordMapFile.getAbsolutePath());			
		else
		{
			WordsIndexer = new WordMap (Vocab, MINCOUNT);
			WordsIndexer.saveToFile(wordMapFile.getAbsolutePath());
		}
		
		EmbedWords();
		
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
	
	private void LoadTrainingData (File Dir, String[] files) throws IOException
	{
		int CurrentLabel = 0;
		Arrays.sort(files);
		for(String fileName : files )
		{
			String FullFileName = (new File(Dir,fileName)).getAbsolutePath();
			if (isFile(FullFileName) && !isTestDataFile(FullFileName))
			{
				LoadTrainFile(FullFileName,CurrentLabel);
				labelSet.put(CurrentLabel,CurrentLabel);
				labelMapping.put(getLabelString(fileName),CurrentLabel);
				CurrentLabel++;
			}
		}	
	}
	
	private void LoadTestData (File Dir, String[] files) throws IOException
	{
		int CurrentLabel = 0;
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
		
	}
	
	private void LoadTrainFile(String FileName, int Label) throws IOException
	{
		assert Label != ReviewDatum.UnknownLabel;
		
		BufferedReader inBr = new BufferedReader(new InputStreamReader(
    new FileInputStream(FileName), "UTF-8"));
		int iCountLines = 0;
		String sLine;
		PTBTokenizer ptbt;
		while ((sLine = inBr.readLine())!=null) {
			sLine = sLine.trim();
			 ptbt = new PTBTokenizer(new StringReader(sLine), new WordTokenFactory(), "");
			// tokenizer = new RobustTokenizer<Word>(sLine);
			List<Word> list = ptbt.tokenize();
			// for (Word word : list){
			// 	System.out.println("Word" + word.word());
			// }

			Word[] words = list.toArray(new Word[list.size()]);
			
			if (words == null || words.length == 0)
				continue;
			Data.add( new ReviewDatum(words, Label, counter) );
			for (Word word : words)
				Vocab.addKey( word.word() );
			
			Sentence_lengths[counter] = words.length;
			counter++;	
			iCountLines++;
		}
		inBr.close();
		System.out.println(iCountLines+" train examples read from " + FileName);
	}
	
	private void LoadTestFile(String FileName, int Label) throws IOException
	{
		BufferedReader inBr = new BufferedReader(new InputStreamReader(
													new FileInputStream(FileName), "UTF-8"));
   		int iCountLines = 0;
		String sLine;
		PTBTokenizer ptbt;

		while ((sLine = inBr.readLine())!=null) {
			sLine = sLine.trim();
			sLine = sLine.trim();

			ptbt = new PTBTokenizer(new StringReader(sLine), new WordTokenFactory(), "");
			
			// Alternatively, could use robust tokenizer for English or some other tokenizer
			// for languages with scripts like Chinese or Arabic
			// tokenizer = new RobustTokenizer<Word>(sLine);
			List<Word> list = ptbt.tokenize();

			Word[] words = list.toArray(new Word[list.size()]);
			
			if (words == null || words.length == 0)
				continue;
			
			TestData.add( new ReviewDatum(words, Label, testCounter) );
			testCounter++;
			iCountLines++;
		}
		inBr.close();
		System.out.println(iCountLines+" test examples read from " + FileName);		
	}
	
	protected void EmbedWords()
	{
		Vocab = WordsIndexer.getVocab();
		
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
	
	public boolean isTestLablesKnown()
	{
		return testLabelsKnown;
	}
	
	public int getLabel(String label)
	{
		return labelMapping.getLabel(label);
	}
	
	public String getLabelString (int label)
	{
		return labelMapping.getLabelString(label);
	}
	
	public int getWordIndex(String Word)
	{
		return WordsIndexer.getMapping(Word);
	}
}
