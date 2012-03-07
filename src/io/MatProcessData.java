package io;

import java.io.*;
import java.util.*;
import util.Counter;
import classify.LabeledDatum;
import classify.ReviewDatum;

public class MatProcessData extends LabeledDataSet<LabeledDatum<Integer,Integer>,Integer,Integer> {
	private Map<Integer,String> WordsIndexer;
	private Map<String,Integer> ReverseIndexer;
	
    public MatProcessData() {
        WordsIndexer = new HashMap<Integer,String>();
        ReverseIndexer = new HashMap<String,Integer>();
    }
    
    public MatProcessData(String DirectoryName) throws IOException {
        this();
        System.out.println("Creating MatProcessData " + DirectoryName);
        Vocab = readVocabFile(DirectoryName + "/wordmap.txt");
        populateData(DirectoryName + "/data.txt");
        
        for (LabeledDatum<Integer,Integer> datum : Data)
        	labelSet.put(datum.getLabel(),datum.getLabel());   
	}

	private void populateData(String DataFileName) throws IOException
	{
		String strLine;
		int NumLines = IOUtils.countLines(DataFileName);
        Data = new ArrayList<LabeledDatum<Integer,Integer>>( NumLines );
        
        DataInputStream in = new DataInputStream(new FileInputStream(DataFileName));
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        int LineCount = 0;
        while ((strLine = br.readLine()) != null) 
        {
			String[] StrIndices = strLine.split(" ");
            int[] Indices = new int[ StrIndices.length-1 ];
            for(int i=1; i < StrIndices.length; i++)
            {
                Indices[ i-1 ] = Integer.parseInt(StrIndices[i]);
                
                if(Indices[i-1] < 0 || Indices[i-1] >= Vocab.size()){
                	System.err.println("Corrupted data " + StrIndices[i] );
                	break;
                }
            }
            int Label = Integer.parseInt( StrIndices[0] );
            Data.add( new ReviewDatum(StrIndices, Label, LineCount, Indices) );
            LineCount++;
        }
        in.close();
	}

	public int getWordIndex(String Word)
	{
		if( Vocab.containsKey(Word) )
			return ReverseIndexer.get(Word);
		return ReverseIndexer.get( DataSet.UNK );
	}
	
	private Counter<String> readVocabFile(String FileName) throws IOException {
        Counter<String> Vocab = new Counter<String>();

        String strLine;
        FileInputStream fstream = new FileInputStream(FileName);
        DataInputStream in = new DataInputStream(fstream);
        BufferedReader br = new BufferedReader(new InputStreamReader(in));

        while ((strLine = br.readLine()) != null)
        {
        	strLine = strLine.trim();
        	int SplitPoint = strLine.indexOf(' ');
        	if( SplitPoint < 0 )
        		System.err.println("Corrupt WordMap: No split point in [" + strLine + "]");
        	int Index = Integer.parseInt( strLine.substring(0,SplitPoint) );
        	String Word = strLine.substring(SplitPoint+1).trim();
        	WordsIndexer.put(Index, Word);
        	ReverseIndexer.put(Word, Index);
        	Vocab.addKey (Word);
        }
        in.close();
        return Vocab;
	}
}
