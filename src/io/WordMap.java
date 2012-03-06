package io;

import java.io.IOException;
import util.Counter;

public class WordMap extends StringIntegerMap
{
	private static final long serialVersionUID = 4905859108716169193L;

	public WordMap() {

	}

	public WordMap(String fileName) throws IOException{
		loadFile(fileName);
	}

	public WordMap(Counter<String> Vocab, int minCount) {
		int i = 0;
		for (String s : Vocab.keySet())
			if (Vocab.getCount(s) >= minCount) {
				put(s, i);
				i++;
			}
		put (DataSet.UNK, i);
	}

	public Counter<String> getVocab() {
		Counter<String> Vocab = new Counter<String> ();
		Vocab.addAll (keySet ());
		return Vocab;
	}

	public void saveToFile(String wordmapFileName) {
		super.saveToFileName(wordmapFileName);
	}
	
	public int getWordIndex (String word){
		int index = super.getMapping(word);
		return index == -1 ? super.getMapping(DataSet.UNK) : index;
	}
	
	public String getIndexedWord (int index){
		String word = super.getReverseMapping(index);
		return word == null ? DataSet.UNK : word;
	}
	
}
