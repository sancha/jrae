package main;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Random;


import io.LabeledDataSet;
import classify.LabeledDatum;
import classify.ReviewDatum;

public class CurriculumLearning {
  LabeledDataSet<LabeledDatum<Integer,Integer>, Integer, Integer> Dataset;
  Random rgen;
  int maxInd;
  
  public CurriculumLearning(
      LabeledDataSet<LabeledDatum<Integer,Integer>, Integer, Integer> Dataset){
    this.Dataset = Dataset;
    maxInd = Dataset.Data.size() - 1;
    rgen = new Random();
  }
  
  public List<LabeledDatum<Integer,Integer>> getNGrams(int n, int count){
    List<LabeledDatum<Integer,Integer>> Data = null;
    LabeledDatum<Integer,Integer> inputDatum = null;
    Data = new ArrayList<LabeledDatum<Integer,Integer>>(count);
    for(int i=0; i<2*count && Data.size() < count; i++)
    {
      LabeledDatum<Integer,Integer> ngram = null;
      int randpos = rgen.nextInt(maxInd);
      inputDatum = Dataset.Data.get(randpos);
      ngram = getRandomNgram(n,inputDatum, Data.size());
      if (ngram != null)
        Data.add(ngram);
    }
    return Data;
  }
  
  private LabeledDatum<Integer,Integer> 
    getRandomNgram(int nmax, LabeledDatum<Integer,Integer> datum, int index){
    
    int n = nmax <=2 ? nmax : rgen.nextInt(nmax-2)+2;
    Collection<Integer> features = datum.getFeatures();
    int length = features.size(); 
    if(length < n)
      return null;
    
    int startPos = Math.max(0, rgen.nextInt(length) - n + 1);
    int cpos = 0, i = 0;
    int[] words = new int[n];
    Iterator<Integer> itr = features.iterator();
    
    while (i<n) {
      assert itr.hasNext();
      if (cpos >= startPos){
        words[i] = itr.next();
        i++;
      }
      cpos++;
    }
    
    return new ReviewDatum(words, datum.getLabel(), index);
  }
}
