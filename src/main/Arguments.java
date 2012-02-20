package main;

import io.LabeledDataSet;
import io.MatProcessData;
import io.ParsedReviewData;

import java.io.IOException;
import java.util.Map;

import parallel.Parallel;

import util.CommandLineUtils;
import classify.LabeledDatum;

public class Arguments {
	boolean TrainModel = false;
	String dir = "data/parsed/"; //ssanjeev
	String ModelFile = null;
	int NumFolds = 10,	MaxIterations = 80,	EmbeddingSize = 50, CatSize = 1;
	int DictionarySize, hiddenSize, visibleSize;
	double AlphaCat = 0.2, Beta = 0.5;
	double[] Lambda = new double[]{1e-05, 0.0001, 1e-05, 0.01};
	boolean exitOnReturn = false;
	LabeledDataSet<LabeledDatum<Integer,Integer>,Integer,Integer> Dataset = null;
	
	public void parseArguments(String[] args) throws IOException
	{
		Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);
		
		if (argMap.containsKey("-NumFolds")) 
			NumFolds = Integer.parseInt( argMap.get("-NumFolds") )-1;
		
		if (argMap.containsKey("-MaxIterations")) 
			MaxIterations = Integer.parseInt( argMap.get("-MaxIterations") );
		
		if (argMap.containsKey("-embeddingSize")) 
			EmbeddingSize = Integer.parseInt( argMap.get("-embeddingSize") );
		
		if (argMap.containsKey("-alphaCat")) 
			AlphaCat = Double.parseDouble( argMap.get("-alphaCat") );
		
		if (argMap.containsKey("-lambdaW")) 
			Lambda[0] = Double.parseDouble( argMap.get("-lambdaW") );
		
		if (argMap.containsKey("-lambdaL")) 
			Lambda[1] = Double.parseDouble( argMap.get("-lambdaL") );
		
		if (argMap.containsKey("-lambdaCat")) 
			Lambda[2] = Double.parseDouble( argMap.get("-lambdaCat") );
		
		if (argMap.containsKey("-lambdaLRAE")) 
			Lambda[3] = Double.parseDouble( argMap.get("-lambdaLRAE") );
		
		if (argMap.containsKey("-Beta")) 
			Beta = Double.parseDouble( argMap.get("-Beta") );

		if (argMap.containsKey("-TrainModel")) 
			TrainModel = Boolean.parseBoolean( argMap.get("-TrainModel") );		
		
		if (argMap.containsKey("-ModelFile"))
			ModelFile = argMap.get("-ModelFile");
		else{
				exitOnReturn = true;
				printUsage();
		}
		
		if (argMap.containsKey("-NumCores"))
			Parallel.setPoolSize( Integer.parseInt(argMap.get("-NumCores")) );
		
		if (argMap.containsKey("--help") || argMap.containsKey("-h"))
		{
			exitOnReturn = true;
			printUsage();
		}

		if (argMap.containsKey("-ProcessedDataDir"))
		{
			dir = argMap.get("-ProcessedDataDir");
			Dataset = new MatProcessData(dir);
		}
		else if (argMap.containsKey("-DataDir"))
		{
			dir = argMap.get("-DataDir");
			Dataset = new ParsedReviewData(dir);
		}else
			Dataset = new MatProcessData(dir);
		
		CatSize = Dataset.getCatSize()-1;
		DictionarySize = Dataset.Vocab.size();
		hiddenSize = EmbeddingSize;
		visibleSize = EmbeddingSize;		
		
		
		System.out.println(
				"CatSize : " + CatSize + "\n" +
				"DictionarySize : " + DictionarySize  				
		);
		
	}
	
	public void printUsage()
	{
		
	}

}
