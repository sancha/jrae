package main;

import io.LabeledDataSet;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.List;

import org.jblas.DoubleMatrix;

import math.DifferentiableFunction;
import math.DifferentiableMatrixFunction;
import math.Minimizer;
import math.Norm1Tanh;
import math.QNMinimizer;

import classify.Accuracy;
import classify.ClassifierTheta;
import classify.LabeledDatum;
import classify.ReviewDatum;
import classify.SoftmaxClassifier;

import rae.FineTunableTheta;
import rae.RAECost;
import rae.RAEFeatureExtractor;
import rae.LabeledRAETree;
import rae.RAENode;
import util.ArraysHelper;

public class RAEBuilder {
	FineTunableTheta InitialTheta;
	RAEFeatureExtractor FeatureExtractor;
	DifferentiableMatrixFunction f;

	public RAEBuilder() {
		InitialTheta = null;
		FeatureExtractor = null;
		f = new Norm1Tanh();
	}

	public static void main(final String[] args) throws Exception {
		RAEBuilder rae = new RAEBuilder();

		Arguments params = new Arguments();
		params.parseArguments(args);
		if (params.exitOnReturn)
			return;

		System.out.printf("DictionarySize : %d\nEmbeddingSize : %d\n",
				params.DictionarySize, params.hiddenSize);

		if (params.TrainModel) {
			System.out.println("Training the RAE. Model file will be saved in "
					+ params.ModelFile);
			FineTunableTheta tunedTheta = rae.train(params);			
			tunedTheta.Dump(params.ModelFile);
			
			System.out.println("RAE trained. The model file is saved in "
					+ params.ModelFile);

			RAEFeatureExtractor fe = new RAEFeatureExtractor(
					params.EmbeddingSize, tunedTheta, params.AlphaCat,
					params.Beta, params.CatSize, params.Dataset.Vocab.size(),
					rae.f);
			
			List<LabeledRAETree> Trees = fe.getRAETrees (params.Dataset.Data);
			List<LabeledDatum<Double, Integer>> classifierTrainingData = fe.extractFeaturesIntoArray(Trees);

			SoftmaxClassifier<Double, Integer> classifier = new SoftmaxClassifier<Double, Integer>();
			Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
			System.out.println("Train Accuracy :" + TrainAccuracy.toString());
			
			System.out
					.println("Classifier trained. The model file is saved in "
							+ params.ClassifierFile);
			classifier.Dump(params.ClassifierFile);

			if (params.featuresOutputFile != null)
				rae.DumpFeatures(params.featuresOutputFile,
						classifierTrainingData);

			if (params.ProbabilitiesOutputFile != null)
				rae.DumpProbabilities(params.ProbabilitiesOutputFile,
						classifier.getTrainScores());

			if (params.TreeDumpDir != null)
				rae.DumpTrees(Trees, params.TreeDumpDir, params.Dataset, params.Dataset.Data);

		} else {
			System.out.println
					("Using the trained RAE. Model file retrieved from " + params.ModelFile
					+ "\nNote that this overrides all RAE specific arguments you passed.");

			FineTunableTheta tunedTheta = rae.loadRAE(params);
			assert tunedTheta.getNumCategories() == params.Dataset.getCatSize();

			SoftmaxClassifier<Double, Integer> classifier = rae.loadClassifier(params);

			RAEFeatureExtractor fe = new RAEFeatureExtractor(
					params.EmbeddingSize, tunedTheta, params.AlphaCat,
					params.Beta, params.CatSize, params.Dataset.Vocab.size(),
					rae.f);
			
			if (params.Dataset.Data.size() > 0) {
				System.err.println("There is training data in the directory.");
				System.err.println("It will be ignored when you are not in the training mode.");
			}

			List<LabeledRAETree> testTrees = fe.getRAETrees (params.Dataset.TestData);
			List<LabeledDatum<Double, Integer>> classifierTestingData = fe.extractFeaturesIntoArray(testTrees);

			Accuracy TestAccuracy = classifier.test(classifierTestingData);
			if (params.isTestLabelsKnown) {
				System.out.println("Test Accuracy : " + TestAccuracy);
			}

			if (params.featuresOutputFile != null)
				rae.DumpFeatures(params.featuresOutputFile,
						classifierTestingData);

			if (params.ProbabilitiesOutputFile != null)
				rae.DumpProbabilities(params.ProbabilitiesOutputFile,
						classifier.getTestScores());
			
			if (params.TreeDumpDir != null)
				rae.DumpTrees(testTrees, params.TreeDumpDir, params.Dataset, params.Dataset.TestData);			
		}
	}

	private void DumpTrees( List<LabeledRAETree> trees, String treeDumpDir,
			LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> dataset,
			List<LabeledDatum<Integer, Integer>> data) throws Exception {
		
		if (trees.size () != data.size())
			throw new Exception ("Inconsistent data!");
			
		File treeStructuresFile = new File (treeDumpDir, "treeStructures.txt");
		PrintStream treeStructuresStream = new PrintStream(treeStructuresFile);
		
		for (int i=0; i<trees.size(); i++)
		{
			LabeledRAETree tree = trees.get(i);
			ReviewDatum datum = (ReviewDatum) data.get(i);
			int[] parentStructure = tree.getStructureString();
			
			treeStructuresStream.println(ArraysHelper.makeStringFromIntArray(parentStructure));
			File vectorsFile = new File (treeDumpDir, "sent"+(i+1)+"_nodeVecs.txt");
			PrintStream vectorsStream = new PrintStream(vectorsFile);
			
			File substringsFile = new File (treeDumpDir, "sent"+(i+1)+"_strings.txt");
			PrintStream substringsStream = new PrintStream(substringsFile);
			
			File classifierOutputFile = new File (treeDumpDir, "sent"+(i+1)+"_classifierOutput.txt");
			PrintStream classifierOutputStream = new PrintStream(classifierOutputFile);
			
			for (RAENode node : tree.getNodes())
			{
				double[] features = node.getFeatures();
				double[] scores = node.getScores();
				List<Integer> subTreeWords = node.getSubtreeWordIndices();
				
				String subTreeString = subTreeWords.size() + " ";
				for (int pos : subTreeWords)
					subTreeString += datum.getToken(pos) + " ";
				
				vectorsStream.println(ArraysHelper.makeStringFromDoubleArray(features));
				classifierOutputStream.println(ArraysHelper.makeStringFromDoubleArray(scores));
				substringsStream.println(subTreeString);
			}
			
			vectorsStream.close();
			classifierOutputStream.close();
			substringsStream.close();
		}
	}

	public void DumpFeatures(String featuresOutputFile,
			List<LabeledDatum<Double, Integer>> Features)
			throws FileNotFoundException {

		PrintStream out = new PrintStream(featuresOutputFile);
		for (LabeledDatum<Double, Integer> data : Features) {
			Collection<Double> features = data.getFeatures();
			for (Double f : features)
				out.printf("%.8f ", f.doubleValue());
			out.println();
		}
		out.close();
	}

	public void DumpProbabilities(String ProbabilitiesOutputFile,
			DoubleMatrix classifierScores) throws IOException {

		PrintStream out = new PrintStream(ProbabilitiesOutputFile);
		for (int dataIndex = 0; dataIndex < classifierScores.columns; dataIndex++) {
			// params.Dataset.getLabelString(l.intValue())
			for (int classNum = 0; classNum < classifierScores.rows; classNum++)
				out.printf("%d : %.3f, ", classNum, classifierScores.get(
						classNum, dataIndex));
			out.println();
		}
		out.close();
	}
	
	private FineTunableTheta train(Arguments params) throws IOException,
			ClassNotFoundException {

		InitialTheta = new FineTunableTheta(params.EmbeddingSize,
				params.EmbeddingSize, params.CatSize, params.DictionarySize,
				true);

		FineTunableTheta tunedTheta = null;

		RAECost RAECost = new RAECost(params.AlphaCat, params.CatSize,
				params.Beta, params.DictionarySize, params.hiddenSize,
				params.visibleSize, params.Lambda, InitialTheta.We,
				params.Dataset.Data, null, f);

		Minimizer<DifferentiableFunction> minFunc = new QNMinimizer(10,
				params.MaxIterations);

		double[] minTheta = minFunc.minimize(RAECost, 1e-6, InitialTheta.Theta,
				params.MaxIterations);

		tunedTheta = new FineTunableTheta(minTheta, params.hiddenSize,
				params.visibleSize, params.CatSize, params.DictionarySize);

		// Important step
		tunedTheta.setWe(tunedTheta.We.add(InitialTheta.We));
		return tunedTheta;
	}

	private FineTunableTheta loadRAE(Arguments params) throws IOException,
			ClassNotFoundException {
		FineTunableTheta tunedTheta = null;
		FileInputStream fis = new FileInputStream(params.ModelFile);
		ObjectInputStream ois = new ObjectInputStream(fis);
		tunedTheta = (FineTunableTheta) ois.readObject();
		ois.close();
		return tunedTheta;
	}
	
	private SoftmaxClassifier<Double,Integer> loadClassifier 
			(Arguments params) throws IOException, ClassNotFoundException {
		SoftmaxClassifier<Double,Integer> classifier = null;
		FileInputStream fis = new FileInputStream(params.ClassifierFile);
		ObjectInputStream ois = new ObjectInputStream(fis);
		ClassifierTheta ClassifierTheta = (ClassifierTheta) ois.readObject();
		ois.close();
		classifier = new SoftmaxClassifier<Double, Integer>
					(ClassifierTheta, params.Dataset.getLabelSet());
		return classifier;
	}
}
