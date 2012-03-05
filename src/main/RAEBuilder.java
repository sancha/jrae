package main;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.List;

import org.jblas.DoubleMatrix;

import util.Pair;

import math.DifferentiableFunction;
import math.DifferentiableMatrixFunction;
import math.Minimizer;
import math.Norm1Tanh;
import math.QNMinimizer;

import classify.Accuracy;
import classify.LabeledDatum;
import classify.SoftmaxClassifier;

import rae.FineTunableTheta;
import rae.RAECost;
import rae.RAEFeatureExtractor;
import rae.Tree;

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
			SoftmaxClassifier<Double, Integer> classifier = new SoftmaxClassifier<Double, Integer>();

			Pair<List<LabeledDatum<Double, Integer>>, List<Tree>> processedOutput = fe
					.extractFeaturesIntoArray(params.Dataset.Data);
			List<LabeledDatum<Double, Integer>> classifierTrainingData = processedOutput
					.getFirst();

			Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
			System.out.println("Train Accuracy :" + TrainAccuracy.toString());

			System.out
					.println("Classifier trained. The model file is saved in "
							+ params.ClassifierFile);
			classifier.Dump(params.ClassifierFile);

			if (params.featuresOutputFile != null)
				rae.DumpFeatures(params.featuresOutputFile,
						classifierTrainingData);

			if (params.featuresOutputFile != null)
				rae.DumpProbabilities(params.ProbabilitiesOutputFile,
						classifier.getTrainScores());

			if (params.featuresOutputFile != null)
				rae.DumpTrees(params.TreeDumpDir, processedOutput.getSecond());

		} else {
			System.out
					.println("Using the trained RAE. Model file retrieved from "
							+ params.ModelFile
							+ "\nNote that this overrides all RAE specific arguments you passed.");

			List<LabeledDatum<Double, Integer>> classifierTestingData = null;

			FineTunableTheta tunedTheta = rae.load(params);
			assert tunedTheta.getNumCategories() == params.Dataset.getCatSize();

			SoftmaxClassifier<Double, Integer> classifier = new SoftmaxClassifier<Double, Integer>(
					params.ClassifierFile);

			RAEFeatureExtractor fe = new RAEFeatureExtractor(
					params.EmbeddingSize, tunedTheta, params.AlphaCat,
					params.Beta, params.CatSize, params.Dataset.Vocab.size(),
					rae.f);

			if (params.Dataset.Data.size() > 0) {
				System.err.println("There is training data in the directory.");
				System.err
						.println("It will be ignored when you are not in the training mode.");
			}

			Pair<List<LabeledDatum<Double, Integer>>, List<Tree>> processedOutput = fe
					.extractFeaturesIntoArray(params.Dataset.TestData);

			classifierTestingData = processedOutput.getFirst();

			Accuracy TestAccuracy = classifier.test(classifierTestingData);
			if (params.isTestLabelsKnown) {
				System.out.println("Test Accuracy : " + TestAccuracy);
			}

			if (params.featuresOutputFile != null)
				rae.DumpFeatures(params.featuresOutputFile,
						classifierTestingData);

			if (params.featuresOutputFile != null)
				rae.DumpProbabilities(params.ProbabilitiesOutputFile,
						classifier.getTestScores());
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

	public void DumpTrees(String TreesDumpDirectory, List<Tree> trees)
			throws Exception {
		throw new Exception("Dumping trees not implemented yet!");
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

	private FineTunableTheta load(Arguments params) throws IOException,
			ClassNotFoundException {
		FineTunableTheta tunedTheta = null;
		FileInputStream fis = new FileInputStream(params.ModelFile);
		ObjectInputStream ois = new ObjectInputStream(fis);
		tunedTheta = (FineTunableTheta) ois.readObject();
		ois.close();
		return tunedTheta;
	}
}
