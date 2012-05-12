package rae;

import io.LabeledDataSet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.locks.*;
import math.DifferentiableMatrixFunction;

import org.jblas.*;
import util.ArraysHelper;
import classify.*;
import parallel.*;

public class RAEFeatureExtractor {
	int HiddenSize, CatSize, DictionaryLength;
	double AlphaCat, Beta;
	FineTunableTheta Theta;
	RAEPropagation Propagator;
	DoubleMatrix features;
	Lock lock;
	DifferentiableMatrixFunction f;

	public RAEFeatureExtractor(int HiddenSize, FineTunableTheta Theta, double AlphaCat, double Beta, int CatSize,
			int DictionaryLength, DifferentiableMatrixFunction f) {
		this.HiddenSize = HiddenSize;
		this.Theta = Theta;
		this.AlphaCat = AlphaCat;
		this.Beta = Beta;
		this.HiddenSize = HiddenSize;
		this.CatSize = CatSize;
		this.DictionaryLength = DictionaryLength;
		this.f = f;
		lock = new ReentrantLock();
		Propagator = new RAEPropagation(AlphaCat, Beta, HiddenSize, CatSize, DictionaryLength, f);
	}

	class TreeDumpThread extends Thread {
		LabeledRAETree tree;
		String treeDumpDir;
		final LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> dataset;
		ReviewDatum datum;
		int index;

		public TreeDumpThread(LabeledRAETree tree, String treeDumpDir,
				LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> dataset, ReviewDatum datum, int index) {
			this.tree = tree;
			this.treeDumpDir = treeDumpDir;
			this.dataset = dataset;
			this.datum = datum;
			this.index = index;
			start();
		}

		public void run() {
			try {
				File vectorsFile = new File(treeDumpDir, "sent" + (index) + "_nodeVecs.txt");
				PrintStream vectorsStream = new PrintStream(vectorsFile);

				File substringsFile = new File(treeDumpDir, "sent" + (index) + "_strings.txt");
				PrintStream substringsStream = new PrintStream(substringsFile);

				File classifierOutputFile = new File(treeDumpDir, "sent" + (index) + "_classifierOutput.txt");
				PrintStream classifierOutputStream = new PrintStream(classifierOutputFile);

				for (RAENode node : tree.getNodes()) {
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
			} catch (Exception e) {
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}

	}

	public List<LabeledDatum<Double, Integer>> extractFeaturesIntoArray(
			final LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> dataset,
			final List<LabeledDatum<Integer, Integer>> Data, final String treeDumpDir) {
		final int numExamples = Data.size();
		final LabeledDatum<Double, Integer>[] DataFeatures = new ReviewFeatures[numExamples];
		final boolean dump = (dataset != null && treeDumpDir != null);
		try {
			File treeStructuresFile = new File(treeDumpDir, "treeStructures.txt");
			FileWriter treeStructuresFileWriter = new FileWriter(treeStructuresFile.getAbsolutePath(), true);
			final BufferedWriter treeStructuresStream = new BufferedWriter(treeStructuresFileWriter);

			/*** Actual feature extraction and dumping ***/
			Parallel.For(Data, new Parallel.Operation<LabeledDatum<Integer, Integer>>() {
				@Override
				public void perform(int index, LabeledDatum<Integer, Integer> data) {
					LabeledRAETree tree = getRAETree(Propagator, data);
					if (dump)
						writeTree(treeStructuresStream, tree, treeDumpDir, dataset, (ReviewDatum) data, index);
					double[] feature = tree.getFeaturesVector();
					lock.lock();
					{
						ReviewFeatures r = new ReviewFeatures(null, data.getLabel(), index, feature);
						DataFeatures[index] = r;
					}
					lock.unlock();
				}
			});
			System.gc();
			
			treeStructuresStream.close();
			treeStructuresFileWriter.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}

		return Arrays.asList(DataFeatures);
	}

	protected void writeTree(BufferedWriter treeStructuresStream, LabeledRAETree tree, String treeDumpDir,
			LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> dataset, ReviewDatum data, int index) {
		int[] parentStructure = tree.getStructureString();
		try {
			treeStructuresStream.write(ArraysHelper.makeStringFromIntArray(parentStructure) + "\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
		new TreeDumpThread(tree, treeDumpDir, dataset, data, index);
	}

	public List<LabeledDatum<Double, Integer>> extractFeaturesIntoArray(final List<LabeledDatum<Integer, Integer>> data) {
		return extractFeaturesIntoArray(null, data, null);
	}

	public DoubleMatrix extractFeatures(List<LabeledDatum<Integer, Integer>> Data) {
		int numExamples = Data.size();
		features = DoubleMatrix.zeros(2 * HiddenSize, numExamples);

		Parallel.For(Data, new Parallel.Operation<LabeledDatum<Integer, Integer>>() {
			@Override
			public void perform(int index, LabeledDatum<Integer, Integer> data) {
				double[] feature = extractFeatures(Propagator, data);
				lock.lock();
				{
					features.putColumn(index, new DoubleMatrix(feature));
				}
				lock.unlock();
			}
		});

		return features;
	}

	public double[] extractFeatures(LabeledDatum<Integer, Integer> Data) {
		return getRAETree(Propagator, Data).getFeaturesVector();
	}

	public double[] extractFeatures(RAEPropagation Propagator, LabeledDatum<Integer, Integer> Data) {
		return getRAETree(Propagator, Data).getFeaturesVector();
	}

	public List<LabeledRAETree> getRAETrees(List<LabeledDatum<Integer, Integer>> Data) {
		int numExamples = Data.size();
		final LabeledRAETree[] ExtractedTrees = new LabeledRAETree[numExamples];

		Parallel.For(Data, new Parallel.Operation<LabeledDatum<Integer, Integer>>() {
			@Override
			public void perform(int index, LabeledDatum<Integer, Integer> data) {
				LabeledRAETree tree = getRAETree(Propagator, data);
				lock.lock();
				{
					ExtractedTrees[index] = tree;
				}
				lock.unlock();
			}
		});
		return Arrays.asList(ExtractedTrees);
	}

	public LabeledRAETree getRAETree(RAEPropagation Propagator, LabeledDatum<Integer, Integer> data) {
		int SentenceLength = data.getFeatures().size();

		if (SentenceLength == 0)
			System.err.println("Zero length data");

		int[] wordIndices = ArraysHelper.getIntArray(data.getFeatures());

		DoubleMatrix WordsEmbedded = Theta.We.getColumns(wordIndices);
		int CurrentLabel = data.getLabel();

		LabeledRAETree tree = Propagator.ForwardPropagate(Theta, WordsEmbedded, null, CurrentLabel, SentenceLength);

		tree = Propagator.ForwardPropagate(Theta, WordsEmbedded, null, CurrentLabel, SentenceLength, tree);

		return tree;
	}
}
