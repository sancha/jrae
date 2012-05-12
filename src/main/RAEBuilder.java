package main;

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

import classify.SoftmaxClassifier;

import rae.FineTunableTheta;
import rae.RAECost;
import rae.RAEFeatureExtractor;

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

      RAEFeatureExtractor fe = new RAEFeatureExtractor(params.EmbeddingSize,
          tunedTheta, params.AlphaCat, params.Beta, params.CatSize,
          params.Dataset.Vocab.size(), rae.f);

      List<LabeledDatum<Double, Integer>> classifierTrainingData = fe
          .extractFeaturesIntoArray(params.Dataset, params.Dataset.Data,
              params.TreeDumpDir);

      SoftmaxClassifier<Double, Integer> classifier = new SoftmaxClassifier<Double, Integer>();
      Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
      System.out.println("Train Accuracy :" + TrainAccuracy.toString());

      System.out.println("Classifier trained. The model file is saved in "
          + params.ClassifierFile);
      classifier.Dump(params.ClassifierFile);

      if (params.featuresOutputFile != null)
        rae.DumpFeatures(params.featuresOutputFile, classifierTrainingData);

      if (params.ProbabilitiesOutputFile != null)
        rae.DumpProbabilities(params.ProbabilitiesOutputFile, classifier
            .getTrainScores());

      // if (params.TreeDumpDir != null)
      // rae.DumpTrees(Trees, params.TreeDumpDir, params.Dataset,
      // params.Dataset.Data);

      System.out.println("Dumping complete");

    } else {
      System.out
          .println("Using the trained RAE. Model file retrieved from "
              + params.ModelFile
              + "\nNote that this overrides all RAE specific arguments you passed.");

      FineTunableTheta tunedTheta = rae.loadRAE(params);
      assert tunedTheta.getNumCategories() == params.Dataset.getCatSize();

      SoftmaxClassifier<Double, Integer> classifier = null;
      try {
        classifier = rae.loadClassifier(params);
      } catch (IOException e) {
        if (params.Dataset.Data.size() == 0)
          throw e;
        System.err.println("Your classifier could not be loaded.");
        System.err.println("Learning the classifier again ...");
        classifier = rae.trainClassifier(params, tunedTheta);
        classifier.Dump(params.ClassifierFile);
      }

      RAEFeatureExtractor fe = new RAEFeatureExtractor(params.EmbeddingSize,
          tunedTheta, params.AlphaCat, params.Beta, params.CatSize,
          params.Dataset.Vocab.size(), rae.f);

      if (params.Dataset.Data.size() > 0) {
        System.err.println("There is training data in the directory.");
        System.err
            .println("It will be ignored when you are not in the training mode.");
      }

      List<LabeledDatum<Double, Integer>> classifierTestingData = fe
          .extractFeaturesIntoArray(params.Dataset, params.Dataset.TestData,
              params.TreeDumpDir);

      Accuracy TestAccuracy = classifier.test(classifierTestingData);
      if (params.isTestLabelsKnown) {
        System.out.println("Test Accuracy : " + TestAccuracy);
      }

      if (params.featuresOutputFile != null)
        rae.DumpFeatures(params.featuresOutputFile, classifierTestingData);

      if (params.ProbabilitiesOutputFile != null)
        rae.DumpProbabilities(params.ProbabilitiesOutputFile, classifier
            .getTestScores());

      // if (params.TreeDumpDir != null)
      // rae.DumpTrees(testTrees, params.TreeDumpDir, params.Dataset,
      // params.Dataset.TestData);
    }

    System.exit(0);
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
        out.printf("%d : %.3f, ", classNum, classifierScores.get(classNum,
            dataIndex));
      out.println();
    }
    out.close();
  }

  private FineTunableTheta train(Arguments params) throws IOException,
      ClassNotFoundException {

    InitialTheta = new FineTunableTheta(params.EmbeddingSize,
        params.EmbeddingSize, params.CatSize, params.DictionarySize, true);

    DoubleMatrix InitialWe = InitialTheta.We.dup();
    
    RAECost RAECost = null;
    FineTunableTheta tunedTheta = null;
    Minimizer<DifferentiableFunction> minFunc = null;
    
    if(params.CurriculumLearning)
      slowTrain(params, InitialTheta, InitialWe);
      
    RAECost = new RAECost(params.AlphaCat, params.CatSize, params.Beta,
        params.DictionarySize, params.hiddenSize, params.visibleSize,
        params.Lambda, InitialWe, params.Dataset.Data, null, f);

    minFunc = new QNMinimizer(10, params.MaxIterations);

    double[] minTheta = minFunc.minimize(RAECost, 1e-6, InitialTheta.Theta,
        params.MaxIterations);

    tunedTheta = new FineTunableTheta(minTheta, params.hiddenSize,
        params.visibleSize, params.CatSize, params.DictionarySize);
  
    
    // Important step
    tunedTheta.setWe(tunedTheta.We.add(InitialWe));
    return tunedTheta;
  }
  
  private FineTunableTheta slowTrain
    (Arguments params, FineTunableTheta tunedTheta, DoubleMatrix InitialWe){
    
    CurriculumLearning slowLearner = new CurriculumLearning(params.Dataset);
    final int MILLION = 10000;
    
    int [] curriculum = new int[]{2,3,4,6,8,10};
    
    RAECost RAECost = null;
    List<LabeledDatum<Integer,Integer>> Data = null;
    Minimizer<DifferentiableFunction> minFunc = null;
    
    for (int ngram : curriculum)
    {
      Data = slowLearner.getNGrams(ngram, MILLION);
      
      System.out.println("SLOW LEARNING : " + ngram + " with " + Data.size() + " data points.");
      
      RAECost = new RAECost(params.AlphaCat, params.CatSize, params.Beta,
          params.DictionarySize, params.hiddenSize, params.visibleSize,
          params.Lambda, InitialWe, Data, null, f);
  
      minFunc = new QNMinimizer(10, params.MaxIterations);
  
      double[] minTheta = minFunc.minimize(RAECost, 1e-6, tunedTheta.Theta,
          params.MaxIterations);
  
      tunedTheta = new FineTunableTheta(minTheta, params.hiddenSize,
          params.visibleSize, params.CatSize, params.DictionarySize);
    
      tunedTheta.setWe(tunedTheta.We.add(InitialWe));
    }
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

  private SoftmaxClassifier<Double, Integer> loadClassifier(Arguments params)
      throws IOException, ClassNotFoundException {
    SoftmaxClassifier<Double, Integer> classifier = null;
    FileInputStream fis = new FileInputStream(params.ClassifierFile);
    ObjectInputStream ois = new ObjectInputStream(fis);
    ClassifierTheta ClassifierTheta = (ClassifierTheta) ois.readObject();
    ois.close();
    classifier = new SoftmaxClassifier<Double, Integer>(ClassifierTheta,
        params.Dataset.getLabelSet());

    return classifier;
  }

  private SoftmaxClassifier<Double, Integer> trainClassifier(Arguments params,
      FineTunableTheta tunedTheta) throws IOException {

    RAEFeatureExtractor fe = new RAEFeatureExtractor(params.EmbeddingSize,
        tunedTheta, params.AlphaCat, params.Beta, params.CatSize,
        params.Dataset.Vocab.size(), f);

    List<LabeledDatum<Double, Integer>> classifierTrainingData = fe
        .extractFeaturesIntoArray(params.Dataset, params.Dataset.Data,
            params.TreeDumpDir);

    SoftmaxClassifier<Double, Integer> classifier = new SoftmaxClassifier<Double, Integer>();
    Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
    System.out.println("Train Accuracy :" + TrainAccuracy.toString());

    System.out.println("Classifier trained. The model file is saved in "
        + params.ClassifierFile);
    classifier.Dump(params.ClassifierFile);
    return classifier;
  }
}
