package classify;

import java.util.List;

/**
 */
public interface ClassifierFactory<F,L> {
  Classifier<F,L> trainClassifier(List<LabeledDatum<F,L>> trainingData);
}
