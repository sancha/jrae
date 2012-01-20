package classify;

import util.Counter;

/**
 * @author Dan Klein
 */
public interface ProbabilisticClassifier<F,L> extends Classifier<F,L> {
  Counter<L> getProbabilities(Datum<F> datum);
  Counter<L> getLogProbabilities(Datum<F> datum);
}
