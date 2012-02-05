package classify;


import java.util.List;

import classify.LabeledDatum;

/**
 * A minimal implementation of a labeled datum, wrapping a list of features and
 * a label.
 *
 * @author Dan Klein
 */
public class BasicLabeledDatum <F,L> implements LabeledDatum<F, L> {
  L label;
  List<F> features;

  public L getLabel() {
    return label;
  }

  public List<F> getFeatures() {
    return features;
  }

  @Override
public String toString() {
    return "<" + getLabel() + " : " + getFeatures().toString() + ">";
  }

  public BasicLabeledDatum(L label, List<F> features) {
    this.label = label;
    this.features = features;
  }
}
