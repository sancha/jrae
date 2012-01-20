package classify;

/**
 */
public interface Classifier<F,L> {
  L getLabel(Datum<F> datum);
}
