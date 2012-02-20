package classify;

/**
 * LabeledDatums add a label to the basic Datum interface.
 * @author Dan Klein
 */
public interface LabeledDatum<F,L> extends Datum<F> {
  L getLabel();
  
}
