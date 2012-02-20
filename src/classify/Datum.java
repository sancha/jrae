package classify;

import java.util.Collection;

/**
 * A Datum is a collection of features.  This collection may or may not be a
 * list, depending on the implementation.
 */
public interface Datum <F> {	
  Collection<F> getFeatures();
}
