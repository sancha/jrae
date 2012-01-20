package util;

/**
 * Convenience Extension of Counter to use an IdentityHashMap.
 *
 * @author Dan Klein
 */
public class IdentityCounter<E> extends Counter<E> {
	private static final long serialVersionUID = -6528773644247880481L;

	public IdentityCounter() {
    super(new MapFactory.IdentityHashMapFactory<E,Double>());
  }
}
