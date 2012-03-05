package util;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

/**
 * Generic Counter that does not support in/de-crement operations
 * 
 * @author Sanjeev
 * 
 */
public class GenericCounter<E, V> implements Serializable {
	
	private static final long serialVersionUID = -767969689583483814L;
	Map<E, V> entries;

	/**
	 * The elements in the counter.
	 * 
	 * @return set of keys
	 */
	public Set<E> keySet() {
		return entries.keySet();
	}

	/**
	 * The number of entries in the counter (not the total count -- use
	 * totalCount() instead).
	 */
	public int size() {
		return entries.size();
	}

	/**
	 * True if there are no entries in the counter (false does not mean
	 * totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	/**
	 * Returns whether the counter contains the given key. Note that this is the
	 * way to distinguish keys which are in the counter with count zero, and
	 * those which are not in the counter (and will therefore return count zero
	 * from getCount().
	 * 
	 * @param key
	 * @return whether the counter contains the key
	 */
	public boolean containsKey(E key) {
		return entries.containsKey(key);
	}

	/**
	 * Get the value mapped to the element.
	 * 
	 * @param key
	 */
	public V getValue(E key) {
		V value = entries.get(key);
		return value;
	}

	/**
	 * Set the count for the given key, clobbering any previous count.
	 * 
	 * @param key
	 * @param value
	 */
	public void setValue(E key, V value) {
		entries.put(key, value);
	}

	public GenericCounter() {
		this(new MapFactory.HashMapFactory<E, V>());
	}

	public GenericCounter(MapFactory<E, V> mf) {
		entries = mf.buildMap();
	}
}
