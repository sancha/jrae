package util;

import java.util.Map;
import java.util.Set;

public class GenericCounterMap<K, V, X> {

	private MapFactory<V, X> mf;
	private Map<K, GenericCounter<V, X>> counterMap;

	// -----------------------------------------------------------------------

	public GenericCounterMap() {
		this(new MapFactory.HashMapFactory<K, GenericCounter<V,X>>(),
				new MapFactory.HashMapFactory<V, X>());
	}

	public GenericCounterMap(MapFactory<K, GenericCounter<V,X>> outerMF,
			MapFactory<V, X> innerMF) {
		mf = innerMF;
		counterMap = outerMF.buildMap();
	}

	// -----------------------------------------------------------------------

	protected GenericCounter<V,X> ensureCounter(K key) {
		GenericCounter<V,X> valueCounter = counterMap.get(key);
		if (valueCounter == null) {
			valueCounter = new GenericCounter<V,X>(mf);
			counterMap.put(key, valueCounter);
		}
		return valueCounter;
	}

	/**
	 * Returns the keys that have been inserted into this CounterMap.
	 */
	public Set<K> keySet() {
		return counterMap.keySet();
	}

	/**
	 * Sets the count for a particular (key, value) pair.
	 */
	public void setValue(K key1, V key2, X value) {
		GenericCounter<V,X> valueCounter = ensureCounter(key1);
		valueCounter.setValue(key2, value);
	}

	/**
	 * Gets the count of the given (key, value) entry, or zero if that entry is
	 * not present. Does not create any objects.
	 */
	public X getValue(K key1, V key2) {
		GenericCounter<V,X> valueCounter = ensureCounter(key1);
		return valueCounter.getValue(key2);
	}

	/**
	 * Gets the sub-counter for the given key. If there is none, a counter is
	 * created for that key, and installed in the CounterMap. You can, for
	 * example, add to the returned empty counter directly (though you
	 * shouldn't). This is so whether the key is present or not, modifying the
	 * returned counter has the same effect (but don't do it).
	 */
	public GenericCounter<V,X> getCounter(K key) {
		return ensureCounter(key);
	}

	/**
	 * Returns the total number of (key, value) entries in the CounterMap (not
	 * their total counts).
	 */
	public int totalSize() {
		int total = 0;
		for (Map.Entry<K, GenericCounter<V,X>> entry : counterMap.entrySet()) {
			GenericCounter<V,X> counter = entry.getValue();
			total += counter.size();
		}
		return total;
	}

	/**
	 * The number of keys in this CounterMap (not the number of key-value
	 * entries -- use totalSize() for that)
	 */
	public int size() {
		return counterMap.size();
	}

	/**
	 * True if there are no entries in the CounterMap (false does not mean
	 * totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("[\n");
		for (Map.Entry<K, GenericCounter<V,X>> entry : counterMap.entrySet()) {
			sb.append("  ");
			sb.append(entry.getKey());
			sb.append(" -> ");
			sb.append(entry.getValue());
			sb.append("\n");
		}
		sb.append("]");
		return sb.toString();
	}
}
