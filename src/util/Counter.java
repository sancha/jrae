package util;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;
import java.util.Collection;

/**
 * A map from objects to doubles.  Includes convenience methods for getting,
 * setting, and incrementing element counts.  Objects not in the counter will
 * return a count of zero.  The counter is backed by a HashMap (unless specified
 * otherwise with the MapFactory constructor).
 *
 * @author Dan Klein
 */
public class Counter <E> implements Serializable {
  
	private static final long serialVersionUID = 1537708639813200431L;
	Map<E, Double> entries;

  /**
   * The elements in the counter.
   *
   * @return set of keys
   */
  public Set<E> keySet() {
    return entries.keySet();
  }

  /**
   * The number of entries in the counter (not the total count -- use totalCount() instead).
   */
  public int size() {
    return entries.size();
  }

  /**
   * True if there are no entries in the counter (false does not mean totalCount > 0)
   */
  public boolean isEmpty() {
    return size() == 0;
  }

  /**
   * Returns whether the counter contains the given key.  Note that this is the
   * way to distinguish keys which are in the counter with count zero, and those
   * which are not in the counter (and will therefore return count zero from
   * getCount().
   *
   * @param key
   * @return whether the counter contains the key
   */
  public boolean containsKey(E key) {
    return entries.containsKey(key);
  }

  /**
   * Get the count of the element, or zero if the element is not in the
   * counter.
   *
   * @param key
   */
  public double getCount(E key) {
    Double value = entries.get(key);
    if (value == null)
      return 0;
    return value;
  }

  /**
   * Set the count for the given key, clobbering any previous count.
   *
   * @param key
   * @param count
   */
  public void setCount(E key, double count) {
    entries.put(key, count);
  }

  /**
   * Increment a key's count by the given amount.
   *
   * @param key
   * @param increment
   */
  public void incrementCount(E key, double increment) {
    setCount(key, getCount(key) + increment);
  }

  /**
   * Increment each element in a given collection by a given amount.
   */
  public void incrementAll(Collection<? extends E> collection, double count) {
    for (E key : collection) {
      incrementCount(key, count);
    }
  }
  
  /**
   * Adds a key to the counter if it does not exist and sets the count to 1.
   * If not, increments the count of the key.
   */
  public void addKey (E key){
	  incrementCount(key,1);
  }
  
  public void addAll (E[] keys){
	  for (E key : keys)
		  addKey(key);
  }
  
  public void addAll (Collection<E> keys){
	  for (E key : keys)
		  addKey(key);
  }
  
  public <T extends E> void incrementAll(Counter<T> counter) {
    for (T key : counter.keySet()) {
      double count = counter.getCount(key);
      incrementCount(key, count);
    }
  }

  /**
   * Finds the total of all counts in the counter.  This implementation iterates
   * through the entire counter every time this method is called.
   *
   * @return the counter's total
   */
  public double totalCount() {
    double total = 0.0;
    for (Map.Entry<E, Double> entry : entries.entrySet()) {
      total += entry.getValue();
    }
    return total;
  }

  /**
   * Finds the key with maximum count.  This is a linear operation, and ties are broken arbitrarily.
   *
   * @return a key with minumum count
   */
  public E argMax() {
    double maxCount = Double.NEGATIVE_INFINITY;
    E maxKey = null;
    for (Map.Entry<E, Double> entry : entries.entrySet()) {
      if (entry.getValue() > maxCount || maxKey == null) {
        maxKey = entry.getKey();
        maxCount = entry.getValue();
      }
    }
    return maxKey;
  }

  /**
   * Returns a string representation with the keys ordered by decreasing
   * counts.
   *
   * @return string representation
   */
  @Override
public String toString() {
    return toString(keySet().size());
  }

  /**
   * Returns a string representation which includes no more than the
   * maxKeysToPrint elements with largest counts.
   *
   * @param maxKeysToPrint
   * @return partial string representation
   */
  public String toString(int maxKeysToPrint) {
    return asPriorityQueue().toString(maxKeysToPrint);
  }

  /**
   * Builds a priority queue whose elements are the counter's elements, and
   * whose priorities are those elements' counts in the counter.
   */
  public PriorityQueue<E> asPriorityQueue() {
    PriorityQueue<E> pq = new PriorityQueue<E>(entries.size());
    for (Map.Entry<E, Double> entry : entries.entrySet()) {
      pq.add(entry.getKey(), entry.getValue());
    }
    return pq;
  }

  public Counter() {
    this(new MapFactory.HashMapFactory<E, Double>());
  }

  public Counter(MapFactory<E, Double> mf) {
    entries = mf.buildMap();
  }

}
