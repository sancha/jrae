package util;

import java.util.List;
import java.util.ArrayList;

/**
 * Utility methods related to Counters and CounterMaps.
 *
 * @author Dan Klein
 */
public class Counters {

  public static <E> Counter<E> normalize(Counter<E> counter) {
    Counter<E> normalizedCounter = new Counter<E>();
    double total = counter.totalCount();
    for (E key : counter.keySet()) {
      normalizedCounter.setCount(key, counter.getCount(key) / total);
    }
    return normalizedCounter;
  }

  public static <K,V> CounterMap<K,V> conditionalNormalize(CounterMap<K,V> counterMap) {
    CounterMap<K,V> normalizedCounterMap = new CounterMap<K,V>();
    for (K key : counterMap.keySet()) {
      Counter<V> normalizedSubCounter = normalize(counterMap.getCounter(key));
      for (V value : normalizedSubCounter.keySet()) {
        double count = normalizedSubCounter.getCount(value);
        normalizedCounterMap.setCount(key, value, count);
      }
    }
    return normalizedCounterMap;
  }

  public static <E> String toBiggestValuesFirstString(Counter<E> c) {
    return c.asPriorityQueue().toString();
  }

  public static <E> String toBiggestValuesFirstString(Counter<E> c, int k) {
    PriorityQueue<E> pq = c.asPriorityQueue();
    PriorityQueue<E> largestK = new PriorityQueue<E>();
    while (largestK.size() < k && pq.hasNext()) {
      double firstScore = pq.getPriority();
      E first = pq.next();
      largestK.add(first, firstScore);
    }
    return largestK.toString();
  }

  public static <E> List<E> sortedKeys(Counter<E> counter) {
    List<E> sortedKeyList = new ArrayList<E>();
    PriorityQueue<E> pq = counter.asPriorityQueue();
    while (pq.hasNext()) {
      sortedKeyList.add(pq.next());
    }
    return sortedKeyList;
  }

}
