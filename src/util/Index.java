package util;

import java.util.*;

/**
 * Maintains a two-way map between a set of objects and contiguous integers from
 * 0 to the number of objects.  Use get(i) to look up object i, and
 * indexOf(object) to look up the index of an object.
 *
 * @author Dan Klein
 */
public class Index <E> extends AbstractList<E> {
  List<E> objects;
  Map<E, Integer> indexes;

  /**
   * Return the object with the given index
   *
   * @param index
   */
  @Override
public E get(int index) {
    return objects.get(index);
  }

  /**
   * Returns the number of objects indexed.
   */
  @Override
public int size() {
    return objects.size();
  }

  /**
   * Returns the index of the given object, or -1 if the object is not present
   * in the indexer.
   *
   * @param o
   */
  @Override
public int indexOf(Object o) {
    Integer index = indexes.get(o);
    if (index == null)
      return -1;
    return index;
  }

  /**
   * Constant time override for contains.
   */
  @Override
public boolean contains(Object o) {
    return indexes.keySet().contains(o);
  }

  /**
   * Add an element to the indexer.  If the element is already in the indexer,
   * the indexer is unchanged (and false is returned).
   *
   * @param e
   */
  @Override
public boolean add(E e) {
    if (contains(e)) return false;
    objects.add(e);
    indexes.put(e, size() - 1);
    return true;
  }

  public Index() {
    objects = new ArrayList<E>();
    indexes = new HashMap<E, Integer>();
  }

  public Index(Collection<? extends E> c) {
    this();
    addAll(c);
  }

}
