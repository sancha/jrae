package util;

import java.util.*;

/**
 * Concatenates an iterator over iterators into one long iterator.
 *
 * @author Dan Klein
 */
public class ConcatenationIterator<E> implements Iterator<E> {

  Iterator<Iterator<E>> sourceIterators;
  Iterator<E> currentIterator;
  Iterator<E> lastIteratorToReturn;

  public boolean hasNext() {
    if (currentIterator.hasNext())
      return true;
    return false;
  }

  public E next() {
    if (currentIterator.hasNext()) {
      while (true) {
        try {
          E e = currentIterator.next();
          lastIteratorToReturn = currentIterator;
          advance();
          return e;
        } catch (Exception e) {
          System.err.println("bad file");
          if (!sourceIterators.hasNext()) {
            throw new NoSuchElementException();
          }
          currentIterator = sourceIterators.next();        
        }
      }
    }
    throw new NoSuchElementException();
  }

  private void advance() {
    while (! currentIterator.hasNext() && sourceIterators.hasNext()) {
      currentIterator = sourceIterators.next();
    }
  }

  public void remove() {
    if (lastIteratorToReturn == null)
      throw new IllegalStateException();
    currentIterator.remove();
  }

  public ConcatenationIterator(Iterator<Iterator<E>> sourceIterators) {
    this.sourceIterators = sourceIterators;
    this.currentIterator = (new ArrayList<E>()).iterator();
    this.lastIteratorToReturn = null;
    advance();
  }

  public ConcatenationIterator(Collection<Iterator<E>> iteratorCollection) {
    this(iteratorCollection.iterator());
  }
}
