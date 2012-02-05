package util;

/**
 * A generic-typed tiplet of objects.
 * 
 * @author Paul Baumstarck
 */
public class Triplet<F, S, T> {
	F first;
	S second;
	T third;

	public F getFirst() {
		return first;
	}

	public S getSecond() {
		return second;
	}

	public T getThird() {
		return third;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof Triplet<?,?,?>))
			return false;

		final Triplet<?,?,?> triplet = (Triplet<?,?,?>) o;

		if (first != null ? !first.equals(triplet.first)
				: triplet.first != null)
			return false;
		if (second != null ? !second.equals(triplet.second)
				: triplet.second != null)
			return false;
		if (third != null ? !third.equals(triplet.third)
				: triplet.third != null)
			return false;

		return true;
	}

	@Override
	public int hashCode() {
		int result;
		result = (first != null ? first.hashCode() : 0);
		result = 29 * result + (second != null ? second.hashCode() : 0);
		result = 37 * result + (second != null ? second.hashCode() : 0);
		return result;
	}

	@Override
	public String toString() {
		return "(" + getFirst() + ", " + getSecond() + ", " + getThird() + ")";
	}

	public Triplet(F first, S second, T third) {
		this.first = first;
		this.second = second;
		this.third = third;
	}
}