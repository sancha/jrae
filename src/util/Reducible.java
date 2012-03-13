package util;

public interface Reducible<T> extends Duplicatable{
	public void reduce (T instance);
}
