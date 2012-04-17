package util;

import java.util.Collection;
import java.util.Iterator;

public class Reducer<T extends Reducible<T>> {

	public T reduce (Collection<T> list)
	{
		if (list.size() == 0)
			return null;
		
		T proto = null;
		Iterator<T> itr = list.iterator();
		if (itr.hasNext())
			proto = itr.next();
			
		while (itr.hasNext())
			proto.reduce(itr.next());
		
		list.clear();
		System.gc();
		
		return proto;
	}
}
