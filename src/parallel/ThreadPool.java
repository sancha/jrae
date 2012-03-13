package parallel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import util.Reducible;
import util.Reducer;
import util.Pair;

public class ThreadPool {
	private static final int NUM_CORES = Runtime.getRuntime().availableProcessors();
	private static int poolSize = NUM_CORES; 
	
	public static void setPoolSize(int poolSize)
	{
		ThreadPool.poolSize = poolSize;
	}
	
//	@SuppressWarnings("unchecked")
	public static <T,E extends Reducible<E>,F> E mapReduce
		(final Collection<T> pElements, final E Operator, final Operation<E,T> pOperation) {
		
	    final LinkedList<E> queue = new LinkedList<E>();
	    for (int i=0; i<ThreadPool.poolSize; i++)
		{
			queue.add( (E)Operator.copy() );
		}
	    
		List<Pair<Integer,T>> indexedElements = new ArrayList<Pair<Integer,T>>(pElements.size());
		
		int index = 0;
		for(final T element : pElements)
		{
			indexedElements.add( new Pair<Integer,T>(index,element) );
			index++;
		}
		int size = index;
		final ArrayList<E> executors = new ArrayList<E>(size); 
		int numStarted = 0;
		
		for (final Pair<Integer,T> element : indexedElements) {
            synchronized(queue) {
                while (queue.isEmpty()) {
                    try{
                        queue.wait();						
                        numStarted++;
						if (numStarted % 5000 == 0)
							System.out.printf (".");
                    }
                    catch (InterruptedException e){
                    	System.err.println (e.getMessage());
                    }
                }
                executors.add(queue.removeFirst());
            }
            
            try{
				new Runnable() {
					@Override
					public void run() {
						pOperation.perform(executors.get(element.getFirst()), 
								element.getFirst(), element.getSecond());	
						synchronized(queue) {
							queue.add(executors.get(element.getFirst()));
							queue.notify();
						}
					}
				}.run();
			}
			catch(Exception e)
			{
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
		
		if (queue.size() != poolSize)
			System.err.println ("Some data processing was lost! " + "Only " + 
					queue.size() + " processors of " + poolSize + " exists now");
		
		Reducer<E> accumulator = new Reducer<E>();
		return accumulator.reduce(queue);
	}
	
	public static interface Operation<E,T> {
		public void perform(E operator, int index, T pParameter);
	}
	
}

