package parallel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import util.Reducible;
import util.Reducer;
import util.Pair;

public class ThreadPool {
	private static int poolSize = Runtime.getRuntime().availableProcessors(); 
	
	public static void setPoolSize(int poolSize)
	{
		ThreadPool.poolSize = poolSize;
	}

	public static <T,E extends Reducible<E>,F> Collection<E> map
	(final Collection<T> pElements, final E Operator, final Operation<E,T> pOperation) {
		
		ExecutorService execSvc = Executors.newFixedThreadPool( ThreadPool.poolSize );
		final LinkedList<E> queue = new LinkedList<E>();
	    for (int i=0; i<ThreadPool.poolSize; i++)
		{
			queue.add( (E)Operator.copy() );
		}
	    
	    System.err.printf("Performing map-reduce on %d cores\n", queue.size());
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
						if (numStarted % 500 == 0)
							System.out.printf (".",queue.size());
                    }
                    catch (InterruptedException e){
                    	System.err.println (e.getMessage());
                    }
                }
                executors.add(queue.removeFirst());
            }
            
            try{
            	execSvc.execute( 
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
				});
			}
			catch(Exception e)
			{
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
		
		synchronized(queue) {
			while (queue.size() != poolSize)
			{
//				System.err.printf ("Waiting for the last threads to finish " +
//						"%d of %d\n", queue.size(), ThreadPool.poolSize );
				try {
						queue.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		
		
		return queue;	
	}

	public static <T,E extends Reducible<E>,F> E mapReduce
		(final Collection<T> pElements, final E Operator, final Operation<E,T> pOperation) {
		Collection<E> queue = map(pElements, Operator, pOperation);
		Reducer<E> accumulator = new Reducer<E>();
		return accumulator.reduce(queue);
	}
	
	public static interface Operation<E,T> {
		public void perform(E operator, int index, T pParameter);
	}
	
}

