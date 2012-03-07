package parallel;

import util.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Copyright 2011 Tantaman LLC
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 * 
 * @author tantaman from https://github.com/tantaman/commons
 */

public class Parallel {
	private static final int NUM_CORES = Runtime.getRuntime().availableProcessors();
	private static ExecutorService forPool;
	private static int poolSize = NUM_CORES; 
	
	public static void setPoolSize(int poolSize)
	{
		Parallel.poolSize = poolSize;
	}
	
	public static <T,F> void For(final Collection<T> pElements, final Operation<T> pOperation) {
		forPool = Executors.newFixedThreadPool(poolSize);
		
		List<Future<?>> futures = new LinkedList<Future<?>>();
		List<Pair<Integer,T>> indexedElements = new ArrayList<Pair<Integer,T>>(pElements.size());
		
		int index = 0;
		for(final T element : pElements)
		{
			indexedElements.add( new Pair<Integer,T>(index,element) );
			index++;
		}
		int size = index;
		
		for (final Pair<Integer,T> element : indexedElements) {
			try{
				Future<?> future = forPool.submit(new Runnable() {
					@Override
					public void run() {
						pOperation.perform(element.getFirst(), element.getSecond());					
					}
				});
				futures.add(future);
			}
			catch(Exception e)
			{
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
			
		}

		int numComplete = 0;
		for (Future<?> f : futures) {
			try {
				f.get();
				numComplete++;
				if (numComplete % 5000 == 0)
					System.out.printf (".");
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			} catch (ExecutionException e) {
				System.err.println(e.getMessage());
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		if (size > 5000)
			System.out.printf ("\n");
		
		forPool.shutdown();
	}
	
	public void shutdown()
	{
		forPool.shutdown();
	}

	public static interface Operation<T> {
		public void perform(int index, T pParameter);
	}
}
