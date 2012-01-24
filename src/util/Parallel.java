package util;

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

	// TODO: replace with custom cached thread pool.
	private static final ExecutorService forPool = Executors.newFixedThreadPool(NUM_CORES * 1);

	public static <T> void For(final Iterable<T> pElements, final Operation<T> pOperation) {
		ExecutorService executor = forPool;
		List<Future<?>> futures = new LinkedList<Future<?>>();
		for (final T element : pElements) {
			Future<?> future = executor.submit(new Runnable() {
				@Override
				public void run() {
					pOperation.perform(element);
				}
			});
			futures.add(future);
		}

		for (Future<?> f : futures) {
			try {
				f.get();
			} catch (InterruptedException e) {
			} catch (ExecutionException e) {
			}
		}
		executor.shutdown();
	}

	public static interface Operation<T> {
		public void perform(T pParameter);
	}

	public static void main(String[] args) {
		List<Integer> elems = new LinkedList<Integer>();
		for (int i = 0; i < 20; ++i) {
			elems.add(i);
		}

		Parallel.For(elems, new Parallel.Operation<Integer>() {
			public void perform(Integer pParameter) {
				System.out.println(pParameter);
			};
		});
	}
}