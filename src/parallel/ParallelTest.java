package parallel;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.Test;

import util.Reducible;

public class ParallelTest {
	
	class TotalMaintain implements Parallel.Operation<Integer> {
		long sum;
		private Lock lock;

		public TotalMaintain() {
			sum = 0;
			lock = new ReentrantLock();
		}

		@Override
		public void perform(int index, Integer pParameter) {
			lock.lock();
			sum += pParameter;
			lock.unlock();
		}

		public long getTotal() {
			return sum;
		}
	}
	
	class ReducibleTotaller implements Reducible<ReducibleTotaller>
	{
		long sum;
		
		@Override
		public void reduce(ReducibleTotaller instance) {
			sum += instance.sum;
		}

		@Override
		public Object copy() {
			return new ReducibleTotaller();
		}
		
		public void add(short val){
			sum += val;
		}
	}
	
	long sum;
	
	@Test
	public void test ()
	{
		Random rgen = new Random();
		int HK = 1000000; 
		List<Short> array = new ArrayList<Short>(HK);
		long expTotal = 0;
		for (int i=0; i<HK; i++)
		{
			array.add((short) (rgen.nextInt() % Short.MAX_VALUE));
			expTotal += array.get(i);
		}
		
		long startTime = System.currentTimeMillis();
		long endTime;
		try {
			sum = 0;
			final Lock lock = new ReentrantLock();
			Parallel.For(array, new Parallel.Operation<Short>() {
				@Override
				public void perform(int index, Short pParameter) {
					lock.lock();
					sum += pParameter;
					lock.unlock();
				}
			});
		} finally {
			endTime = System.currentTimeMillis();
		}
		long duration = endTime - startTime;
		System.out.println ("Parallel took " + duration+ " milliseconds");
		assertTrue(sum == expTotal);
	
		startTime = System.currentTimeMillis();
		ReducibleTotaller mrt = new ReducibleTotaller();
		try {
			mrt = ThreadPool.mapReduce (array, mrt, 
					new ThreadPool.Operation<ReducibleTotaller,Short>() {
						public void perform(ReducibleTotaller tot, int index, Short val)
						{
							tot.add(val);
						}
			});
			
		} finally {
			endTime = System.currentTimeMillis();
		}
		duration = endTime - startTime;
		System.out.println ("Mapreduce took " + duration+ " milliseconds");
		assertTrue(mrt.sum == expTotal);
	}
}
