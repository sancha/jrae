package classify;

import static org.junit.Assert.*;

import io.*;

import java.io.*;
import java.util.*;
import org.junit.Test;

public class StratifiedCrossValidationTest {

	@Test
	/**
	 * This test case uses dummy data that has 50 bad cases and 50 good cases,
	 * so all folds should have equal number of good and bad examples.
	 */
	public void test1() {
		String dir = "data/parsed";
		LabeledDataSet<DummyDatum,Double,Integer> Data = new LabeledDataSet<DummyDatum,Double,Integer>(100);
		try 
		{
			FileInputStream fstream = new FileInputStream(dir + "/test.txt");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			int goodCount = 0, badCount = 0;
			
			for (int i = 0; i < 100; i++) {
				String[] parts = br.readLine().split(" ");
				double x = Double.parseDouble(parts[0]);
				double y = Double.parseDouble(parts[1]);
				int l = Integer.parseInt(parts[2]);

				goodCount += l == 1 ? 1 : 0;
				badCount += l == 0 ? 1 : 0;
				
				Data.add( new DummyDatum(x,y,l,i) );
			}
			
			System.out.println(goodCount+ " " + badCount);
			assertTrue(goodCount == badCount);

			fstream.close();
			in.close();
			br.close();
		} 
		catch (Exception e) 
		{
			System.err.println(e.getMessage());
		}
	
		StratifiedCrossValidation<DummyDatum,Double,Integer> cv = 
				new StratifiedCrossValidation<DummyDatum,Double,Integer>(10, Data);
		List<DummyDatum> td = cv.getTrainingData(0);
		List<DummyDatum> vd = cv.getValidationData(0);
		Set<Integer> s = new HashSet<Integer>();
		
		System.out.println( td.size() + " " + vd.size() );
		
		int total = 0, count = 0, goodCount = 0, badCount = 0;
		
		for(DummyDatum d : td)
		{
			s.add(d.getIndex());
			total += d.getIndex();
			goodCount += d.getLabel() == 1 ? 1 : 0;
			badCount += d.getLabel() == 0 ? 1 : 0;
			count++;
		}
		System.out.println(goodCount+ " " + badCount);
		assertTrue( goodCount == badCount );
		
		for(DummyDatum d : vd)
		{
			if( s.contains( d.getIndex() ))
			{
				assertTrue(false);
				break;
			}
			goodCount += d.getLabel() == 0 ? 0 : 1;
			badCount += d.getLabel() == 1 ? 0 : 1;			
			total += d.getIndex();
			count++;
		}
		
		assertTrue( goodCount == badCount );
		assertTrue(total == (count * (count-1))/2);
		assertTrue(td.size() + vd.size() == Data.Data.size());
	}
}