package classify;

import static org.junit.Assert.*;
import io.DataSet;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.Test;

public class CrossValidationTest{
	
	@Test
	public void test1() {
		String dir = "data/parsed";
		DataSet<DummyDatum,Double> Data = new DataSet<DummyDatum,Double>(100);
		try 
		{
			FileInputStream fstream = new FileInputStream(dir + "/test.txt");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));

			for (int i = 0; i < 100; i++) {
				String[] parts = br.readLine().split(" ");
				double x = Double.parseDouble(parts[0]);
				double y = Double.parseDouble(parts[1]);
				int l = Integer.parseInt(parts[2]);
				
				Data.add( new DummyDatum(x,y,l,i) );
			}

			fstream.close();
			in.close();
			br.close();
		} 
		catch (Exception e) 
		{
			System.err.println(e.getMessage());
		}
	
		CrossValidation<DummyDatum,Double> cv = new CrossValidation<DummyDatum,Double>(10, Data);
		List<DummyDatum> td = cv.getTrainingData(0);
		List<DummyDatum> vd = cv.getValidationData(0);
		Set<Integer> s = new HashSet<Integer>();
		
		System.out.println( td.size() + " " + vd.size() );
		
		int total = 0, count = 0;
		
		for(DummyDatum d : td)
		{
			s.add(d.getIndex());
			System.out.println(d.getIndex());
			total += d.getIndex();
			count++;
		}
		
		for(DummyDatum d : vd)
		{
			System.out.println(d.getIndex());
			if( s.contains( d.getIndex() ))
			{
				assertTrue(false);
				break;
			}
			total += d.getIndex();
			count++;
		}
		
		assertTrue(total == (count * (count-1))/2);
		assertTrue(td.size() + vd.size() == Data.Data.size());
	}

	public void test2()
	{
		int[] permutation = new int[ 10662 ];
		
		try{
		BufferedReader inBr = new BufferedReader(new FileReader("data/parsed/permutation.txt"));
		String sLine; int i=0;
		while ((sLine = inBr.readLine())!=null) {
			permutation[i++] = Integer.parseInt(sLine.trim());
		}
		inBr.close();
		}catch(Exception e)
		{
			System.err.println(e.getLocalizedMessage());
		}
	}
}

class DummyDatum implements LabeledDatum<Double, Integer>
{
	double x,y;
	int l, i;
	
	DummyDatum(double x, double y, int l, int i)
	{
		this.x = x;
		this.y = y;
		this.l = l;
		this.i = i;
	}
	
	@Override
	public Collection<Double> getFeatures() {
		List<Double> f = new ArrayList<Double>(2);
		f.add(x);
		f.add(y);
		return f;
	}

	@Override
	public Integer getLabel() {
		return l;
	}
	
	public int getIndex()
	{
		return i;
	}
}
