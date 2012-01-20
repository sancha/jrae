package io;

import org.jblas.*;
import java.io.*;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.*;


/**
 * TODO This class needs to be more generic.
 */
public class MatFile {

	private String FileName;
	private MatFileReader mfr;

	public MatFile(String Path) throws IOException {
		FileName = Path;
		mfr = new MatFileReader(FileName);
	}

	public double[] readThetaVector(String VarName) {
		DoubleMatrix ret = null;
		try {
			MLArray mlArrayRetrived = mfr.getMLArray(VarName);
			ret = new DoubleMatrix(((MLDouble) mlArrayRetrived).getArray());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return ret.data;
	}

	public DoubleMatrix readOriginalWe(String VarName) {
		DoubleMatrix ret = null;
		try {
			MLArray mlArrayRetrived = mfr.getMLArray(VarName);
			ret = new DoubleMatrix(((MLDouble) mlArrayRetrived).getArray());
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return ret;
	}
	
	public void writeDoubleMatrix(DoubleMatrix inp, String VarName)
	{
		
	}

	public static void main(String[] args) throws Exception 
	{
		String dir = "data/parsed";
		MatFile DataLoader = new MatFile(dir + "/optParams.mat");

		final long startTime = System.nanoTime();
		final long endTime;
		double[] OptTheta = DataLoader.readThetaVector("opttheta");
		endTime = System.nanoTime();
		final long duration = endTime - startTime;
		System.out.println("OptTheta loaded of size : " + OptTheta.length + " in " + duration + " nanoseconds");
	

		 DataLoader = new MatFile(dir + "/We2.mat");
		 DoubleMatrix WeOrig = DataLoader.readOriginalWe("We2");
		 System.out.println("WeOrig loaded of size : " + WeOrig.rows + " " + WeOrig.columns);
	}
}
