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
}
