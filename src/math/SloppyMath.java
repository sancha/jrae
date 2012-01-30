package math;

import java.lang.Math;

/**
 * Routines for some approximate math functions.
 * 
 * @author Dan Klein
 * @author Teg Grenager
 */
public class SloppyMath {

	/**
	 * If a difference is bigger than this in log terms, then the sum or
	 * difference of them will just be the larger (to 12 or so decimal places
	 * for double, and 7 or 8 for float).
	 */
	static final double LOGTOLERANCE = 30.0;
	static final float LOGTOLERANCE_F = 20.0f;
	
	public static double min(int x, int y) {
		if (x > y)
			return y;
		return x;
	}

	public static double max(int x, int y) {
		if (x > y)
			return x;
		return y;
	}

	public static double abs(double x) {
		if (x > 0)
			return x;
		return -1.0 * x;
	}

	/**
	 * Returns log(X+Y). Calculation is approximate. If logX or logY is much
	 * bigger than the other, then just return the much bigger value.
	 * 
	 * @param logX
	 *            natural log of X
	 * @param logY
	 *            natural log of Y
	 * @return log(X+Y) - natural log of (X+Y)
	 */
	public static double logAdd(double logX, double logY) {
		// make logX the max
		if (logY > logX) {
			double temp = logX;
			logX = logY;
			logY = temp;
		}
		// now logX is bigger
		if (logX == Double.NEGATIVE_INFINITY) {
			return logX;
		}
		double negDiff = logY - logX;
		if (negDiff < -20) {
			return logX;
		}
		return logX + java.lang.Math.log(1.0 + java.lang.Math.exp(negDiff));
	}

	/**
	 * Returns log( sum V[i] ).
	 * 
	 * @param logV
	 *            array of natural logs of V[i]
	 * @return log( sum V[i]) - natural log of the sum the the V[i]'s
	 */
	public static double logAdd(double[] logV) {
		double max = Double.NEGATIVE_INFINITY;
		double maxIndex = 0;
		for (int i = 0; i < logV.length; i++) {
			// if (logV[i] > Double.NEGATIVE_INFINITY) {
			if (logV[i] > max) {
				max = logV[i];
				maxIndex = i;
			}
		}
		if (max == Double.NEGATIVE_INFINITY)
			return Double.NEGATIVE_INFINITY;
		// compute the negative difference
		double threshold = max - 20;
		double sumNegativeDifferences = 0.0;
		for (int i = 0; i < logV.length; i++) {
			if (i != maxIndex && logV[i] > threshold) {
				sumNegativeDifferences += Math.exp(logV[i] - max);
			}
		}
		if (sumNegativeDifferences > 0.0) {
			return max + Math.log(1.0 + sumNegativeDifferences);
		} else {
			return max;
		}
	}

	public static double exp(double logX) {
		// if x is very near one, use the linear approximation
		if (abs(logX) < 0.001)
			return 1 + logX;
		return Math.exp(logX);
	}

}
