package math;

import java.util.LinkedList;

/**
 * @author Dan Klein
 */
public class LBFGSMinimizer implements GradientMinimizer {
	double EPS = 1e-10;
	int maxIterations = 20;
	int maxHistorySize = 5;
	LinkedList<double[]> inputDifferenceVectorList = new LinkedList<double[]>();
	LinkedList<double[]> derivativeDifferenceVectorList = new LinkedList<double[]>();

	public double[] minimize(DifferentiableFunction function, double[] initial,
			double tolerance) {
		
		maxHistorySize = Math.min( maxHistorySize, function.dimension() );
		
		BacktrackingLineSearcher lineSearcher = new BacktrackingLineSearcher();
		double[] guess = DoubleArrays.clone(initial);
		for (int iteration = 0; iteration < maxIterations; iteration++) {
			double value = function.valueAt(guess);
			double[] derivative = function.derivativeAt(guess);
			double[] initialInverseHessianDiagonal = getInitialInverseHessianDiagonal(function);
			double[] direction = implicitMultiply(
					initialInverseHessianDiagonal, derivative);
			// System.out.println(" Derivative is: "+DoubleArrays.toString(derivative,
			// 100));
			// DoubleArrays.assign(direction, derivative);
			DoubleArrays.scale(direction, -1.0);
			// System.out.println(" Looking in direction: "+DoubleArrays.toString(direction,
			// 100));
			if (iteration == 0)
				lineSearcher.stepSizeMultiplier = 0.01;
			else
				lineSearcher.stepSizeMultiplier = 0.5;
			double[] nextGuess = lineSearcher.minimize(function, guess,
					direction);
			double nextValue = function.valueAt(nextGuess);
			double[] nextDerivative = function.derivativeAt(nextGuess);
			 System.err.println("Iteration " + iteration + " ended with value "
				+ nextValue);
			if (converged(value, nextValue, tolerance))
				return nextGuess;
			updateHistories(guess, nextGuess, derivative, nextDerivative);
			guess = nextGuess;
			value = nextValue;
			derivative = nextDerivative;
		}
		// System.err.println("LBFGSMinimizer.minimize: Exceeded maxIterations without converging.");
		return guess;
	}

	private boolean converged(double value, double nextValue, double tolerance) {
		if (value == nextValue)
			return true;
		double valueChange = SloppyMath.abs(nextValue - value);
		double valueAverage = SloppyMath.abs(nextValue + value + EPS) / 2.0;
		if (valueChange / valueAverage < tolerance)
			return true;
		return false;
	}

	private void updateHistories(double[] guess, double[] nextGuess,
			double[] derivative, double[] nextDerivative) {
		double[] guessChange = DoubleArrays.addMultiples(nextGuess, 1.0, guess,
				-1.0);
		double[] derivativeChange = DoubleArrays.addMultiples(nextDerivative,
				1.0, derivative, -1.0);
		pushOntoList(guessChange, inputDifferenceVectorList);
		pushOntoList(derivativeChange, derivativeDifferenceVectorList);
	}

	private void pushOntoList(double[] vector, LinkedList<double[]> vectorList) {
		vectorList.addFirst(vector);
		if (vectorList.size() > maxHistorySize)
			vectorList.removeLast();
	}

	private int historySize() {
		return inputDifferenceVectorList.size();
	}

	private double[] getInputDifference(int num) {
		// 0 is previous, 1 is the one before that
		return inputDifferenceVectorList.get(num);
	}

	private double[] getDerivativeDifference(int num) {
		return derivativeDifferenceVectorList.get(num);
	}

	private double[] getLastDerivativeDifference() {
		return derivativeDifferenceVectorList.getFirst();
	}

	private double[] getLastInputDifference() {
		return inputDifferenceVectorList.getFirst();
	}

	private double[] implicitMultiply(double[] initialInverseHessianDiagonal,
			double[] derivative) {
		double[] rho = new double[initialInverseHessianDiagonal.length];
		double[] alpha = new double[initialInverseHessianDiagonal.length];
		double[] right = DoubleArrays.clone(derivative);
		// loop last backward
		for (int i = historySize() - 1; i >= 0; i--) {
			double[] inputDifference = getInputDifference(i);
			double[] derivativeDifference = getDerivativeDifference(i);
			rho[i] = DoubleArrays.innerProduct(inputDifference,
					derivativeDifference);
			if (rho[i] == 0.0)
				throw new RuntimeException(
						"LBFGSMinimizer.implicitMultiply: Curvature problem.");
			alpha[i] = DoubleArrays.innerProduct(inputDifference, right)
					/ rho[i];
			right = DoubleArrays.addMultiples(right, 1.0, derivativeDifference,
					-1.0 * alpha[i]);
		}
		double[] left = DoubleArrays.pointwiseMultiply(
				initialInverseHessianDiagonal, right);
		for (int i = 0; i < historySize(); i++) {
			double[] inputDifference = getInputDifference(i);
			double[] derivativeDifference = getDerivativeDifference(i);
			double beta = DoubleArrays.innerProduct(derivativeDifference, left)
					/ rho[i];
			left = DoubleArrays.addMultiples(left, 1.0, inputDifference,
					alpha[i] - beta);
		}
		return left;
	}

	private double[] getInitialInverseHessianDiagonal(
			DifferentiableFunction function) {
		double scale = 1.0;
		if (derivativeDifferenceVectorList.size() >= 1) {
			double[] lastDerivativeDifference = getLastDerivativeDifference();
			double[] lastInputDifference = getLastInputDifference();
			double num = DoubleArrays.innerProduct(lastDerivativeDifference,
					lastInputDifference);
			double den = DoubleArrays.innerProduct(lastDerivativeDifference,
					lastDerivativeDifference);
			scale = num / den;
		}
		return DoubleArrays.constantArray(scale, function.dimension());
	}

	public LBFGSMinimizer() {
	}

	public LBFGSMinimizer(int maxIterations) {
		this.maxIterations = maxIterations;
	}

}
