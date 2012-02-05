package math;

/**
 */
public class BacktrackingLineSearcher implements GradientLineSearcher {
  private double EPS = 1e-10;
  double stepSizeMultiplier = 0.9;
  private double sufficientDecreaseConstant = 1e-4;//0.9;

  public double[] minimize(DifferentiableFunction function, double[] initial, double[] direction) {
    double stepSize = 1.0;
    double initialFunctionValue = function.valueAt(initial);
    double initialDirectionalDerivative = DoubleArrays.innerProduct(function.derivativeAt(initial), direction);
    double[] guess = null;
    double guessValue = 0.0;
    boolean sufficientDecreaseObtained = false;
//    if (false) {
//      guess = DoubleArrays.addMultiples(initial, 1.0, direction, EPS);
//      double guessValue = function.valueAt(guess);
//      double sufficientDecreaseValue = initialFunctionValue + sufficientDecreaseConstant * initialDirectionalDerivative * EPS;
//      System.out.println("NUDGE TEST:");
//      System.out.println("  Trying step size:  "+EPS);
//      System.out.println("  Required value is: "+sufficientDecreaseValue);
//      System.out.println("  Value is:          "+guessValue);
//      System.out.println("  Initial was:       "+initialFunctionValue);
//      if (guessValue > initialFunctionValue) {
//        System.err.println("NUDGE TEST FAILED");
//        return initial;
//      }
//    }
    while (! sufficientDecreaseObtained) {
      guess = DoubleArrays.addMultiples(initial, 1.0, direction, stepSize);
      guessValue = function.valueAt(guess);
      double sufficientDecreaseValue = initialFunctionValue + sufficientDecreaseConstant * initialDirectionalDerivative * stepSize;
//      System.out.println("Trying step size:  "+stepSize);
//      System.out.println("Required value is: "+sufficientDecreaseValue);
//      System.out.println("Value is:          "+guessValue);
//      System.out.println("Initial was:       "+initialFunctionValue);
      sufficientDecreaseObtained = (guessValue <= sufficientDecreaseValue);
      if (! sufficientDecreaseObtained) {
        stepSize *= stepSizeMultiplier;
        if (stepSize < EPS) {
          //throw new RuntimeException("BacktrackingSearcher.minimize: stepSize underflow.");
          System.err.println("BacktrackingSearcher.minimize: stepSize underflow.");
          return initial;
        }
      }
    }
//    double lastGuessValue = guessValue;
//    double[] lastGuess = guess;
//    while (lastGuessValue >= guessValue) {
//      lastGuessValue = guessValue;
//      lastGuess = guess;
//      stepSize *= stepSizeMultiplier;
//      guess = DoubleArrays.addMultiples(initial, 1.0, direction, stepSize);
//      guessValue = function.valueAt(guess);
//    }
//    return lastGuess;
    return guess;
  }
}
