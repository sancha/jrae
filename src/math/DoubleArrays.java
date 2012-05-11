package math;

import java.util.Arrays;

/**
 */
public class DoubleArrays {
  public static double[] clone(double[] x) {
    double[] y = new double[x.length];
    assign(y,x);
    return y;
  }
  public static void assign(double[] y, double[] x) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    System.arraycopy(x, 0, y, 0, x.length);
  }
  public static double innerProduct(double[] x, double[] y) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    double result = 0.0;
    for (int i = 0; i < x.length; i++) {
      result += x[i]*y[i];
    }
    return result;
  }
  public static double[] addMultiples(double[] x, double xMultiplier, double[] y, double yMuliplier) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    double[] z = new double[x.length];
    for (int i = 0; i < z.length; i++) {
      z[i] = x[i]*xMultiplier + y[i]*yMuliplier;
    }
    return z;
  }
  public static double[] constantArray(double c, int length) {
    double[] x = new double[length];
    Arrays.fill(x, c);
    return x;
  }
  public static double[] pointwiseMultiply(double[] x, double[] y) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    double[] z = new double[x.length];
    for (int i = 0; i < z.length; i++) {
      z[i] = x[i]*y[i];
    }
    return z;
  }
  public static String toString(double[] x) {
    return toString(x, x.length);
  }
  public static String toString(double[] x, int length) {
    StringBuffer sb = new StringBuffer();
    sb.append("[");
    for (int i = 0; i < SloppyMath.min(x.length, length); i++) {
      sb.append(x[i]);
      if (i + 1 < SloppyMath.min(x.length, length))
        sb.append(", ");
    }
    sb.append("]");
    return sb.toString();
  }
  public static void scale(double[] x, double s) {
    if (s== 1.0) return;
    for (int i = 0; i < x.length; i++) {
      x[i] *= s;
    }
  }

  public static double[] multiply(double[] x, double s) {
    double[] result = new double[x.length];
    if (s== 1.0) {
      System.arraycopy(x, 0, result, 0, x.length);
      return result;
    }
    for (int i = 0; i < x.length; i++) {
      result[i] = x[i] * s;
    }
    return result;
  }

  public static int argMax(double[] v) {
    int maxI = -1;
    double maxV = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < v.length; i++) {
      if (v[i] > maxV) {
        maxV = v[i];
        maxI = i;
      }
    }
    return maxI;
  }

  public static double max(double[] v) {
    double maxV = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < v.length; i++) {
      if (v[i] > maxV) {
        maxV = v[i];
      }
    }
    return maxV;
  }

  public static double[] add(double[] a, double b) {
    double[] result = new double[a.length];
    for (int i = 0; i < a.length; i++) {
      double v = a[i];
      result[i] = v+b;
    }
    return result;
  }

  public static double vectorLength(double[] x) {
    return Math.sqrt(innerProduct(x, x));
  }
  
  public static double[] addi(double[] x, double[] y) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    for (int i = 0; i < x.length; i++) {
      x[i] = x[i] + y[i];
    }
    return x;
  }

  public static double[] add(double[] x, double[] y) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    double[] result = new double[x.length];
    for (int i = 0; i < x.length; i++) {
      result[i] = x[i] + y[i];
    }
    return result;
  }

  public static double[] subtract(double[] x, double[] y) {
    if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    double[] result = new double[x.length];
    for (int i = 0; i < x.length; i++) {
      result[i] = x[i] - y[i];
    }
    return result;
  }

  public static void initialize(double[] x, double d) {
    Arrays.fill(x, d);
  }
  
  public static void initialize(Object[] x, double d) {
    for (int i = 0; i < x.length; i++) {
      Object o = x[i];
      if (o instanceof double[])
        initialize((double[]) o, d);
      else
        initialize((Object[]) o, d);
    }
  }
  
  public static boolean equals(double[] x, double[] y){
	if (x.length != y.length) throw new RuntimeException("diff lengths: " + x.length + " " + y.length);
    boolean equal = true;
	for (int i = 0; i < x.length && equal; i++) {
    	equal = (x[i] == y[i]);
    }
    return equal;
  }
  
  public static double total(double[] x)
  {
	double sum = 0;
	for(int i=0; i<x.length; i++) sum+=x[i];
	return sum;
  }  

  public static void prettyPrint(double[] x)
  {
	  double sum = 0, abs_sum = 0;
	  for(int i=0; i<x.length; i++)
	  {
		  sum += x[i];
		  abs_sum += Math.abs(x[i]);
		  System.out.printf("%.3f ",x[i]);
	  }
	  System.out.println("(" + sum + "," + abs_sum +")");	  
  }
  
}
