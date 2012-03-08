package classify;

import org.jblas.DoubleMatrix;

public class Accuracy
{
	double Precision, Recall, Accuracy, F1;
	
	public Accuracy(int[] Predictions, int[] GoldLabels, int CatSize)
	{
		System.err.println(CatSize);
		DoubleMatrix ConfusionMatrix = DoubleMatrix.zeros(CatSize,CatSize);
		for(int i=0; i<GoldLabels.length; i++)
		{
			double Val = ConfusionMatrix.get(Predictions[i], GoldLabels[i]);
			ConfusionMatrix.put( Predictions[i], GoldLabels[i], Val + 1 );
		}
		System.out.println(ConfusionMatrix);
		
		DoubleMatrix Diag = ConfusionMatrix.diag();
		Precision = ((Diag.div( ConfusionMatrix.columnSums() )).sum()) / (CatSize);
		Recall = ((Diag.div( ConfusionMatrix.rowSums() )).sum()) / (CatSize);
		Accuracy = Diag.sum() / ConfusionMatrix.sum();
		F1 = ( 2 * Precision * Recall ) / (Precision + Recall);
	}
	
	@Override
	public String toString()
	{
		String S = "\n{" + 
				"\n\tPrecision : " + Precision +
				"\n\tRecall : " + Recall +
				"\n\tAccuracy : "+ Accuracy +
				"\n\tF1 Score : " + F1 +
				"\n}";
		return S;
	}
}

