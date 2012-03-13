package rae;

import math.*;

import org.jblas.*;

import classify.ClassifierTheta;
import classify.SoftmaxCost;
import util.*;

import java.util.*;

public class RAEPropagation 
	implements Reducible<RAEPropagation>{
	DifferentiableMatrixFunction f;
	DoubleMatrix GW1, GW2, GW3, GW4, GWCat, GWe_total;
	DoubleMatrix Gb1, Gb2, Gb3, Gbcat;
	int HiddenSize, DictionaryLength, CatSize;
	double AlphaCat, Beta;
	
	private RAEPropagation(double AlphaCat, double Beta, int HiddenSize,
			int DictionaryLength, DifferentiableMatrixFunction f) {
		this.AlphaCat = AlphaCat;
		this.Beta = Beta;
		this.HiddenSize = HiddenSize;
		this.DictionaryLength = DictionaryLength;
		this.f = f;
		CatSize = -1;
		initializeGradients();
	}

	public RAEPropagation(double AlphaCat, double Beta, int HiddenSize,
			int CatSize, int DictionaryLength, DifferentiableMatrixFunction f) {
		this.AlphaCat = AlphaCat;
		this.Beta = Beta;
		this.HiddenSize = HiddenSize;
		this.DictionaryLength = DictionaryLength;
		this.f = f;
		this.CatSize = CatSize;
		initializeFineGradients();
	}
	
	/*
	 * (non-Javadoc) RAEPropagation's clone does not copy all the 
	 * intermediate gradient information. It copies only the paramters
	 * and then re-initializes them.
	 * @see java.lang.Object#clone()
	 */
	public Object copy()
	{
		return (CatSize != -1) ?
			new RAEPropagation(AlphaCat, Beta, HiddenSize, CatSize, DictionaryLength, f) :
			new RAEPropagation(AlphaCat, Beta, HiddenSize, DictionaryLength, f);
	}

	public RAEPropagation(Theta theta, double AlphaCat, double Beta,
			int HiddenSize, int DictionaryLength, DifferentiableMatrixFunction f) {
		this(AlphaCat, Beta, HiddenSize, DictionaryLength, f);
	}

	public RAEPropagation(FineTunableTheta theta, double AlphaCat, double Beta,
			int HiddenSize, int DictionaryLength, DifferentiableMatrixFunction f) {
		this(AlphaCat, Beta, HiddenSize, theta.Wcat.rows, DictionaryLength, f);
	}

	/**
	 * Building the tree / kids and forward propagation
	 */
	public LabeledRAETree ForwardPropagate(Theta theta, DoubleMatrix WordsEmbedded,
			FloatMatrix Freq, int CurrentLabel, int SentenceLength) {
		LabeledRAETree tree = new LabeledRAETree(SentenceLength, CurrentLabel, HiddenSize, WordsEmbedded);
		ArrayList<Integer> CollapsedSentence = new ArrayList<Integer>(
				SentenceLength);
		for (int i = 0; i < SentenceLength; i++)
			CollapsedSentence.add(i);

		for (int j = 0; j < SentenceLength - 1; j++) {
			int NumComponents = WordsEmbedded.columns;

			/**
			 * C1 and C2 are matrices, not vectors. Therefore, we use C1 instead
			 * of c1 Since this is the training phase, we compute the
			 * reconstruction error for each pair of words, i.e., C1(:,i) and
			 * C2(:,i) for each i indicates one pair. However note that in the
			 * last iteration, C1 and C2 are column vectors
			 **/
			DoubleMatrix C1 = WordsEmbedded.getColumns(ArraysHelper.makeArray(0, NumComponents - 2));
			DoubleMatrix C2 = WordsEmbedded.getColumns(ArraysHelper.makeArray(1, NumComponents - 1));

			// FloatMatrix Freq1 = Freq.getColumns( makeArray(0,NumComponents-2));
			// FloatMatrix Freq2 = Freq.getColumns( makeArray(1,NumComponents-1));

			// W1*c1 + W2*c2
			DoubleMatrix ActivationInp = (theta.W1.mmul(C1)).add(theta.W2.mmul(C2)); 
			DoubleMatrix P = f.valueAt(ActivationInp.addiColumnVector(theta.b1));

			/** Internal representation **/
			DoubleMatrix PNorm = DoubleMatrixFunctions.ColumnWiseNormalize(P);

			/**
			 * Deviation in naming convention in the original source. It is now
			 * uniform and can be understood that the variable without "norm"
			 * suffixed to their names may be unnormalized
			 **/
			DoubleMatrix Y1 = f.valueAt((theta.W3.mmul(PNorm)).addColumnVector(theta.b2));
			DoubleMatrix Y2 = f.valueAt((theta.W4.mmul(PNorm)).addColumnVector(theta.b3));

			/** Reconstruction of C1 and C2 **/
			DoubleMatrix Y1Norm = DoubleMatrixFunctions.ColumnWiseNormalize(Y1);
			DoubleMatrix Y2Norm = DoubleMatrixFunctions.ColumnWiseNormalize(Y2);

			/** Reconstruction error **/
			DoubleMatrix Y1MinusC1 = Y1Norm.sub(C1);
			DoubleMatrix Y2MinusC2 = Y2Norm.sub(C2);

			DoubleMatrix Y1C1 = Y1MinusC1.mul(AlphaCat);
			DoubleMatrix Y2C2 = Y2MinusC2.mul(AlphaCat);

			/** Equation (4) in the paper: reconstruction error (row vector) **/
			DoubleMatrix J = (((Y1C1.mul(Y1MinusC1)).add(Y2C2.mul(Y2MinusC2)))
					.columnSums()).muli(0.5);
			// System.out.println("J : " + J.rows + " " + J.columns + " [ " +
			// NumComponents);
			/**
			 * finding the pair with smallest reconstruction error for
			 * constructing tree
			 **/
			double J_min = J.min();
			int J_minpos = J.argmin();

			// System.out.println(J_min + " " + J_minpos);

			RAENode NewParent = tree.T[SentenceLength + j];
			NewParent.Y1C1 = Y1C1.getColumn(J_minpos);
			NewParent.Y2C2 = Y2C2.getColumn(J_minpos);
			NewParent.DeltaOut1 = f.derivativeAt(Y1.getColumn(J_minpos))
					.mmul(Y1C1.getColumn(J_minpos));
			NewParent.DeltaOut2 = f.derivativeAt(Y2.getColumn(J_minpos))
					.mmul(Y2C2.getColumn(J_minpos));
			NewParent.Features = PNorm.getColumn(J_minpos);
			NewParent.UnnormalizedFeatures = P.getColumn(J_minpos);
			NewParent.scores = new double[]{ J_min };
			tree.TotalScore += J_min;
			
			int LeftChildIndex = CollapsedSentence.get(J_minpos), 
				RightChildIndex = CollapsedSentence.get(J_minpos + 1);

			RAENode LeftChild = tree.T[LeftChildIndex], RightChild = tree.T[RightChildIndex];

			NewParent.LeftChild = LeftChild;
			NewParent.RightChild = RightChild;
			NewParent.SubtreeSize = LeftChild.SubtreeSize
					+ RightChild.SubtreeSize;

			LeftChild.parent = NewParent;
			RightChild.parent = NewParent;
			tree.structure.set(SentenceLength + j, 
					new Pair<Integer, Integer>(LeftChildIndex, RightChildIndex));

			// freq(J_minpos+1) = [];
			// freq(J_minpos) = (Tree.numkids(Tree.kids(sl+j,1))*freq1(J_minpos)
			// +
			// Tree.numkids(Tree.kids(sl+j,2))*freq2(J_minpos))/(Tree.numkids(Tree.kids(sl+j,1))+Tree.numkids(Tree.kids(sl+j,2)));
			
			WordsEmbedded = UpdateEmbedding(WordsEmbedded, J_minpos,
					PNorm.getColumn(J_minpos));
			CollapsedSentence.remove(J_minpos + 1);
			CollapsedSentence.set(J_minpos, SentenceLength + j);
		}
		return tree;
	}

	/**
	 * Returning the classification error for the given tree.
	 */
	public LabeledRAETree ForwardPropagate(FineTunableTheta theta,
			DoubleMatrix WordsEmbedded, FloatMatrix Freq, int CurrentLabel,
			int SentenceLength, LabeledRAETree tree) {
		CatSize = theta.Wcat.rows;
		int TreeSize = 2 * SentenceLength - 1;
//		LabeledRAETree tree = new LabeledRAETree(SentenceLength, CurrentLabel, HiddenSize, CatSize, WordsEmbedded);
		int[] SubtreeSize = new int[TreeSize];
		int[] requiredEntries = ArraysHelper.makeArray(0, CatSize-1);
		
		SoftmaxCost softmaxCalc = new SoftmaxCost (1+CatSize, HiddenSize, 0);
		ClassifierTheta ClassifierTheta = theta.getClassifierParameters();
		int[] labelVector = new int[SentenceLength];
		Arrays.fill(labelVector, CurrentLabel);
		DoubleMatrix Labels = softmaxCalc.getLabelRepresentation(labelVector);
		DoubleMatrix LabelVector = Labels.getColumn(0); 
		
		for (int i = SentenceLength; i < TreeSize; i++) {
			int LeftChild = tree.structure.get(i).getFirst(), 
				RightChild = tree.structure.get(i).getSecond();

			SubtreeSize[i] = SubtreeSize[LeftChild] + SubtreeSize[RightChild];
		}

		DoubleMatrix Predictions = softmaxCalc.getPredictions(ClassifierTheta, WordsEmbedded);		
		
		DoubleMatrix Error = softmaxCalc.getLoss(Predictions, Labels).mul(1-AlphaCat);
		DoubleMatrix ErrorGradient = softmaxCalc
			.getGradient(ClassifierTheta, WordsEmbedded, Labels)
			.muli(1-AlphaCat); 
		
		for (int i = 0; i < TreeSize; i++) {
			RAENode CurrentNode = tree.T[i];
			if (i < SentenceLength) {
				CurrentNode.scores = Predictions.getColumn(i).data;
				CurrentNode.catDelta = ErrorGradient.getColumn(i).getRows(requiredEntries);
				tree.TotalScore += Error.getColumn(i).sum ();
			} else {
				int LeftChild = tree.structure.get(i).getFirst(), 
					RightChild = tree.structure.get(i).getSecond();
				
				DoubleMatrix C1 = tree.T[LeftChild].Features, C2 = tree.T[RightChild].Features;

				CurrentNode.LeftChild = tree.T[LeftChild];
				CurrentNode.RightChild = tree.T[RightChild];
				CurrentNode.LeftChild.parent = CurrentNode;
				CurrentNode.RightChild.parent = CurrentNode;

				// Eq. (2) in the paper: p = f(W(1)[c1; c2] + b(1))
				DoubleMatrix p = f.valueAt(((theta.W1.mmul(C1)).addi(theta.W2
						.mmul(C2))).addColumnVector(theta.b1));

				// See last paragraph in Section 2.3
				DoubleMatrix pNorm1 = p.div(p.norm2());

				CurrentNode.UnnormalizedFeatures = p;
				CurrentNode.Features = pNorm1;		
				
				Predictions = softmaxCalc.getPredictions(ClassifierTheta, pNorm1);
				
				CurrentNode.scores = Predictions.data;
				
				CurrentNode.catDelta = softmaxCalc
					.getGradient(ClassifierTheta, pNorm1, LabelVector)
					.mul((1-AlphaCat)* Beta)
					.getRows(requiredEntries);
			
				tree.TotalScore += ((1-AlphaCat)* Beta) *
					softmaxCalc.getLoss(Predictions, LabelVector).sum();
				
				CurrentNode.SubtreeSize = SubtreeSize[i];
			}
		}
		return tree;
	}

	public void BackPropagate(final LabeledRAETree tree, Theta theta, int[] WordsIndexed) {
		int SentenceLength = WordsIndexed.length;
		if (tree.T.length != 2 * SentenceLength - 1)
			System.err.println("Bad Tree for backpropagation!");

		DoubleMatrix GL = DoubleMatrix.zeros(HiddenSize, SentenceLength);

		// Stack of currentNode, Left(1) or Right(2), Parent Node pointer
		Stack<Triplet<RAENode, Integer, RAENode>> ToPopulate = new Stack<Triplet<RAENode, Integer, RAENode>>();

		RAENode Root = tree.T[tree.T.length - 1];
		Root.ParentDelta = DoubleMatrix.zeros(HiddenSize, 1);
		ToPopulate.push(new Triplet<RAENode, Integer, RAENode>(Root, 0, null));

		DoubleMatrix[] W = new DoubleMatrix[3];
		W[0] = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		W[1] = theta.W1;
		W[2] = theta.W2;
		DoubleMatrix Y0C0 = DoubleMatrix.zeros(HiddenSize, 1);

		while (!ToPopulate.empty()) {
			Triplet<RAENode, Integer, RAENode> top = ToPopulate.pop();
			int LeftOrRight = top.getSecond();
			RAENode CurrentNode = top.getFirst(), ParentNode = top.getThird();
			DoubleMatrix[] YCSelector = null;
			if (ParentNode == null)
				YCSelector = new DoubleMatrix[] { Y0C0, null, null };
			else
				YCSelector = new DoubleMatrix[] { Y0C0, ParentNode.Y1C1,
						ParentNode.Y2C2 };

			DoubleMatrix NodeW = W[LeftOrRight];
			DoubleMatrix delta = YCSelector[LeftOrRight];

			if (!CurrentNode.isLeaf()) {
				ToPopulate.push(new Triplet<RAENode, Integer, RAENode>(
						CurrentNode.LeftChild, 1, CurrentNode));
				ToPopulate.push(new Triplet<RAENode, Integer, RAENode>(
						CurrentNode.RightChild, 2, CurrentNode));

				DoubleMatrix A1 = CurrentNode.UnnormalizedFeatures, A1Norm = CurrentNode.Features;
				DoubleMatrix ND1 = CurrentNode.DeltaOut1, ND2 = CurrentNode.DeltaOut2;
				DoubleMatrix PD = CurrentNode.ParentDelta;

				DoubleMatrix Activation = ((theta.W3.transpose()).mmul(ND1))
						.addi((theta.W4.transpose()).mmul(ND2));
				Activation = Activation.addi(((NodeW.transpose()).mmul(PD))
						.subi(delta));
				DoubleMatrix CurrentDelta = f.derivativeAt(A1).mmul(Activation);

				CurrentNode.LeftChild.ParentDelta = CurrentDelta;
				CurrentNode.RightChild.ParentDelta = CurrentDelta;

				DoubleMatrix 
					GW1_upd = CurrentDelta.mmul(CurrentNode.LeftChild.Features.transpose()),
					GW2_upd = CurrentDelta.mmul(CurrentNode.RightChild.Features.transpose()),
					GW3_upd = ND1.mmul(A1Norm.transpose()),
					GW4_upd = ND2.mmul(A1Norm.transpose());
					
				
				accumulate(GW1_upd, GW2_upd, GW3_upd, GW4_upd, CurrentDelta, ND1, ND2);
				
			} else {
				DoubleMatrixFunctions.IncrementColumn(GL, CurrentNode.NodeName,
						((NodeW.transpose()).mmul(CurrentNode.ParentDelta))
								.subi(delta));
			}
		}
		
		incrementWordEmbedding(GL,WordsIndexed);
	}

	public void BackPropagate(LabeledRAETree tree, FineTunableTheta theta,
			int[] WordsIndexed) {
		int SentenceLength = WordsIndexed.length;
		if (tree.T.length != 2 * SentenceLength - 1)
			System.err.println("Bad Tree for backpropagation!");

		DoubleMatrix GL = DoubleMatrix.zeros(HiddenSize, SentenceLength);
		
		// Stack of currentNode, Left(1) or Right(2), Parent Node pointer
		Stack<Triplet<RAENode, Integer, RAENode>> ToPopulate = new Stack<Triplet<RAENode, Integer, RAENode>>();

		RAENode Root = tree.T[tree.T.length - 1];
		Root.ParentDelta = DoubleMatrix.zeros(HiddenSize, 1);
		ToPopulate.push(new Triplet<RAENode, Integer, RAENode>(Root, 0, null));

		DoubleMatrix[] W = new DoubleMatrix[3];
		W[0] = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		W[1] = theta.W1;
		W[2] = theta.W2;
		DoubleMatrix Y0C0 = DoubleMatrix.zeros(HiddenSize, 1);

		while (!ToPopulate.empty()) {
			Triplet<RAENode, Integer, RAENode> top = ToPopulate.pop();
			int LeftOrRight = top.getSecond();
			RAENode CurrentNode = top.getFirst(), ParentNode = top.getThird();
			DoubleMatrix[] YCSelector = null;
			if (ParentNode == null)
				YCSelector = new DoubleMatrix[] { Y0C0, null, null };
			else
				YCSelector = new DoubleMatrix[] { Y0C0, ParentNode.Y1C1, ParentNode.Y2C2 };

			DoubleMatrix NodeW = W[LeftOrRight];
			DoubleMatrix delta = YCSelector[LeftOrRight];

			if (!CurrentNode.isLeaf()) 
			{
				ToPopulate.push(new Triplet<RAENode, Integer, RAENode>(CurrentNode.LeftChild, 1, CurrentNode));
				ToPopulate.push(new Triplet<RAENode, Integer, RAENode>(CurrentNode.RightChild, 2, CurrentNode));

				DoubleMatrix A1 = CurrentNode.UnnormalizedFeatures, A1Norm = CurrentNode.Features;
				DoubleMatrix ND1 = CurrentNode.DeltaOut1, ND2 = CurrentNode.DeltaOut2;
				DoubleMatrix PD = CurrentNode.ParentDelta;

				DoubleMatrix Activation = ((theta.W3.transpose()).mmul(ND1))
						.addi(theta.W4.transpose().mmul(ND2))
						.addi(NodeW.transpose().mmul(PD))
						.addi(theta.Wcat.transpose().mmul(CurrentNode.catDelta));
				Activation.subi(delta);
				DoubleMatrix CurrentDelta = f.derivativeAt(A1).mmul(Activation);

				CurrentNode.LeftChild.ParentDelta = CurrentDelta;
				CurrentNode.RightChild.ParentDelta = CurrentDelta;
				
				DoubleMatrix 
					GW1_upd = CurrentDelta.mmul(CurrentNode.LeftChild.Features.transpose()),
					GW2_upd = CurrentDelta.mmul(CurrentNode.RightChild.Features.transpose()),
					GW3_upd = ND1.mmul(A1Norm.transpose()),
					GW4_upd = ND2.mmul(A1Norm.transpose()),
					GWCat_upd = CurrentNode.catDelta.mmul(A1Norm.transpose());
				
				accumulate(GW1_upd, GW2_upd, GW3_upd, GW4_upd, CurrentDelta, ND1, ND2, 
												GWCat_upd, CurrentNode.catDelta.rowSums());				
				
			} else {
				accumulate(CurrentNode.catDelta.mmul(CurrentNode.Features.transpose()),
						CurrentNode.catDelta);
				
				DoubleMatrixFunctions.IncrementColumn(GL, CurrentNode.NodeName,
						(((NodeW.transpose()).mmul(CurrentNode.ParentDelta))
							.addi(theta.Wcat.transpose().mmul(CurrentNode.catDelta)))
						.subi(delta));
			}
		}

		incrementWordEmbedding(GL,WordsIndexed);
	}
	
	private synchronized void incrementWordEmbedding(DoubleMatrix GL, int[] WordsIndexed)
	{
		for (int l = 0; l < GL.columns; l++)
			DoubleMatrixFunctions.IncrementColumn(GWe_total, WordsIndexed[l], GL.getColumn(l));		
	}
	
	private synchronized void accumulate(
				DoubleMatrix GW1_upd, DoubleMatrix GW2_upd, DoubleMatrix GW3_upd,
				DoubleMatrix GW4_upd, DoubleMatrix Gb1_upd, DoubleMatrix Gb2_upd,
				DoubleMatrix Gb3_upd)
	{
		GW1.addi(GW1_upd);
		GW2.addi(GW2_upd);
		GW3.addi(GW3_upd);
		GW4.addi(GW4_upd);

		Gb1.addi(Gb1_upd);
		Gb2.addi(Gb2_upd);
		Gb3.addi(Gb3_upd);		
	}
	
	private synchronized void accumulate(DoubleMatrix GWcat_upd, DoubleMatrix Gbcat_upd)
	{
		Gbcat.addi(Gbcat_upd);
		GWCat.addi(GWcat_upd);
	}
	
	private synchronized void accumulate(
			DoubleMatrix GW1_upd, DoubleMatrix GW2_upd, DoubleMatrix GW3_upd,
			DoubleMatrix GW4_upd, DoubleMatrix Gb1_upd, DoubleMatrix Gb2_upd,
			DoubleMatrix Gb3_upd, DoubleMatrix GWcat_upd, DoubleMatrix Gbcat_upd)
	{
		accumulate(GWcat_upd, Gbcat_upd);
		accumulate(GW1_upd,GW2_upd,GW3_upd,GW4_upd,Gb1_upd,Gb2_upd,Gb3_upd);
	}

	private void initializeGradients() {
		GW1 = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		GW2 = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		GW3 = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		GW4 = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		GWe_total = DoubleMatrix.zeros(HiddenSize, DictionaryLength);

		Gb1 = DoubleMatrix.zeros(HiddenSize, 1);
		Gb2 = DoubleMatrix.zeros(HiddenSize, 1);
		Gb3 = DoubleMatrix.zeros(HiddenSize, 1);
	}

	private void initializeFineGradients() {
		initializeGradients();
		GWCat = DoubleMatrix.zeros(CatSize, HiddenSize);
		Gbcat = DoubleMatrix.zeros(CatSize, 1);
	}

	private DoubleMatrix UpdateEmbedding(DoubleMatrix CurrentEmbedding,
			int Column, DoubleMatrix ColumnVector) {
		int[] leftColumns = ArraysHelper.makeArray(0, Column - 1);
		int[] rightColumns = ArraysHelper.makeArray(Column + 2, CurrentEmbedding.columns - 1);

		if (leftColumns == null && rightColumns == null)
			return ColumnVector;
		else if (leftColumns == null)
			return DoubleMatrix.concatHorizontally(ColumnVector, 
									CurrentEmbedding.getColumns(rightColumns));
		else if (rightColumns == null)
			return DoubleMatrix.concatHorizontally(CurrentEmbedding.getColumns(leftColumns), 
									ColumnVector);
		else
			return DoubleMatrix.concatHorizontally(CurrentEmbedding.getColumns(leftColumns), 
							DoubleMatrix.concatHorizontally(
									ColumnVector, CurrentEmbedding.getColumns(rightColumns)));
	}

	
	@Override
	public synchronized void reduce(RAEPropagation t) {
		accumulate(t.GW1, t.GW2, t.GW3, t.GW4, t.Gb1, t.Gb2, t.Gb3);
		if(t.GWCat != null)
			accumulate(t.GWCat, t.Gbcat);
		GWe_total.addi(t.GWe_total);
	}
}