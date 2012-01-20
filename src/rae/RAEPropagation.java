package rae;

import math.*;

import org.jblas.*;

import util.*;

import java.io.*;
import java.util.*;

public class RAEPropagation {
	DifferentiableMatrixFunction f;
	DoubleMatrix GW1, GW2, GW3, GW4, GWCat, GWe_total;
	DoubleMatrix Gb1, Gb2, Gb3, Gbcat;
	int HiddenSize, DictionaryLength, CatSize;
	double AlphaCat, Beta;
	Tree tree;
	
	private RAEPropagation(double AlphaCat, double Beta, int HiddenSize, 
			int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this.AlphaCat = AlphaCat;
		this.Beta = Beta;
		this.HiddenSize = HiddenSize;
		this.DictionaryLength = DictionaryLength;
		this.f = f;
		initializeGradients( );
	}
	
	public RAEPropagation(double AlphaCat, double Beta, int HiddenSize, int CatSize,
								int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this.AlphaCat = AlphaCat;
		this.Beta = Beta;
		this.HiddenSize = HiddenSize;
		this.DictionaryLength = DictionaryLength;
		this.f = f;
		this.CatSize = CatSize;
		initializeFineGradients( );		
	}
	
	public RAEPropagation(Theta theta, double AlphaCat, double Beta, int HiddenSize, 
						int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this(AlphaCat,Beta,HiddenSize,DictionaryLength,f);		
	}
	
	public RAEPropagation(FineTunableTheta theta, double AlphaCat, double Beta, int HiddenSize, 
						int DictionaryLength, DifferentiableMatrixFunction f)
	{
		this(AlphaCat,Beta,HiddenSize, theta.Wcat.rows, DictionaryLength,f);
	}
	
	/**
	 * Building the tree / kids and forward propagation
	 * @param WordsEmbedded
	 * @param Freq
	 * @param CurrentLabel
	 * @param SentenceLength
	 * @return
	 */
	public Tree ForwardPropagate(Theta theta, DoubleMatrix WordsEmbedded, FloatMatrix Freq, 
						int CurrentLabel, int SentenceLength)
	{
		tree = new Tree( SentenceLength, HiddenSize, WordsEmbedded );
		ArrayList<Integer> CollapsedSentence = new ArrayList<Integer>( SentenceLength );
		for(int i=0; i<SentenceLength; i++)		
			CollapsedSentence.add(i);
		
		for(int j=0; j<SentenceLength-1; j++)
		{
			int NumComponents = WordsEmbedded.columns;
			
			/** C1 and C2 are matrices, not vectors. Therefore, we use C1 instead of c1
			 * Since this is the training phase, we compute the reconstruction error
			 * for each pair of words, i.e., C1(:,i) and C2(:,i) for each i indicates
			 * one pair. 
			 * However note that in the last iteration, C1 and C2 are column vectors
			 **/
			DoubleMatrix C1 = WordsEmbedded.getColumns( ArraysHelper.makeArray(0,NumComponents-2) );
			DoubleMatrix C2 = WordsEmbedded.getColumns( ArraysHelper.makeArray(1,NumComponents-1) );

//			FloatMatrix Freq1 = Freq.getColumns( makeArray(0,NumComponents-2) );
//			FloatMatrix Freq2 = Freq.getColumns( makeArray(1,NumComponents-1) );
	        
	        DoubleMatrix ActivationInp = ( theta.W1.mmul(C1) ).add( theta.W2.mmul(C2) ); //W1*c1 + W2*c2
	        DoubleMatrix P = f.valueAt(ActivationInp.addiColumnVector( theta.b1 ));
	        
	        /** Internal representation **/
	        DoubleMatrix PNorm = DoubleMatrixFunctions.ColumnWiseNormalize(P);

	        /** Deviation in naming convention in the original source. 
	         *  It is now uniform and can be understood that the variable without
	         *  "norm" suffixed to their names may be unnormalized
	         **/
	        DoubleMatrix Y1 = f.valueAt( (theta.W3.mmul(PNorm) ).addColumnVector(theta.b2) );
	        DoubleMatrix Y2 = f.valueAt( (theta.W4.mmul(PNorm) ).addColumnVector(theta.b3) );
	        
	        /** Reconstruction of C1 and C2 **/
	        DoubleMatrix Y1Norm = DoubleMatrixFunctions.ColumnWiseNormalize(Y1);
	        DoubleMatrix Y2Norm = DoubleMatrixFunctions.ColumnWiseNormalize(Y2);
	        
	        /** Reconstruction error **/
	        DoubleMatrix Y1MinusC1 = Y1Norm.sub(C1);
	        DoubleMatrix Y2MinusC2 = Y2Norm.sub(C2);
	        
	        DoubleMatrix Y1C1 = Y1MinusC1.mul(AlphaCat);
	        DoubleMatrix Y2C2 = Y2MinusC2.mul(AlphaCat);
	        
	        /** Equation (4) in the paper: reconstruction error (row vector) **/
	        DoubleMatrix J = ( ( ( Y1C1.mul(Y1MinusC1) ).add( Y2C2.mul(Y2MinusC2) ) ).columnSums() ).muli(0.5);
	        //System.out.println("J : " + J.rows + " " + J.columns + " [ " + NumComponents);
	        /** finding the pair with smallest reconstruction error for constructing tree **/
	        double J_min = J.min();
	        int J_minpos = J.argmin();
	        
	        //System.out.println(J_min + " " + J_minpos);
	        
	        Node NewParent = tree.T[ SentenceLength + j ];
	        NewParent.Y1C1 = Y1C1.getColumn(J_minpos);
	        NewParent.Y2C2 = Y2C2.getColumn(J_minpos);
	        NewParent.DeltaOut1 = f.derivativeAt( Y1.getColumn(J_minpos) ).mmul( Y1C1.getColumn(J_minpos) );
	        NewParent.DeltaOut2 = f.derivativeAt( Y2.getColumn(J_minpos) ).mmul( Y2C2.getColumn(J_minpos) );
	        NewParent.Features = PNorm.getColumn(J_minpos);
	        NewParent.UnnormalizedFeatures = P.getColumn(J_minpos);
	        NewParent.score = J_min;
	        tree.TotalScore += J_min;
	        
	        //System.out.println("Delta size : " + NewParent.DeltaOut1.rows + " " + NewParent.DeltaOut1.columns);
	        
	        int LeftChildIndex = CollapsedSentence.get(J_minpos),
	        		RightChildIndex = CollapsedSentence.get(J_minpos+1);
	        
	        Node LeftChild = tree.T[ LeftChildIndex ],
	        		RightChild = tree.T[ RightChildIndex ];
	        
	        NewParent.LeftChild = LeftChild;
	        NewParent.RightChild = RightChild;
	        NewParent.SubtreeSize = LeftChild.SubtreeSize + RightChild.SubtreeSize;
	        
	        LeftChild.parent = NewParent;
	        RightChild.parent = NewParent;
	        tree.structure.set(SentenceLength+j, new Pair<Integer,Integer>(LeftChildIndex, RightChildIndex));
	        		
//	        freq(J_minpos+1) = [];
//	        freq(J_minpos) = (Tree.numkids(Tree.kids(sl+j,1))*freq1(J_minpos) + Tree.numkids(Tree.kids(sl+j,2))*freq2(J_minpos))/(Tree.numkids(Tree.kids(sl+j,1))+Tree.numkids(Tree.kids(sl+j,2)));
//	        
	        WordsEmbedded = UpdateEmbedding(WordsEmbedded,J_minpos,PNorm.getColumn(J_minpos));
	        CollapsedSentence.remove(J_minpos+1);
	        CollapsedSentence.set(J_minpos, SentenceLength + j);
		}
		return tree;
	}
	
	/**
	 * Returning the classification error for the given tree.
	 * @param Kid
	 * @param WordsEmbedded
	 * @param Freq
	 * @param CurrentLabel
	 * @param SentenceLength
	 * @return
	 */
	public Tree ForwardPropagate(FineTunableTheta theta, DoubleMatrix WordsEmbedded, FloatMatrix Freq, 
						int CurrentLabel, int SentenceLength, Structure TreeStructure)
	{
		int CatSize = theta.Wcat.columns;
		int TreeSize = 2 * SentenceLength - 1;
		tree = new Tree( SentenceLength, HiddenSize, CatSize, WordsEmbedded );
		int[] SubtreeSize = new int[ TreeSize ];

		for(int i=SentenceLength; i< TreeSize; i++)
		{
			int LeftChild = TreeStructure.get(i).getFirst(),
					RightChild = TreeStructure.get(i).getSecond();
			
			SubtreeSize[i] = SubtreeSize[ LeftChild ] + SubtreeSize[ RightChild ];
		}

//	    classifier on single words
		Sigmoid SigmoidCalc = new Sigmoid();
	    DoubleMatrix SM = SigmoidCalc.valueAt( (theta.Wcat.mmul(WordsEmbedded).addColumnVector(theta.bcat) ) );
	    DoubleMatrix Diff = SM.sub(CurrentLabel);
	    DoubleMatrix SquaredError = (Diff.mul(Diff)).mul((1-AlphaCat)*0.5f);
	    DoubleMatrix ErrorGradient = Diff.mul(1-AlphaCat).mul( SigmoidCalc.derivativeAt(SM) );
	    
	   for(int i=0; i<TreeSize; i++)
	   {
		   Node CurrentNode = tree.T[i];
		   if( i<SentenceLength)
		   {
			 //sum is just for converting to double. getColumn(i) should return only one value
			   CurrentNode.score = SquaredError.getColumn(i).sum();  
			   CurrentNode.catDelta = ErrorGradient.getColumn(i);
		   }
		   else
		   {   
			   int LeftChild = TreeStructure.get(i).getFirst(),
					   RightChild = TreeStructure.get(i).getSecond();
			   DoubleMatrix C1 = tree.T[LeftChild].Features,
					   C2 = tree.T[RightChild].Features;
			   
			   CurrentNode.LeftChild = tree.T[LeftChild];
			   CurrentNode.RightChild = tree.T[RightChild];
			   CurrentNode.LeftChild.parent = CurrentNode;
			   CurrentNode.RightChild.parent = CurrentNode;
			   
			   // Eq. (2) in the paper: p = f(W(1)[c1; c2] + b(1))
			   DoubleMatrix p = f.valueAt( ((theta.W1.mmul(C1)).addi(theta.W2.mmul(C2))).addColumnVector(theta.b1) );
		        
			   // See last paragraph in Section 2.3
			   DoubleMatrix pNorm1 = p.div(p.norm2());
			   
			   CurrentNode.UnnormalizedFeatures = p;
			   CurrentNode.Features = pNorm1;
			   
		       // Eq. (7) in the paper (for special case of 1d label)
			   SM = SigmoidCalc.valueAt( (theta.Wcat.mmul(pNorm1)).addColumnVector(theta.bcat) );
			   Diff = SM.sub(CurrentLabel);
		       CurrentNode.catDelta = (Diff.mul(Beta*(1-AlphaCat))).mul(SigmoidCalc.derivativeAt(SM));  
		       CurrentNode.score = DoubleMatrixFunctions.SquaredNorm(Diff) * 0.5 * Beta* (1-AlphaCat);
		       
		       CurrentNode.SubtreeSize = SubtreeSize[i];
		   }
		   tree.TotalScore += CurrentNode.score;
	   }
	    tree.structure = TreeStructure;
		return tree;
	}
	
	public void BackPropagate(Tree tree, Theta theta, int[] WordsIndexed)
	{
		int SentenceLength = WordsIndexed.length;
		if( tree.T.length != 2*SentenceLength - 1 )
			System.err.println("Bad Tree for backpropagation!");
		
		DoubleMatrix GL = DoubleMatrix.zeros(HiddenSize, SentenceLength);
		
		// Stack of currentNode, Left(1) or Right(2), Parent Node pointer 
		Stack<Triplet<Node,Integer,Node>> ToPopulate = new Stack<Triplet<Node,Integer,Node>>();
		
		Node Root = tree.T[ tree.T.length-1 ];
		Root.ParentDelta = DoubleMatrix.zeros(HiddenSize,1);
		ToPopulate.push( new Triplet<Node,Integer,Node>(Root,0,null));
		
		DoubleMatrix[] W = new DoubleMatrix[3];
		W[0] = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		W[1] = theta.W1;
		W[2] = theta.W2;
		DoubleMatrix Y0C0 = DoubleMatrix.zeros(HiddenSize,1);
		
		while( !ToPopulate.empty() )
		{
			Triplet<Node,Integer,Node> top = ToPopulate.pop();
			int LeftOrRight = top.getSecond();
			Node CurrentNode = top.getFirst(), ParentNode = top.getThird();
			DoubleMatrix[] YCSelector = null;
			if(ParentNode == null)
				YCSelector = new DoubleMatrix[]{ Y0C0, null, null };
			else
				YCSelector = new DoubleMatrix[]{ Y0C0, ParentNode.Y1C1, ParentNode.Y2C2 };
			
			DoubleMatrix NodeW = W[ LeftOrRight ];
			DoubleMatrix delta = YCSelector[ LeftOrRight ];
			
			if(!CurrentNode.isLeaf())
			{
				ToPopulate.push(new Triplet<Node,Integer,Node>(CurrentNode.LeftChild,1,CurrentNode));
				ToPopulate.push(new Triplet<Node,Integer,Node>(CurrentNode.RightChild,2,CurrentNode));
				
                DoubleMatrix A1 = CurrentNode.UnnormalizedFeatures, A1Norm = CurrentNode.Features;
                DoubleMatrix ND1 = CurrentNode.DeltaOut1, ND2 = CurrentNode.DeltaOut2;
                DoubleMatrix PD = CurrentNode.ParentDelta; 
               
                DoubleMatrix Activation = ((theta.W3.transpose()).mmul(ND1)).addi((theta.W4.transpose()).mmul(ND2));
                Activation = Activation.addi( ((NodeW.transpose()).mmul(PD)).subi(delta) );
                DoubleMatrix CurrentDelta = f.derivativeAt(A1).mmul(Activation); 

                CurrentNode.LeftChild.ParentDelta = CurrentDelta;
                CurrentNode.RightChild.ParentDelta = CurrentDelta;
                
                GW1.addi( (CurrentDelta.mmul(CurrentNode.LeftChild.Features.transpose()) ));
                GW2.addi( (CurrentDelta.mmul(CurrentNode.RightChild.Features.transpose()) ));
                GW3.addi( ND1.mmul(A1Norm.transpose()));
                GW4.addi( ND2.mmul(A1Norm.transpose()));                
        		
	            Gb1.addi(CurrentDelta);
	            Gb2.addi(ND1);
	            Gb3.addi(ND2);
			}
			else
			{
				DoubleMatrixFunctions.IncrementColumn(GL, CurrentNode.NodeName, 
						((NodeW.transpose()).mmul( CurrentNode.ParentDelta )).subi( delta ));
			}
		}
		
		for(int l=0; l<SentenceLength; l++)
			DoubleMatrixFunctions.IncrementColumn(GWe_total,WordsIndexed[l],GL.getColumn(l));
	}
	
	public void BackPropagate(Tree tree, FineTunableTheta theta, int[] WordsIndexed)
	{
		int SentenceLength = WordsIndexed.length;
		if( tree.T.length != 2*SentenceLength - 1 )
			System.err.println("Bad Tree for backpropagation!");
		
		DoubleMatrix GL = DoubleMatrix.zeros(HiddenSize, SentenceLength);
		
		// Stack of currentNode, Left(1) or Right(2), Parent Node pointer 
		Stack<Triplet<Node,Integer,Node>> ToPopulate = new Stack<Triplet<Node,Integer,Node>>();
		
		Node Root = tree.T[ tree.T.length-1 ];
		Root.ParentDelta = DoubleMatrix.zeros(HiddenSize,1);
		ToPopulate.push( new Triplet<Node,Integer,Node>(Root,0,null));
		
		DoubleMatrix[] W = new DoubleMatrix[3];
		W[0] = DoubleMatrix.zeros(HiddenSize, HiddenSize);
		W[1] = theta.W1;
		W[2] = theta.W2;
		DoubleMatrix Y0C0 = DoubleMatrix.zeros(HiddenSize,1);
		
		while( !ToPopulate.empty() )
		{
			Triplet<Node,Integer,Node> top = ToPopulate.pop();
			int LeftOrRight = top.getSecond();
			Node CurrentNode = top.getFirst(), ParentNode = top.getThird();
			DoubleMatrix[] YCSelector = null;
			if(ParentNode == null)
				YCSelector = new DoubleMatrix[]{ Y0C0, null, null };
			else
				YCSelector = new DoubleMatrix[]{ Y0C0, ParentNode.Y1C1, ParentNode.Y2C2 };
			
			DoubleMatrix NodeW = W[ LeftOrRight ];
			DoubleMatrix delta = YCSelector[ LeftOrRight ];
			
			if(!CurrentNode.isLeaf())
			{
				ToPopulate.push(new Triplet<Node,Integer,Node>(CurrentNode.LeftChild,1,CurrentNode));
				ToPopulate.push(new Triplet<Node,Integer,Node>(CurrentNode.RightChild,2,CurrentNode));
				
                DoubleMatrix A1 = CurrentNode.UnnormalizedFeatures, A1Norm = CurrentNode.Features;
                DoubleMatrix ND1 = CurrentNode.DeltaOut1, ND2 = CurrentNode.DeltaOut2;
                DoubleMatrix PD = CurrentNode.ParentDelta; 

                Gbcat = Gbcat.addi( CurrentNode.catDelta );
                GWCat = GWCat.addi( CurrentNode.catDelta.mmul(A1Norm.transpose()) );
                
                DoubleMatrix Activation = ((theta.W3.transpose()).mmul(ND1)).addi((theta.W4.transpose()).mmul(ND2));
                Activation.addi( ((NodeW.transpose()).mmul(PD)).addi((theta.Wcat.transpose()).mmul(CurrentNode.catDelta)) );
                Activation.subi(delta);
                DoubleMatrix CurrentDelta = f.derivativeAt(A1).mmul(Activation); 

                CurrentNode.LeftChild.ParentDelta = CurrentDelta;
                CurrentNode.RightChild.ParentDelta = CurrentDelta;
                
                GW1.addi( (CurrentDelta.mmul(CurrentNode.LeftChild.Features.transpose()) ));
                GW2.addi( (CurrentDelta.mmul(CurrentNode.RightChild.Features.transpose()) ));
                GW3.addi( ND1.mmul(A1Norm.transpose()));
                GW4.addi( ND2.mmul(A1Norm.transpose()));                
        		
	            Gb1.addi(CurrentDelta);
	            Gb2.addi(ND1);
	            Gb3.addi(ND2);
			}
			else
			{
                GWCat = GWCat.addi( CurrentNode.catDelta.mmul( CurrentNode.Features.transpose() ) );
                Gbcat = Gbcat.addi( CurrentNode.catDelta );

				DoubleMatrixFunctions.IncrementColumn(GL, CurrentNode.NodeName, 
						(((NodeW.transpose()).mmul( CurrentNode.ParentDelta ))
								.addi( theta.Wcat.transpose().mmul(CurrentNode.catDelta)) )
								.subi( delta ));
			}
		}
		
		for(int l=0; l<SentenceLength; l++)
			DoubleMatrixFunctions.IncrementColumn(GWe_total,WordsIndexed[l],GL.getColumn(l));		
	}
	
	/**
	 * Pre-allocating memory for the gradients
	 * Line 18-28 in /ComputeCostAndGradRAE.m
	 * @param Theta
	 */
	private void initializeGradients( )
	{
		GW1 = new DoubleMatrix(HiddenSize, HiddenSize);
		GW2 = new DoubleMatrix(HiddenSize, HiddenSize);
		GW3 = new DoubleMatrix(HiddenSize, HiddenSize);
		GW4 = new DoubleMatrix(HiddenSize, HiddenSize);
		GWe_total = new DoubleMatrix(HiddenSize, DictionaryLength);
		
		Gb1 = new DoubleMatrix(HiddenSize, 1);
		Gb2 = new DoubleMatrix(HiddenSize, 1);
		Gb3 = new DoubleMatrix(HiddenSize, 1);		
	}
	
	/**
	 * Pre-allocating memory for the gradients
	 * Line 18-28 in /ComputeCostAndGradRAE.m
	 * @param Theta
	 */
	private void initializeFineGradients( )
	{
		initializeGradients( );	
		GWCat = new DoubleMatrix(CatSize, HiddenSize);
		Gbcat = new DoubleMatrix(CatSize, 1);		
	} 

	private DoubleMatrix UpdateEmbedding(DoubleMatrix CurrentEmbedding, int Column, DoubleMatrix ColumnVector)
	{
		int[] leftColumns = ArraysHelper.makeArray(0,Column-1);
		int[] rightColumns = ArraysHelper.makeArray(Column+2,CurrentEmbedding.columns-1);
		
		if( leftColumns == null && rightColumns == null )
			return ColumnVector;
		else if( leftColumns == null )
			return DoubleMatrix.concatHorizontally(ColumnVector,CurrentEmbedding.getColumns(rightColumns));
		else if( rightColumns == null )
			return DoubleMatrix.concatHorizontally(CurrentEmbedding.getColumns(leftColumns),ColumnVector);
		else	
			return DoubleMatrix.concatHorizontally(CurrentEmbedding.getColumns(leftColumns), 
								DoubleMatrix.concatHorizontally(ColumnVector,
													CurrentEmbedding.getColumns(rightColumns)));
	}

	public static void main(String[] args) throws Exception
	{
		double alphaCat = 0.2, beta = 0.5; 
		int hiddenSize = 50, DictionarySize = 14043;
		DifferentiableMatrixFunction f = new Norm1Tanh();
		
		int[] data = new int[]{1, 0, 2315, 2914, 5, 13, 1, 7718, 5, 1, 5743, 13, 10718, 15, 141, 760, 14, 7, 3293, 5, 2245, 93, 25, 4236, 3922, 0, 9725, 0, 12, 1999, 2973, 5, 0, 3, 12800, 3, 12800, 3, 0, 12, 0, 3};
		String dir = "data/parsed";
		FileInputStream fis = new FileInputStream(dir + "/opttheta.dat");
		ObjectInputStream ois = new ObjectInputStream(fis);
		FineTunableTheta tunedTheta = (FineTunableTheta) ois.readObject();
		ois.close();
		
		RAEPropagation prop = new RAEPropagation(alphaCat, beta, hiddenSize, DictionarySize, f);
		DoubleMatrix WordsEmbedded = tunedTheta.We.getColumns(data);
		
//		for(int i=0; i<5; i++){
//			for(int j=0; j<5; j++)
//				System.out.printf("%.4f ",tunedTheta.We.get(i,j));
//			System.out.println();
//		}
//		System.out.println();System.out.println();
//		
//		for(int i=0; i<5; i++){
//			for(int j=0; j<5; j++)
//				System.out.printf("%.4f ",WordsEmbedded.get(i,j));
//			System.out.println();
//		}
//		System.out.println();System.out.println();
		
		Tree tree = prop.ForwardPropagate(tunedTheta, WordsEmbedded, null, 0, data.length);
//		DoubleMatrix tf = new DoubleMatrix(hiddenSize,2*data.length-1);
//		for(int i=0; i<2*data.length-1; i++)
//		{
//			tf.putColumn(i, tree.T[i].Features);
//			if( tree.T[i].Features == null )
//				System.err.println("First prop returns NULL");
//		}
//		System.out.printf("%d done\n",2 * data.length - 1 );
		System.out.printf("%.4f \n", tree.TotalScore);
//		DoubleMatrixFunctions.prettyPrint(tf);	
		
		
		prop.BackPropagate(tree, (Theta)tunedTheta, data);
		
		prop = new RAEPropagation(alphaCat, beta, hiddenSize, 1, DictionarySize, f);
		
		tree = prop.ForwardPropagate(tunedTheta, WordsEmbedded, null, 0, data.length, tree.structure);
		prop.BackPropagate(tree, tunedTheta, data);
		
		DoubleMatrixFunctions.prettyPrint(prop.GW1);
		DoubleMatrixFunctions.prettyPrint(prop.GW2);
		DoubleMatrixFunctions.prettyPrint(prop.GW3);
		DoubleMatrixFunctions.prettyPrint(prop.GW4);
		
		DoubleMatrixFunctions.prettyPrint(prop.Gb1);
		DoubleMatrixFunctions.prettyPrint(prop.Gb2);
		DoubleMatrixFunctions.prettyPrint(prop.Gb3);
		
		DoubleMatrixFunctions.prettyPrint(prop.GWCat);
		DoubleMatrixFunctions.prettyPrint(prop.Gbcat);
		
		

	}

}
