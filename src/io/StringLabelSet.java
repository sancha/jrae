package io;

import java.io.IOException;

public class StringLabelSet extends StringIntegerMap
{
	private static final long serialVersionUID = 184711325137903778L;

	public StringLabelSet ()
	{
		
	}
	
	public StringLabelSet (String labelsetFileName) throws IOException{
		super.loadFile(labelsetFileName);
	}
	
	public int getLabel(String label) {
		return super.getMapping(label);
	}

	public String getLabelString(int label) {
		return super.getReverseMapping(label);
	}
	
	public void saveToFile(String labelSetFileName) {
		super.saveToFileName(labelSetFileName);
	}
}
