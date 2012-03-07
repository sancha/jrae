package io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

public abstract class StringIntegerMap extends HashMap<String,Integer> 
{
	private static final long serialVersionUID = 3211964388907840234L;

	public void loadFile (String fileName) throws IOException
	{
		FileInputStream fstream = new FileInputStream(fileName);
		DataInputStream in = new DataInputStream(fstream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));
		String strLine;
		while ((strLine = br.readLine()) != null) {
			int lastSpaceIndex = strLine.lastIndexOf(' ');
			String token = strLine.substring(0, lastSpaceIndex);
			Integer map = Integer.parseInt( strLine.substring(lastSpaceIndex+1) );
			put (token, map);
		}
		in.close();
	}
	
	public String getReverseMapping (int val)
	{
		for (String key : keySet())
			if (get (key) == val)
				return key;
		return null;		
	}
	
	public int getMapping (String key)
	{
		if (keySet ().contains (key))
			return get (key);
		return -1;
	}
	
	public void saveToFileName (String fileName)
	{
		try {
			// Create file
			FileWriter fstream = new FileWriter(fileName);
			BufferedWriter out = new BufferedWriter(fstream);

			for (String word : keySet())
				out.write(word + " " + get(word) + "\n");

			out.close();
		} catch (Exception e) {// Catch exception if any
			System.err.println("Could not write the file : " + fileName);
			System.err.println("Error: " + e.getMessage());
		}
	}
}
