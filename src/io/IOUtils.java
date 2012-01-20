package io;

import java.io.*;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.util.List;
import java.util.ArrayList;

/**
 * Utilities for getting files recursively, with a filter.
 * 
 * @author Dan Klein
 * @author Sanjeev from http://stackoverflow.com/questions/453018/number-of-lines-in-a-file-in-java
 */
public class IOUtils {
	public static List<File> getFilesUnder(String path, FileFilter fileFilter) {
		File root = new File(path);
		List<File> files = new ArrayList<File>();
		addFilesUnder(root, files, fileFilter);
		return files;
	}

	private static void addFilesUnder(File root, List<File> files,
			FileFilter fileFilter) {
		if (!fileFilter.accept(root))
			return;
		if (root.isFile()) {
			files.add(root);
			return;
		}
		if (root.isDirectory()) {
			File[] children = root.listFiles();
			for (int i = 0; i < children.length; i++) {
				File child = children[i];
				addFilesUnder(child, files, fileFilter);
			}
		}
	}

	/**
	 * @param filename Input filename
	 * @return Counts the number of lines in the file, excluding empty lines. 
	 * @throws IOException
	 */
	public static int countLines(String Filename) throws IOException{
	    File file = new File(Filename);
	    FileChannel fc = (new FileInputStream(file)).getChannel(); 
	    MappedByteBuffer buf = fc.map(MapMode.READ_ONLY, 0, file.length());
	    boolean emptyLine = true;
	    int counter = 0;
	    byte element = '\0';
	    
	    while (buf.hasRemaining())
	    {
	        element = buf.get();

	        if (element == '\r' || element == '\n') {
	            if (!emptyLine) 
	            {
	                counter += 1;
	                emptyLine = true;
	            }
	        } else 
	            emptyLine = false;
	    }
	    
	    if(element != '\r' && element != '\n')
	    	counter += 1;
	    
	    return counter;
	}

}
