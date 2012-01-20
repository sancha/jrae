 read_rtPolarity
 
 fid = fopen('wordmap.txt','w');
 for i=1:numel(words2)
 	 fprintf(fid,'%d %s\n',reIndexMap(wordMap(words2{i}))-1,words2{i});
 end
 fclose(fid);
 
 fid = fopen('data.txt','w')
 for i=1:numel(labels)
 	fprintf(fid,'%d %d ',labels(i),allSNum{i}-1);
   fprintf(fid,'\n');
 end
 fclose(fid);
  