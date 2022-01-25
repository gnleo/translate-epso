target=[5     7    13 
        7    13     5
        13    19     7 
        7     5    13 
        7    13    19
        5    13     7 
        13     5     7
        13     7    19 
        19     7    13
        13     7     5]
           
        
somalugares=zeros( 1, length(target));
for ii=1:length(memBestSolution)
    iii=1;
    for jj=1:length(target)
        if memBestSolution(ii,1)==3
            aa=strcat(num2str(target(jj,2)),num2str(target(jj,3)));
            aaa=strcat(num2str(target(jj,1)),aa);
            targetnumconcat=str2num(aaa);
            bb=strcat(num2str(memBestSolution(ii,3)),num2str(memBestSolution(ii,4)));
            bbb=strcat(num2str(memBestSolution(ii,2)),bb);
            memnumconcat=str2num(bbb);
            
            if memnumconcat==targetnumconcat
             somalugares(iii,jj)=somalugares(iii,jj)+1;
            end;
        end;
    end;
end;
target
somalugares