
somalugares=zeros( 1, length(memBestSolution));
for ii=1:length(memBestSolution)
    iii=1;
    for jj=1:22
        if memBestSolution(ii,1)==1
            if memBestSolution(ii,2)==jj
             somalugares(iii,jj)=somalugares(iii,jj)+1;
            end;
        end;
    end;
end;
somalugares