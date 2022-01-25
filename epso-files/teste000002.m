%grafico de frequencia em barra , calcula os grupos mais frequentes

vetortemp=vetormaxrunconsolidadow8;
 teste=vetortemp(:,3:size(vetortemp,2));
 numtemp=size(teste,2)+1;
 vetorzero=zeros(size(vetortemp,1),1);
 vetorfreq=zeros(2,numtemp);
 %vetorfreq=zeros(size(vetortemp,1),numtemp);
 ii=1;
 for i=1:length(teste)
     conta=1;
    testea=sort(teste(i,:));
    
    for j=1:size(teste,1)
        if i~=j
        testeb=sort(teste(j,:));
        tf=isequal(testea,testeb);
        if ((tf==1)&(vetorzero(j,:)==0))
            conta=conta+1;
            vetorzero(j,:)=1;
        end;
        end;
    end;
    if conta>=2
    vetorfreq(ii,:)=[testea conta];
    ii=ii+1;
    else 
        if ((conta==1)&&(vetorzero(i,:)==0))
             vetorfreq(ii,:)=[testea conta];
             ii=ii+1;
        end;
    end;
    vetorzero(i,:)=1;
    
 end; 
 vetorfreq
 totalvetorfreq=sum(vetorfreq(:,size(vetorfreq,2)));
 vetorfreqperc=(vetorfreq(:,size(vetorfreq,2))/totalvetorfreq)*100;
 
for i=1:size(vetorfreq,1)
    tempp=vetorfreq(i,1:(size(vetorfreq,2)-1));
    for j=1:(size(vetorfreq,2)-1)
        if tempp(1,j)==0
            vetorab(i,1)=j+1;
        else if tempp(1,1)~=0
                 vetorab(i,1)=1;
             end;
        end;
    end;
    end;
clear  dd ee ccc;
limiteup=size(vetorfreq,1);

for hh=1:size(vetorfreq,1)
dd(hh,1)=cellstr(num2str(vetorfreq(hh,vetorab(hh,1):(size(vetorfreq,2)-1))));
ee(hh,1)=cellstr(num2str(vetorfreq(hh,vetorab(hh,1):(size(vetorfreq,2)-1))));
end;

ccc = categorical(dd,ee);
figure;
bar(ccc,vetorfreqperc(1:limiteup),0.5);
title('Frequency of Optimal Solutions for Pareto 8 (%)');
xlabel('Opimal Solutions for Gateways Placement')
ylabel('Percentage (%)')

