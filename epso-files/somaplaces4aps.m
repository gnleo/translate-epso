%clc, clear all
clear vetornovo2 somalugares vetornovo


Y = [ 
-48.456547 -1.476941 %Mirante do rio 1
-48.450713  -1.472760 %itec 2
-48.455196 -1.475772 %Reitoria 3
-48.455485 -1.473413 %Ginasio 4
-48.449751, -1.471206 %ICJ 5
-48.458282 -1.474910 %Geociências 6
-48.456951 -1.475333 %poste Basico 7
-48.453613 -1.475561 %Lab. Eng. Quimica 8
-48.453721 -1.474568 %PGitec 9 
-48.453625 -1.475642 %predio ao lado do LEEC 10
-48.451395 -1.473113 %Poste estacionamento profissional 11
-48.4453 -1.463583333 %PCT 12
-48.44536111 -1.464191667 %Espaço Inovação 13
-48.453874 -1.475561 %Ceamazon 14
-48.44614444 -1.466669444 %Lab. Qualidade do Leite 15
-48.44794167 -1.468322222 %Centro de Genomica e Biologia de Sistemas 16
-48.44791944 -1.468741667 %Ao lado do ponto acima 17
-48.446675 -1.469397222 %Bettina 18
-48.44728611 -1.470936111 %Faculdade de Odontologia 19
-48.44614722 -1.471119444 %Faculdade de Fisioterapia 20
-48.44504167 -1.470361111 %Faculdade de Engenharia Naval 21
-48.443924 -1.470306];  %Lab. Eng. Naval 22

if memBestSolution(1,2)==1
somalugares=zeros( 1, length(Y));
memBestSolution
for ii=1:length(memBestSolution)
    iii=1;
     for jj=2:length(Y)
        if memBestSolution(ii,2)==1
            if memBestSolution(ii,2)==1
             somalugares(iii,jj)=somalugares(iii,jj)+1;
            somalugares(iii+1,jj)=jj;
            end
       end
     end
end
somalugares'
end;
if memBestSolution(1,2)>1
    
%memBestSolution
num=size(memBestSolution);
for ii=1:num(1)
        if memBestSolution(ii,2)>1
            numaps=memBestSolution(ii,2);
            aa=num2str(memBestSolution(ii,numaps+2));
            t=1;
            tt=numaps+1;
            while t<=numaps
                aa=strcat(num2str(memBestSolution(ii,tt)),aa);
                tt=tt-1;
                t=t+1;
            end
            memnumconcat=str2num(aa);
            vetornovo(ii,1)= memnumconcat;
            vetornovo(ii,2)=0.0;
        end
end;

vetornovo

ii=1;
for i=1:length(vetornovo)
    ppp=vetornovo(i,1);
    flag=vetornovo(i,2);
    contadordepri=0;
      for j=1:length(vetornovo)
          if flag==0
          teste=vetornovo(j,1);
          flag2=vetornovo(j,2);
          if ppp==teste 
              if flag2==0
              contadordepri=contadordepri+1;
              vetornovo(i,2)=i;
              vetornovo(j,2)=i;
              end
          end
          end
      end
      if contadordepri>0
      vetornovo2(ii,:)=[ppp contadordepri];
      ii=ii+1;
      end;
end
vetornovo2
end;
