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


i=1;
copyPos(i,:)=[ 4     6    15     4     8    11     2     7    11     7     1];

cont=0;  
temigual=0;
        for ki=1:length(Y)
            for kj=1:(length(copyPos(i,:))-1)
                if copyPos(i,kj+1)==ki
                    cont=cont+1;
                end;
                if cont>=2
                    temigual=1;
                end;
            end;
            cont=0;
        end;
        tp=1;
         while temigual~=0
             t=1;
             valorteste=copyPos(i,tp+1);
             while (t<=(length(copyPos(i,:))-2))
                 valor2=copyPos(i,t+2);
                 if t+2~=tp+1
                  if valorteste==valor2
                     varcopy=randperm(length(Y));
                     varcopy=varcopy(1:1);
                     copyPos(i,t+2)=varcopy;    %copyPos(i,t+2)+copyVel(i,t+2);
                  end;
                 end;
                 t=t+1;
             end;
             temigual=0;
             cont=0;
           for ki=1:length(Y)
            for kj=1:(length(copyPos(i,:))-1)
                if copyPos(i,kj+1)==ki
                    cont=cont+1;
                end;
                if cont>=2
                    temigual=1;
                end;
            end;
            cont=0;
           end;
           tp=tp+1;
           if tp==(length(copyPos(i,:)))
               tp=1;
           end;
         end;