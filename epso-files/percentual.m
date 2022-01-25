%calculo do percentual função é chamada pelo MainEPSO_SIMA.m
function [vetorpercentual] = percentual(vetormaxrunconsolidado)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
global vetorsensibilidade;
load rota_sima_ufpa;


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

htt=[15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    15
    ];
     %vetorsensibilidade=-107;
           Pt=20;
           pr=vetorsensibilidade;
            
           x=[28.405 9.2 43.698 23.754];
           %x=[26.938 17.732 6.678 6.16];
            %x = [16.5 14.2 42.5 7.6];

           f=915e6;%freq em Hz
           hr = 3;
           lamb = 3e8/f;
           fM = f*1e-6;
  

      for io=1:size(vetormaxrunconsolidado,1)
          io
          numaps=vetormaxrunconsolidado(io,2);
          vetorotimAP=vetormaxrunconsolidado(io,3:numaps+2);
                 
           for p=1:length(vetorotimAP)
             
                 ht=htt(vetorotimAP(p));
                 raiotemp=10^((Pt-pr-x(2)*log10(fM) - x(3) + x(4)*((ht + hr)*lamb)./(0.1*62))/x(1))
                
           end
           
           for j=1:length(vetorotimAP)
    
                PosicaoAps = vetorotimAP(j);
                Posicaoreallong(j) = Y(PosicaoAps,1);
                Posicaoreallat(j) = Y(PosicaoAps,2);
                ht(j)=htt(PosicaoAps);
           end
           for kk=1:length(X)
            intercobertura(kk,1)= X(kk,2)*pi/180;%lat1
            intercobertura(kk,2)= X(kk,1)*pi/180;%lat1
            intercobertura(kk,3)= 0;%lat1
           end;

           contadordecobertura=0;
            m=1;
            for p=1:length(vetorotimAP)
             contadordecobertura=0;
             for k=1:length(X)
                 radius = 6371;
                 lat1 = X(k,2)*pi/180;
                 lat2 = Posicaoreallat(p)*pi/180;
                 lon1 = X(k,1)*pi/180;
                 lon2 = Posicaoreallong(p)*pi/180;
                 deltaLat = lat2-lat1;
                 deltaLon = lon2-lon1;
                 a = sin((deltaLat)/2)^2 + cos(lat1)*cos(lat2)*sin(deltaLon/2)^2;
                 c = 2*atan2(sqrt(a),sqrt(1-a));
                 distancia(k) = radius*c;    %Haversine distance
                 x=[28.405 9.2 43.698 23.754];
                 % x = [16.5 14.2 42.5 7.6];
                 %ht = 15;
                 hb=ht;
                 hr = 3;
                 hm=hr;
                 if distancia(k)>0
                 pr = Pt - (UFPA(x, ht(p), hr, f, (distancia(k)*1000)));
                   if pr>=vetorsensibilidade
                      pr2(k) = pr; 
                      did(k) = distancia(k);
                      intercobertura(k,3)=1;    
                      contadordecobertura=contadordecobertura+1;
                   end
                 end
             end
            indicedecobertura(p)=(contadordecobertura/(length(X)))*100;
         end
        
         tamanhodacobertura=length(intercobertura);       
         percentualcalculado=(sum(intercobertura(:,3))/tamanhodacobertura)*100
        vetorpercentual(io,1)= percentualcalculado;
      end; 
      
% figure;
% yyaxis left
% plot(1:size(vetormaxrunconsolidado,1),vetormaxrunconsolidado(:,3:size(vetormaxrunconsolidado,2)),'ok')
% ylim([1 24]);
% xlim([0 36]);
% yyaxis right
% plot(1:size(vetormaxrunconsolidado,1),vetorpercentual,'p-r')
% ylim([0 110])
% grid on
% grid minor
         
end

