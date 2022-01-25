function [ fit ] = FITNESS_FUNCTION( popSize, pos, X, Y)
% Computes fitness fot the whole population
% Optimization functions can be obtained from: http://infinity77.net/global_optimization/index.html
global ff_par;
global W;
global vetorobj1 vetorobj2 vetorsensibilidade;
% htt=[30
%     6
%     30
%     15
%     9
%     6
%     9
%     6
%     12
%     6
%     12
%     12
%     30
%     15
%     6
%     6
%     6
%     9
%     6
%     6
%     6
%     6];

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
    15];


fit = zeros( 1, popSize );
for i = 1 : popSize
    switch ff_par.ff
        case 1
            % Sphere function
            fit( i ) = pos( i, : ) * pos( i, : );
        case 2
            % Rosenbrock function
            fit( i ) = 0;
            for j = 1 : ff_par.D - 1
                term1 = pos( i, j + 1 ) - pos( i, j )^2;
                term2 = 1 - pos( i, j );
                fit( i ) = fit( i ) + 100 * term1^2 + term2^2;
            end
        case 3
            % Shaffer 1 function
            fit( i ) = 0;
            for j = 1 : ff_par.D - 1
                term = pos( i, j + 1 ) * pos( i, j + 1 ) + pos( i, j ) * pos( i, j );
                fit( i ) = fit( i ) + ( sin( sqrt( term ) )^2 - 0.5 ) / ( 0.001 * term + 1 )^2 + 0.5;
            end
        case 4
            % Schwefel function
            fit( i ) = 0;
            for j = 1 : ff_par.D
                x = pos( i, j );
                fit( i ) = fit( i ) + x * sin( sqrt( abs( x ) ) );
            end
            fit( i ) = 418.9829 * ff_par.D - fit( i );
        case 5
            % Griewank function
            term1 = 0;
            term2 = 1;
            for j = 1 : ff_par.D
                x = pos( i, j );
                term1 = term1 + x^2;
                term2 = term2 * cos( x / sqrt( j ) );
            end
            fit( i ) = 1 + term1 / 4000 - term2;
        case 6
            % Rastrigin function
            fit( i ) = 0;
            for j = 1 : ff_par.D
                x = pos( i, j );
                fit( i ) = fit( i ) + x^2 - 10 * cos( 2 * pi * x );
            end
            fit( i ) = 10 * ff_par.D + fit( i ) ;
        case 7
            % Ackley function
            term1 = 0;
            term2 = 0;
            for j = 1 : ff_par.D
                x = pos( i, j );
                term1 = term1 + x^2;
                term2 = term2 + cos( 2 * pi * x );
            end
            fit( i ) = 20 - 20 * exp( -0.2 * sqrt( ( term1 / ff_par.D ) ) ) + exp( 1 ) - exp( term2 / ff_par.D );

        case 8
            % SIMA function

            Naps = pos(i,1);
            Pt=20;

            for j=1:Naps

                PosicaoAps = pos(i,j+1);
                Posicaoreallong(j) = Y(PosicaoAps,1);
                Posicaoreallat(j) = Y(PosicaoAps,2);
                ht(j)=htt(PosicaoAps);
            end
            for kk=1:length(X)
                intercobertura(kk,1)= X(kk,2)*pi/180;%lat1
                intercobertura(kk,2)= X(kk,1)*pi/180;%lat1
                intercobertura(kk,3)= 0;%lat1
            end;

            f=915e6;%freq em Hz
            contadordecobertura=0;
            m=1;
            for p=1:Naps
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
                    %x=[26.938 17.732 6.678 6.16];
                    %x = [16.5 14.2 42.5 7.6];
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
            percentualcalculado=(sum(intercobertura(:,3))/tamanhodacobertura)*100;
            Perc = 100;
            obj1=percentualcalculado;
            obj1 = Perc-percentualcalculado;
            %normalizando
            obj1=obj1/100;

            obj2 = Naps;
            %normalizando
            obj2=obj2/ff_par.Xmax(1);
            fit(i)=W*obj1+(1-W)*obj2;
            vetorobj1(i)=obj1;
            vetorobj2(i)=obj2;
            %          figure(15)
            %        plot(obj2*ff_par.Xmax(1),obj1*100,'ob');
            %        hold on
            %        grid on;
            %        xlabel('N�mero de Gateways');
            %        ylabel('�rea n�o coberta (%)');
            %        title('Solu��es Encontradas');
            %        xlim = [1 10];
        otherwise
            % This is never supposed to happen
    end
    ff_par.fitEval = ff_par.fitEval + 1;
    ff_par.memNumFitEval( ff_par.fitEval ) = ff_par.fitEval;
    if ff_par.fitEval == 1
        ff_par.bestFitEval = fit( i );
        ff_par.memFitEval( ff_par.fitEval ) = ff_par.bestFitEval;
    else
        if fit( i ) < ff_par.bestFitEval
            ff_par.bestFitEval = fit( i );
        end
        ff_par.memFitEval( ff_par.fitEval ) = ff_par.bestFitEval;
    end
end
end