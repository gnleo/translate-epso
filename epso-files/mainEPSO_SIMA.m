%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Leonel Carvalho, PhD (email: leonel.m.carvalho@inescporto.pt)
% 6th February 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

global ff_par Y X W;
global vetorsensibilidade;


vetorsensibilidade=-104;
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
% Fitness function
% 1 - Sphere
% 2 - Rosenbrock
% 3 - Shaffer
% 4 - Schwefel
% 5 - Griewank
% 6 - Rastrigin
% 7 - Ackley
% 8- SIMA Planning
ff_par.ff = 8;
switch ff_par.ff
    case 1
        % DimensionalitY of optimization problem
        ff_par.D = 30;
        ff_par.Xmin = -10;
        ff_par.Xmax = 10;
    case 2
        ff_par.D = 30;
        ff_par.Xmin = -5;
        ff_par.Xmax = 10;
    case 3
        ff_par.D = 30;
        ff_par.Xmin = -100;
        ff_par.Xmax = 100;
    case 4
        ff_par.D = 30;
        ff_par.Xmin = -500;
        ff_par.Xmax = 500;
    case 5
        ff_par.D = 30;
        ff_par.Xmin = -600;
        ff_par.Xmax = 600;
    case 6
        ff_par.D = 30;
        ff_par.Xmin = -5.12;
        ff_par.Xmax = 5.12;
    case 7
        ff_par.D = 30;
        ff_par.Xmin = -15;
        ff_par.Xmax = 30;
    case 8
        ff_par.D = 11;
        ff_par.Xmin = [1 1 1 1 1 1 1 1 1 1 1];
        ff_par.Xmax = [10 22 22 22 22 22 22 22 22 22 22];

    otherwise
        % This is never supposed to happen
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE random number generator
% rng('default');
% seed = 1234;
% rng( seed, 'twister'  );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET STRATEGIC PARAMETERS
popSize = 10; % population size
mutationRate = 0.4; % mutation rate ; Domain: ] 0, inf [
communicationProbability = 0.7; % communication probability ; Domain: [ 0, 1 ]
maxFitEval = 20; % maximum number of fitness evaluations
maxGen = 3; % maximum number of generations
maxGenWoChangeBest = 2; % maximum number of generations with the same global best
% SET SIMULATION PARAMETERS
printConvergenceResults = 1;
printConvergenceChart = 1; % 1 -> Chart ; 0 -> No Chart;
maxRun = 5; % maximum number of EPSO runs
memBestFitness = zeros( 1, maxRun ); % memory for the best fitness of each run
memBestSolution = zeros( maxRun, ff_par.D );  % memory for best run of each run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRINT message
% fprintf('            EPSO 2014              \n');
% fprintf('       Leonel Carvalho, PhD        \n');
% fprintf('  leonel.m.carvalho@inescporto.pt  \n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRINT simulation parameters
fprintf('\nMax Gen: %d\n', maxGen);
fprintf('Max Fit Evaluations: %d\n', maxFitEval);
fprintf('Max Gen With Equal Global Best: %d\n', maxGenWoChangeBest);
fprintf('Population Size: %d\n', popSize);
fprintf('Mutation Rate: %.3f\n', mutationRate);
fprintf('Communication Probability: %.3f\n\n', communicationProbability);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NP = 10; % Frente de Pareto 10
dimensao=ff_par.D;
Npp=NP;
g=0;
jhh=1;
for i = 1 : maxRun

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % RUN EPSO
    for ii=1:NP

        W = ii/NP

        ff_par.fitEval = 0;
        ff_par.bestFitEval = 0;
        ff_par.memNumFitEval = zeros( 1, maxFitEval ); % memory for the fitness per evaluation
        ff_par.memFitEval = zeros( 1, maxFitEval ); % memory for the fitness per evaluatio
        fprintf('************* Run %d *************\n', i);
        [ gbestfit, gbest ] = EPSO( popSize, mutationRate, communicationProbability, maxGen, maxFitEval, ...
            maxGenWoChangeBest, printConvergenceResults, printConvergenceChart );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        memBestFitness( i ) = gbestfit
        memBestSolution( i, : ) = gbest

        vetorw(ii)=W;
        vetorbest(ii,:)=gbest;
    end
    vetormaxrun=[vetorw' vetorbest];
    for jh=1:size(vetormaxrun,1)
        for jl=1:size(vetormaxrun,2)
            hjk=vetormaxrun(jh,2);
            switch vetormaxrun(jh,1)
                case vetorw(1)
                    hjk=vetormaxrun(jh,2);
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow1(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(2)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow2(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(3)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow3(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(4)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow4(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(5)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow5(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(6)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow6(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(7)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow7(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(8)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow8(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(9)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow9(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
                case vetorw(10)
                    if jl>=1&jl<=hjk+2
                        vetormaxrunconsolidadow10(jhh+g,jl)=vetormaxrun(jh,jl);
                    end;
            end;

            %dimensao=dimensao+ff_par.D;

        end;

    end;
    jhh=jhh+1;
    %g=g+NP;
end;
if maxRun == 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PRINT final results
    fprintf('Best Solution:\n');
    for id = 1 : ff_par.D
        fprintf('x[%d]: %.4e\n', id, gbest( id ) );
    end
    fprintf('\nQuantidade de APs: %d\n', gbest(1));

    for jj=1:gbest(1)
        fprintf('\nLocalização %d: %.8e %.8e\n',jj, Y(gbest(jj+1),1), Y(gbest(jj+1),2) );
    end;
    fprintf('\nBest Fitness: %.8e\n', gbestfit );
    fprintf('Fitness Evaluations: %d\n', ff_par.fitEval );

    % PRINT convergence chart
    ff_par.memNumFitEval = ff_par.memNumFitEval( 1:ff_par.fitEval );
    ff_par.memFitEval = ff_par.memFitEval( 1:ff_par.fitEval );
    plot( ff_par.memNumFitEval, ff_par.memFitEval, '-.k' );
    xlabel('evaluation');
    ylabel('fitness');
    grid on;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PRINT final results
    fprintf('Average Best Fitness: %.8e\n', mean( memBestFitness ) );
    fprintf('Standard Deviation Best Fitness: %.8e\n', std( memBestFitness ) );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

for io=1:length(vetorw)
    fprintf('\nVetor W: %4.2f  Vetorbest  %.f %.f %.f %.f %.f %.f %.f %.f %.f %.f %.f \n',vetorw(io), vetorbest(io,:));
end;
figure;
plot(1:i,vetormaxrunconsolidadow1(:,3:size(vetormaxrunconsolidadow1,2)),'or')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow2(:,3:size(vetormaxrunconsolidadow2,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow3(:,3:size(vetormaxrunconsolidadow3,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow4(:,3:size(vetormaxrunconsolidadow4,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow5(:,3:size(vetormaxrunconsolidadow5,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow6(:,3:size(vetormaxrunconsolidadow6,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow7(:,3:size(vetormaxrunconsolidadow7,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow8(:,3:size(vetormaxrunconsolidadow8,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow9(:,3:size(vetormaxrunconsolidadow9,2)),'ok')
ylim([1 23]);
grid on
grid minor
figure;
plot(1:i,vetormaxrunconsolidadow10(:,3:size(vetormaxrunconsolidadow10,2)),'ok')
ylim([1 23]);
grid on
grid minor

figure;
subplot(2,2,1)
yyaxis left
plot(1:size(vetormaxrunconsolidadow1,1),vetormaxrunconsolidadow1(:,3:size(vetormaxrunconsolidadow1,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow1,1)+1]);
xlabel('Simulation number');
vetorpercentual=percentual(vetormaxrunconsolidadow1);
yyaxis right
plot(1:size(vetormaxrunconsolidadow1,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 1')


subplot(2,2,2)
yyaxis left
plot(1:size(vetormaxrunconsolidadow2,1),vetormaxrunconsolidadow2(:,3:size(vetormaxrunconsolidadow2,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow2,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow2);
plot(1:size(vetormaxrunconsolidadow2,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 2')

subplot(2,2,3)
yyaxis left
plot(1:size(vetormaxrunconsolidadow3,1),vetormaxrunconsolidadow3(:,3:size(vetormaxrunconsolidadow3,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow3,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow3);
plot(1:size(vetormaxrunconsolidadow3,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 3')

subplot(2,2,4)
yyaxis left
plot(1:size(vetormaxrunconsolidadow4,1),vetormaxrunconsolidadow4(:,3:size(vetormaxrunconsolidadow4,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow4,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow4);
plot(1:size(vetormaxrunconsolidadow4,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 4')

figure;
subplot(2,2,1)
yyaxis left
plot(1:size(vetormaxrunconsolidadow5,1),vetormaxrunconsolidadow5(:,3:size(vetormaxrunconsolidadow5,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow5,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow5);
plot(1:size(vetormaxrunconsolidadow5,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 5')

subplot(2,2,2)
yyaxis left
plot(1:size(vetormaxrunconsolidadow6,1),vetormaxrunconsolidadow6(:,3:size(vetormaxrunconsolidadow6,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow6,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow6);
plot(1:size(vetormaxrunconsolidadow6,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 6')

subplot(2,2,3)
yyaxis left
plot(1:size(vetormaxrunconsolidadow7,1),vetormaxrunconsolidadow7(:,3:size(vetormaxrunconsolidadow7,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow7,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow7);
plot(1:size(vetormaxrunconsolidadow7,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 7')

subplot(2,2,4)
yyaxis left
plot(1:size(vetormaxrunconsolidadow8,1),vetormaxrunconsolidadow8(:,3:size(vetormaxrunconsolidadow8,2)),'ok')
ylim([0 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow8,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow8);
plot(1:size(vetormaxrunconsolidadow8,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 8')

figure;
subplot(2,1,1);
yyaxis left
plot(1:size(vetormaxrunconsolidadow9,1),vetormaxrunconsolidadow9(:,3:size(vetormaxrunconsolidadow9,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow9,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow9);
plot(1:size(vetormaxrunconsolidadow9,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 9')


subplot(2,1,2);
yyaxis left
plot(1:size(vetormaxrunconsolidadow10,1),vetormaxrunconsolidadow10(:,3:size(vetormaxrunconsolidadow10,2)),'ok')
ylim([1 24]);
ylabel('Gateway Placement')
xlim([0 size(vetormaxrunconsolidadow10,1)+1]);
xlabel('Simulation number');
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow10);
plot(1:size(vetormaxrunconsolidadow10,1),vetorpercentual,'p-r')
ylim([0 110])
ylabel('Percentage of coverage area');
grid on
grid minor
title('Monte Carlo Simulation for Pareto Front 10')

