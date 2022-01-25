%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Leonel Carvalho, PhD (email: leonel.m.carvalho@inescporto.pt)
% 6th February 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ gbestval, gbest ] = EPSO( popSize, mutationRate, communicationProbability, maxGen, ...
    maxFitEval, maxGenWoChangeBest, printConvergenceResults, printConvergenceChart )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETTING PARAMETERS
% Global variable
global epso_par;
global ff_par X Y vetorobj1 vetorobj2;
% Maximum number of generations
epso_par.maxGen = maxGen;
% Maximum number of fitness evaluations
epso_par.maxFitEval = maxFitEval;
% Maximum number of generations without changing global best
epso_par.maxGenWoChangeBest = maxGenWoChangeBest;
% Print convergence results every n generations
epso_par.printConvergenceResults = printConvergenceResults ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE strategic parameters of EPSO
epso_par.popSize = popSize;
epso_par.mutationRate = mutationRate;
epso_par.communicationProbability = communicationProbability;
% Weights matrix
% 1 - inertia
% 2 - memory
% 3 - cooperation
% 4 - perturbation
weights = rand( epso_par.popSize, 4 );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RANDOMLY INITIALIZE CURRENT population
% Particles' lower bounds
Xmin = ff_par.Xmin; %repmat( ff_par.Xmin, 1, ff_par.D );
% Particles' upper bounds
Xmax = ff_par.Xmax; %repmat( ff_par.Xmax, 1, ff_par.D );
Vmin = -Xmax + Xmin;
Vmax = -Vmin;
pos = zeros( epso_par.popSize, ff_par.D );
vel = zeros( epso_par.popSize, ff_par.D );

for i = 1 : epso_par.popSize
    varrandAP=randperm(ff_par.D-1);
    varrandAP=varrandAP(1:1);
    varrand=randperm(length(Y));
    varrand=varrand(1:ff_par.D-1);
    pos( i, : ) = [varrandAP varrand];%Xmin + ( Xmax - Xmin ) .* rand( 1, ff_par.D );
    %   pos(i,:)=round(pos(i,:));
    vel( i, : ) = Vmin + ( Vmax - Vmin ) .* rand( 1, ff_par.D );
    % vel(i,:)=round(vel(i,:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATE the CURRENT population
[ fit ] = FITNESS_FUNCTION( epso_par.popSize, pos, X, Y );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UPDATE GLOBAL BEST
[ gbestval, gbestid ] = min( fit );
gbest = pos( gbestid, : );
memGbestval = zeros( 1, ( epso_par.maxGen + 1 ) );
memGbestval( 1 ) = gbestval;

figure(2)
plot(vetorobj2(gbestid)*ff_par.Xmax(1),vetorobj1(gbestid)*100,'ob');
hold on
grid on;
xlabel('Total of Gateways');
ylabel('Uncoverage Area (%)');
title('Pareto Front');
xlim = [1 10];
%fitnessmedio( = mean( fit);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UPDATE INDIVIDUAL BEST
% Individual best position ever of the particles of CURRENT population
myBestPos = pos;
% Fitness of the individual best position ever of the particles of CURRENT population
myBestPosFit = fit;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE generation counter
countGen = 0;
countGenWoChangeBest = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOP until termination criteria isn`t met
while countGen < epso_par.maxGen && countGenWoChangeBest <= epso_par.maxGenWoChangeBest && ff_par.fitEval <= epso_par.maxFitEval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE generation counter
    countGen = countGen + 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % COPY CURRENT population
    copyPos = pos;
    copyVel = vel;
    copyWeights = weights;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % APPLY EPSO movement rule
    for i = 1 : epso_par.popSize
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MUTATE WEIGHTS of the particles of the COPIED population
        copyWeights( i, : ) = MUTATE_WEIGHTS( weights( i, : ), epso_par.mutationRate );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW VELOCITY for the particles of the COPIED population
        copyVel( i, : ) = COMPUTE_NEW_VEL( ff_par.D, copyPos( i, : ), myBestPos( i, : ), gbest, ...
            copyVel( i, : ), Vmin, Vmax, copyWeights( i, : ), epso_par.communicationProbability );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the COPIED population
        [ copyPos( i, : ), copyVel( i, : ) ] = COMPUTE_NEW_POS( copyPos( i, : ), copyVel( i, : ) );
        copyPos(i,:)=round(copyPos(i,:));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW VELOCITY for the particles of the CURRENT population
        vel( i, : ) = COMPUTE_NEW_VEL( ff_par.D, pos( i, : ), myBestPos( i, : ), gbest, ...
            vel( i, : ), Vmin, Vmax, weights( i, : ), epso_par.communicationProbability );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % COMPUTE NEW POSITION for the particles of the CURRENT population
        [ pos( i, : ), vel( i, : ) ] = COMPUTE_NEW_POS( pos( i, : ), vel( i, : ) );
        pos(i,:)=round(pos(i,:));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ENFORCE search space limits of the COPIED population
    [ copyPos, copyVel ] = ENFORCE_POS_LIMITS( ff_par.D, epso_par.popSize, copyPos, Xmin, Xmax, copyVel, Vmin, Vmax );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for kl=1:epso_par.popSize
        cont=0;
        temigual=0;
        for ki=1:length(Y)
            for kj=1:(length(copyPos(kl,:))-1)
                if copyPos(kl,kj+1)==ki
                    cont=cont+1;
                end
                if cont>=2
                    temigual=1;
                end
            end
            cont=0;
        end
        tp=1;
        while temigual~=0
            t=1;
            valorteste=copyPos(kl,tp+1);
            while (t<=(length(copyPos(kl,:))-2))
                valor2=copyPos(kl,t+2);
                if t+2~=tp+1
                    if valorteste==valor2
                        varcopy=randperm(length(Y));
                        varcopy=varcopy(1:1);
                        copyPos(kl,t+2)=varcopy;    %copyPos(i,t+2)+copyVel(i,t+2);
                    end
                end
                t=t+1;
            end
            temigual=0;
            cont=0;
            for ki=1:length(Y)
                for kj=1:(length(copyPos(kl,:))-1)
                    if copyPos(kl,kj+1)==ki
                        cont=cont+1;
                    end
                    if cont>=2
                        temigual=1;
                    end
                end
                cont=0;
            end
            tp=tp+1;
            if tp==(length(copyPos(kl,:)))
                tp=1;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ENFORCE search space limits of the CURRENT population
    [ pos, vel ] = ENFORCE_POS_LIMITS( ff_par.D, epso_par.popSize, pos, Xmin, Xmax, vel, Vmin, Vmax );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for kl=1:epso_par.popSize
        cont=0;
        temigual=0;
        for ki=1:length(Y)
            for kj=1:(length(pos(kl,:))-1)
                if pos(kl,kj+1)==ki
                    cont=cont+1;
                end
                if cont>=2
                    temigual=1;
                end
            end
            cont=0;
        end
        tp=1;
        while temigual~=0
            t=1;
            valorteste=pos(kl,tp+1);
            while (t<=(length(pos(kl,:))-2))
                valor2=pos(kl,t+2);
                if t+2~=tp+1
                    if valorteste==valor2
                        varcopy=randperm(length(Y));
                        varcopy=varcopy(1:1);
                        pos(kl,t+2)=varcopy;    %copyPos(i,t+2)+copyVel(i,t+2);
                    end
                end
                t=t+1;
            end
            temigual=0;
            cont=0;
            for ki=1:length(Y)
                for kj=1:(length(pos(kl,:))-1)
                    if pos(kl,kj+1)==ki
                        cont=cont+1;
                    end
                    if cont>=2
                        temigual=1;
                    end
                end
                cont=0;
            end
            tp=tp+1;
            if tp==(length(pos(kl,:)))
                tp=1;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EVALUATE the COPIED population
    [ copyFit ] = FITNESS_FUNCTION( epso_par.popSize, copyPos,  X, Y );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EVALUATE the CURRENT population
    [ fit ] = FITNESS_FUNCTION( epso_par.popSize, pos,  X, Y );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CREATE NEW population to replace CURRENT population
    selParNewSwarm = ( copyFit < fit );
    for i = 1 : epso_par.popSize
        if selParNewSwarm( i )
            fit( i ) = copyFit( i );
            pos( i, : ) = copyPos( i, : );
            vel( i, : ) = copyVel( i, : );
            weights( i, : ) = copyWeights( i, : );
        end
        if fit( i ) < myBestPosFit( i )
            myBestPos( i, : ) = pos( i, : );
            myBestPosFit( i ) = fit( i );
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE GLOBAL BEST
    [ tmpgbestval, gbestid ] = min( fit );
    if tmpgbestval < gbestval
        gbestval = tmpgbestval;
        gbest = pos( gbestid, : );
        countGenWoChangeBest = 0;
    else
        countGenWoChangeBest = countGenWoChangeBest + 1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(2)
    plot(vetorobj2(gbestid)*ff_par.Xmax(1),vetorobj1(gbestid)*100,'ob');
    hold on
    grid on;
    xlabel('Total of Gateways');
    ylabel('Uncoverage Area (%)');
    title('Pareto Front');
    xlim = [1 10];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SAVE fitness
    memGbestval( countGen + 1 ) = gbestval;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PRINT results
    if rem( countGen, epso_par.printConvergenceResults  ) == 0 || countGen == 1
        fprintf('Gen: %-8d Best Fit: %.6e\n', countGen, gbestval );
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

if rem( countGen, epso_par.printConvergenceResults ) ~= 0
    fprintf('Gen: %-8d Best Fit: %.6e\n', countGen, gbestval );
end
fprintf('\n');

if printConvergenceChart == 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PRINT final results
    x = 0 : 1 : countGen;
    memGbestval = memGbestval( 1 : countGen + 1 );
    figure(1);
    plot( x, memGbestval );
    xlabel('generation');
    ylabel('fitness');
    grid on;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
end