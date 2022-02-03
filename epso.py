import os
import math
import numpy as np
from recordclass import recordclass

# -----------------------
# FUNCTIONS

def UFPA(x, ht, hr, f, d):
    lamb = 3e8 / f
    fM = f * 1e-6
    return x[0] * math.log10(d) + x[1] * math.log10(fM) + x[2] - x[3] * ((ht + hr)*lamb) / (0.1*62)
 

def minimum_fitness(list_fit):
    return list_fit[np.argmin(list_fit)], np.argmin(list_fit)


def compute_new_position(position, velocity):
    new_position = position + velocity
    new_velocity = velocity
    return new_position, new_velocity


def compute_new_vel(D, pos, myBestPos, gbest, vel, Vmin, Vmax, weights, communicationProbability):
    # % Computes new velocity according to the movement rule
    # % Compute inertial term
    inertiaTerm = weights[0] * vel
    # % Compute memory term
    memoryTerm = weights[1] * ( myBestPos - pos )
    # % Compute cooperation term
    # % Sample normally distributed number to perturbate the best position
    # %normrnd : Random arrays from the normal distribution.
    cooperationTerm = weights[2] * ( gbest * ( 1 + weights[3] * np.random.normal( 0, 1 ) ) - pos )
    # %rand returns an N-by-N matrix containing pseudorandom values drawn
    # % from the standard uniform distribution on the open interval(0,1).
    communicationProbabilityMatrix = np.random.rand( 1, D ) < communicationProbability
    cooperationTerm = cooperationTerm * communicationProbabilityMatrix
    # % Compute velocity
    new_vel = inertiaTerm + memoryTerm + cooperationTerm
    # % Check velocity limits
    new_vel = ( new_vel > Vmax ) * Vmax + ( new_vel <= Vmax ) * new_vel
    new_vel = ( new_vel < Vmin ) * Vmin + ( new_vel >= Vmin ) * new_vel

    return new_vel


# 0 |     1    |  2  |   3   |   4   |  5  |   6   |   7
# D | pop_size | pos | x_min | x_max | vel | v_min | v_max
def enforce_position_limit(D, pop_size, pos, x_min, x_max, vel, v_min, v_max):
    new_position = pos
    new_velocity = vel

    for i in range(pop_size):
        for j in range(D):

            # print('newPos: {} || xMin: {}'.format(new_position[i][j], x_min[0][j]))

            if new_position[i][j] < x_min[0][j]:
                new_position[i][j] = x_min[0][j]
                if new_velocity[i][j] < 0:
                    new_velocity[i][j] = - new_velocity[i][j]
            
            elif new_position[i][j] > x_max[0][j]:
                new_position[i][j] = x_max[0][j]
                if new_velocity[i][j] > 0:
                    new_velocity[i][j] = - new_velocity[i][j]

            # check velocity in case of asymmetric velocity limits
            if new_velocity[i][j] < v_min[0][j]:
                new_velocity[i][j] = v_min[0][j]
            
            elif new_velocity[i][j] > v_max[0][j]:
                new_velocity[i][j] = v_max[0][j]

    return new_position, new_velocity


def mutate_weights(weights, mutationRate):
    # % Mutate weights & check weights limits
    mutated_Weights = weights
    for i in range(len(weights)):
        mutated_Weights[i] = weights[i] + np.random.normal( 0, 1 ) * mutationRate
        if  mutated_Weights[i] > 1:
            mutated_Weights[i] = 1
        elif  mutated_Weights[i] < 0:
            mutated_Weights[i] = 0

    return mutated_Weights


def fitness_function(popSize, pos, X, Y):
    
    # definição de variáveis -> não declarada no matlab
    ht = []
    Posicaoreallong = []
    Posicaoreallat = []
    distancia = []
    pr2 = []
    did = []
    indicedecobertura = []
    intercobertura = np.zeros([len(X), 3])

    # %ESCOLHE QUAL VETOR VAI USAR OU COM ALTURAS FIXAS EM 15 metros ou com altura variaveis
    # % htt=[30,6,30,15,9,6,9,6,12,6,12,12,30,15,6,6,6,9,6,6,6,6]
    htt=[15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
    fit = np.zeros(popSize)
    
    for i in range(popSize):
        # % SIMA function
        Naps = int(pos[i,0])
        Pt=20

        if(Naps == 0):
            Naps = 1

        for j in range(Naps):
    
            PosicaoAps = int(pos[i,j+1])
            Posicaoreallong.append(Y[PosicaoAps][1])
            Posicaoreallat.append(Y[PosicaoAps][0])
            ht.append(htt[PosicaoAps])

        for kk in range(len(X)):
            intercobertura[kk,0] = X[kk,1] * math.pi / 180 #%lat1
            intercobertura[kk,1] = X[kk,0] * math.pi / 180 #%lat1

        f=915e6 #%freq em Hz
        contadordecobertura = 0
        m=1
        for p in range(Naps):
            contadordecobertura=0
            for k in range(len(X)):
                radius = 6371
                lat1 = X[k][0] * math.pi / 180
                lat2 = Posicaoreallat[p] * math.pi / 180
                lon1 = X[k][1] * math.pi / 180
                lon2 = Posicaoreallong[p] * math.pi / 180
                deltaLat = lat2-lat1
                deltaLon = lon2-lon1
                a = math.sin((deltaLat/2))**2 + math.cos(lat1) * math.cos(lat2) * math.sin((deltaLon/2))**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distancia.append(radius*c)    #%Haversine distance
                
                # %parametros do modelo
                x = [28.405, 9.2, 43.698, 23.754]
                
                hb = ht
                hr = 3
                hm = hr
                if distancia[k] > 0:
                    pr = Pt - (UFPA(x, ht[p], hr, f, (distancia[k] * 1000)))
                    # print('value pr = {} || value limiar = {}'.format(pr, input_user_limiar))
                    if pr >= input_user_limiar:
                        pr2.append(pr)
                        did.append(distancia[k])
                        intercobertura[k,2] = 1    
                        contadordecobertura = contadordecobertura+1

            indicedecobertura.append((contadordecobertura/(len(X))) * 100)
        
        tamanhodacobertura = len(intercobertura)       
        percentualcalculado = (sum(intercobertura[:,2]) / tamanhodacobertura) * 100
        Perc = 100

        # %normalizando
        obj1 = percentualcalculado
        obj1 = Perc-percentualcalculado
        obj1 = obj1/100
        
        # %normalizando
        obj2 = Naps
        obj2 = obj2 / ff_par.x_max[0][0]
        fit[i] = W * obj1 + (1-W) * obj2
        
        vetor_obj_1.append(obj1)
        vetor_obj_2.append(obj2)
    
        # ainda está no loop de popSize no arquivo original, matlab
        ff_par.fit_eval = ff_par.fit_eval + 1
        ff_par.mem_num_fit_eval[0][ff_par.fit_eval] = ff_par.fit_eval

        if ff_par.fit_eval == 1:
            ff_par.best_fit_eval = fit[i]
            ff_par.mem_fit_eval[0][ff_par.fit_eval] = ff_par.best_fit_eval
        else:
            if fit[i] < ff_par.best_fit_eval:
                ff_par.best_fit_eval = fit[i]
        
            ff_par.mem_fit_eval[0][ff_par.fit_eval] = ff_par.best_fit_eval
    
    return fit


def EPSO(pop_size, mutation_rate, communication_probability, max_gen, max_fit_eval, \
max_gen_wo_change_best, print_convergence_results, print_convergence_chart):

    fit = []

    weights = np.random.random((pop_size, 4))

    # % RANDOMLY INITIALIZE CURRENT population
    # % Particles' lower bounds
    x_min = ff_par.x_min #%repmat( ff_par.x_min, 1, ff_par.D )
    # % Particles' upper bounds
    x_max = ff_par.x_max #%repmat( ff_par.x_max, 1, ff_par.D )
    v_min = -x_max + x_min
    v_max = -v_min
    pos = np.zeros([pop_size, ff_par.D])
    vel = np.zeros([pop_size, ff_par.D])

    limit = len(Y)-1
    for i in range(pop_size):
        varrandAP = np.random.permutation(ff_par.D-1)
        varrandAP = varrandAP[0:1] #seleciona um único valor.. neste caso o primeiro elemento
        varrand = np.random.permutation(limit)
        varrand = varrand[0:ff_par.D-1]
        pos[i] = np.concatenate((varrandAP, varrand)) 
        vel[i] = v_min + ( v_max - v_min ) * np.random.random((1, ff_par.D))
    
    # % EVALUATE the CURRENT population
    fit = fitness_function( pop_size, pos, X, Y )

    # % UPDATE GLOBAL BEST
    gbestval, gbestid = minimum_fitness(fit)
    gbest = pos[gbestid][:]
    memGbestval = np.zeros([1,max_gen+1])
    memGbestval[0][0] = gbestval

    # figure(2)
    # plot(vetorobj2(gbestid)*ff_par.Xmax(1),vetorobj1(gbestid)*100,'ob')
    # hold on
    # grid on
    # xlabel('Total of Gateways')
    # ylabel('Uncoverage Area (%)')
    # title('Pareto Front')
    # xlim = [1 10]

    # % UPDATE INDIVIDUAL BEST
    # % Individual best position ever of the particles of CURRENT population
    myBestPos = pos
    # % Fitness of the individual best position ever of the particles of CURRENT population
    myBestPosFit = fit
    

    # % INITIALIZE generation counter
    countGen = 0
    countGenWoChangeBest = 0

    while countGen < max_gen and countGenWoChangeBest <= max_gen_wo_change_best and ff_par.fit_eval <= max_fit_eval:
    
        # % UPDATE generation counter
        countGen = countGen + 1
        # % COPY CURRENT population
        copyPos = pos
        copyVel = vel
        copyWeights = weights
        
        # % APPLY EPSO movement rule
        for i in range(pop_size):
            # % MUTATE WEIGHTS of the particles of the COPIED population
            copyWeights[i] = mutate_weights( weights[i], mutation_rate )
            
            # % COMPUTE NEW VELOCITY for the particles of the COPIED population
            copyVel[i] = compute_new_vel( ff_par.D, copyPos[i], myBestPos[i], gbest, copyVel[i], v_min, v_max, copyWeights[i], communication_probability )
            
            # % COMPUTE NEW POSITION for the particles of the COPIED population
            copyPos[i], copyVel[i] = compute_new_position( copyPos[i], copyVel[i] )
            copyPos[i] = np.around(copyPos[i])
            
            # % COMPUTE NEW VELOCITY for the particles of the CURRENT population
            vel[i] = compute_new_vel( ff_par.D, pos[i], myBestPos[i], gbest, vel[i], v_min, v_max, weights[i], communication_probability )
            
            # % COMPUTE NEW POSITION for the particles of the CURRENT population
            [ pos[i], vel[i] ] = compute_new_position( pos[i], vel[i] )
            pos[i] = np.around(pos[i])
        

        # % ENFORCE search space limits of the COPIED population
        copyPos, copyVel = enforce_position_limit( ff_par.D, pop_size, copyPos, x_min, x_max, copyVel, v_min, v_max )

        for kl in range(pop_size):
            cont = 0  
            temigual = 0
            for ki in range(len(Y)):
                for kj in range(len(copyPos[kl][:])-1):
                    if copyPos[kl][kj+1] == ki:
                        cont = cont+1
                    
                    if cont>=2:
                        temigual = 1
                
                cont = 0
            
            tp = 0

            # not equal ~= in python !=
            while temigual != 0:
                t = 0
                valorteste = copyPos[kl][tp+1]
                while (t<=(len(copyPos[kl][:])-3)):
                    valor2=copyPos[kl][t+2]
                    # not equal ~= in python !=
                    if t+2 != tp+1:
                        if valorteste == valor2:
                            varcopy = np.random.permutation(len(Y))
                            varcopy = varcopy[0]
                            copyPos[kl][t+2] = varcopy    #%copyPos(i,t+2)+copyVel(i,t+2)
                    
                    t = t+1
                
                temigual = 0
                cont = 0
                    
                for ki in range(len(Y)):
                    for kj in range((len(copyPos[kl][:])-1)):
                        if copyPos[kl][kj+1] == ki:
                            cont = cont+1
                        
                        if cont>=2:
                            temigual = 1
                    cont = 0
                
                tp = tp+1
                if tp == len(copyPos[kl][:])-1:
                    tp = 0
            
            
        # % EVALUATE the COPIED population
        ff_par.fit_eval = 0
        ff_par.best_fit_eval = 0
        copyFit = fitness_function(pop_size, copyPos, X, Y)
        # % EVALUATE the CURRENT population
        ff_par.fit_eval = 0
        ff_par.best_fit_eval = 0
        fit = fitness_function(pop_size, pos, X, Y)

        # % CREATE NEW population to replace CURRENT population
        selParNewSwarm = ( copyFit < fit )
        for i in range(pop_size):
            if selParNewSwarm[i]:
                fit[i] = copyFit[i]
                pos[i] = copyPos[i]
                vel[i] = copyVel[i]
                weights[i] = copyWeights[i]
            if fit[i] < myBestPosFit[i]:
                myBestPos[i] = pos[i]
                myBestPosFit[i] = fit[i]

        # % UPDATE GLOBAL BEST
        tmpgbestval, gbestid = minimum_fitness(fit)
        if tmpgbestval < gbestval:
            gbestval = tmpgbestval
            gbest = pos[gbestid, :]
            countGenWoChangeBest = 0
        else:
            countGenWoChangeBest = countGenWoChangeBest + 1
        
        # figure(2)
        # plot(vetorobj2(gbestid)*ff_par.Xmax(1),vetorobj1(gbestid)*100,'ob')
        # hold on
        # grid on
        # xlabel('Total of Gateways')
        # ylabel('Uncoverage Area (%)')
        # title('Pareto Front')
        # xlim = [1 10]    
        # % SAVE fitness
        
        memGbestval[0][countGen] = gbestval

        # print results
        if math.remainder(countGen, print_convergence_results) == 0 or countGen == 1:
            print('Gen: {} | BestFit: {}'.format(countGen, gbestval))
        
    
    if math.remainder(countGen, print_convergence_results ) != 0:
        print('Gen: {} | Best Fit: {}'.format(countGen, gbestval))

    # plota um gráfico
    # if print_convergence_chart == 1:
    #     # % PRINT final results
    #     x = 0 : 1 : countGen;
    #     memGbestval = memGbestval( 1 : countGen + 1 );
    #     figure(1);
    #     plot( x, memGbestval );
    #     xlabel('generation');
    #     ylabel('fitness');
    #     grid on;

    return gbestval, gbest



# -----------------------
# RUNNING

PATH_FILE = os.getcwd() + '/route-file.txt'

# obj_epso = recordclass("objeto", "max_gen max_fit_eval max_gen_wo_change_best print_convergence_results \
# print_convergence_chart pop_size mutation_rate communication_probability")
# epso_par = obj_epso(max_gen, max_fit_eval, max_gen_wo_change_best, print_convergency_results, print_convergency_chart, \
# pop_size,mutation_rate,communication_probability)

LENGTH_D = 11
# cria referencia de objeto para acesso dentro das funções do programa
obj_ff = recordclass("fitness_function", "D x_min x_max fit_eval best_fit_eval mem_num_fit_eval, mem_fit_eval")
ff_par = obj_ff(LENGTH_D, np.array([0,0,0,0,0,0,0,0,0,0,0]).reshape([1,LENGTH_D]), np.array([10,21,21,21,21,21,21,21,21,21,21]).reshape([1,LENGTH_D]), \
0, 0, [], [])


"""
variáveis de entrada para simulação
---------------------------------------
tamanho da população, taxa de mutação, probabilidade de comunicação
número máximo de avaliações de aptidão, número máximo de gerações, número máximo de gerações com o mesmo melhor global
número máximo de execuções, flag para impressão da convergência de resultados, flag para renderizar os gráficos: 1 - chart || 0 - no chart
memória para o melhor aptidão de cada execução, memória para melhor aptidão de cada execução
"""
pop_size, mutation_rate, communication_probability = 10, 0.4, 0.7
max_fit_eval, max_gen, max_gen_wo_change_best = 20, 5, 5
max_run, print_convergency_chart, print_convergency_results = 3, 1, 1
mem_best_fitness, mem_best_solutions = np.zeros([1, max_run]), np.zeros([max_run, ff_par.D])

X,Y,W, epso_par, vetor_obj_1, vetor_obj_2, input_user_limiar = [], [], [], [], [], [], -104

# rota do circular
X = np.genfromtxt(PATH_FILE, delimiter=', ')

# possíveis pontos de solução
Y = np.array([[-1.476941, -48.456547], [-1.472760, -48.450713], [-1.475772, -48.455196],
[-1.473413, -48.455485], [-1.471206, -48.449751], [-1.474910, -48.458282],
[-1.475333, -48.456951], [-1.475561, -48.453613], [-1.474568, -48.453721],
[-1.475642, -48.453625], [-1.473113, -48.451395], [-1.463583333, -48.4453],
[-1.464191667, -48.44536111], [-1.475561, -48.453874], [-1.466669444, -48.44614444],
[-1.468322222, -48.44794167], [-1.468741667, -48.44791944], [-1.469397222, -48.446675],
[-1.470936111, -48.44728611], [-1.471119444, -48.44614722], [-1.470361111, -48.44504167],
[-1.470306, -48.443924]])


# frente de pareto
NP = 10
dimensao, Npp, g, jhh = ff_par.D, NP, 0, 0

for i in range(max_run):

    # inicialização não existe no pseudo código
    vetor_w, vetor_best= [], np.zeros([NP, ff_par.D])
    vetor_max_run_consolidado_w1, vetor_max_run_consolidado_w2, vetor_max_run_consolidado_w3 = [], [], []
    vetor_max_run_consolidado_w4, vetor_max_run_consolidado_w5, vetor_max_run_consolidado_w6 = [], [], []
    vetor_max_run_consolidado_w7, vetor_max_run_consolidado_w8, vetor_max_run_consolidado_w9 = [], [], []
    vetor_max_run_consolidado_w10 = []

    # EPSO RUN
    for ii in range(NP):
        
        W = ii/NP
        ff_par.fit_eval = 0
        ff_par.best_fit_eval = 0
        ff_par.mem_num_fit_eval = np.zeros([1,max_fit_eval]) #memória para número de fitness por avaliação
        ff_par.mem_fit_eval = np.zeros([1, max_fit_eval]) #memória para o fitness por avaliação

        # EPSO
        print('****** Run {} ******'.format(i))
        g_best_fit, g_best = EPSO(pop_size, mutation_rate, communication_probability, max_gen, max_fit_eval, 
        max_gen_wo_change_best, print_convergency_results, print_convergency_chart)

        mem_best_fitness[i] = g_best_fit
        mem_best_solutions[i,:] = g_best

        # vetor_w[ii] = W
        vetor_w.append(W)
        vetor_best[ii][:] = g_best


    #no arquivo original -> [vetorw' vetorbest] 
    # vetorw' -> é a transposta do complexo conjugado => https://www.mathworks.com/help/matlab/ref/ctranspose.html
    vetor_w = np.reshape(vetor_w, [1,len(vetor_w)])
    vetor_max_run = np.concatenate([np.conj(vetor_w.T), vetor_best], axis=1)


    for jh in range(len(vetor_max_run)): # --- 1 #linha
        for jl in range(len(vetor_max_run)): # --- 2 #coluna

            hjk = vetor_max_run(jh, 2)

            # não existe switch no python, tratamento com if
            if vetor_max_run(jh,1) == vetor_w(1):
                hjk = vetor_max_run(jh,2)
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w1[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(2):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w2[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(3):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w3[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(4):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w4[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(5):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w5[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(6):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w6[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(7):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w7[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(8):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w8[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(9):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w9[jhh+g, jl] = vetor_max_run(jh,jl)
            
            if vetor_max_run(jh,1) == vetor_w(10):
                if jl >= 1 and jl <= hjk+2:
                    vetor_max_run_consolidado_w10[jhh+g, jl] = vetor_max_run(jh,jl)
            # não existe switch no python, tratamento com if
        jhh=jhh+1
    

    # start print results final
    if max_run == 1:
        
        print('Best Solution')
        for id in range(ff_par.D):
            print('x[{}]: {}'.format(id, g_best[id]))

        print('Quantidade de APs: {}'.format(g_best[0]))
        for jj in range(g_best[1]):
            print('Localização {}: {} {}'.format(jj, Y[g_best[jj+1, 1], Y[g_best[jj+1, 2]]]))

        print('Best Fitness: {}'.format(g_best_fit))
        print('Fitness Evaluations: {}'.format(ff_par.fit_eval))

        # plot -> print convergence chart
        ff_par.mem_num_fit_eval = ff_par.mem_num_fit_eval[1:ff_par.fit_eval]
        ff_par.mem_fit_eval = ff_par.mem_fit_eval[1:ff_par.fit_eval]

        # plot( ff_par.memNumFitEval, ff_par.memFitEval, '-.k' );
        # xlabel('evaluation');
        # ylabel('fitness');
        # grid on;
    
    else:
        print('Averange best fitness: {}'.format(np.mean(mem_best_fitness)))
        print('Standart deviation best fitness: {}'.format(np.std(mem_best_fitness)))

    # end print results final

    for io in range(len(vetor_w)):
        print('Vetor_W: {} Vetorbest: {}'.format(vetor_w[io], vetor_best[io,:]))

    # figure;
    # plot(1:i,vetormaxrunconsolidadow1(:,3:size(vetormaxrunconsolidadow1,2)),'or')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow2(:,3:size(vetormaxrunconsolidadow2,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow3(:,3:size(vetormaxrunconsolidadow3,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow4(:,3:size(vetormaxrunconsolidadow4,2)),'ok')
    # ylim([1 23]);    
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow5(:,3:size(vetormaxrunconsolidadow5,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow6(:,3:size(vetormaxrunconsolidadow6,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow7(:,3:size(vetormaxrunconsolidadow7,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow8(:,3:size(vetormaxrunconsolidadow8,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow9(:,3:size(vetormaxrunconsolidadow9,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor

    # figure;
    # plot(1:i,vetormaxrunconsolidadow10(:,3:size(vetormaxrunconsolidadow10,2)),'ok')
    # ylim([1 23]);
    # grid on
    # grid minor
    
    # figure;    
    # subplot(2,2,1)
    # yyaxis left    plot(1:size(vetormaxrunconsolidadow1,1),vetormaxrunconsolidadow1(:,3:size(vetormaxrunconsolidadow1,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow1,1)+1]);
    # xlabel('Simulation number');
    # vetorpercentual=percentual(vetormaxrunconsolidadow1);
    # yyaxis right
    # plot(1:size(vetormaxrunconsolidadow1,1),vetorpercentual,'p-r')
    # ylim([0 110])
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 1')
    
    
    # subplot(2,2,2)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow2,1),vetormaxrunconsolidadow2(:,3:size(vetormaxrunconsolidadow2,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow2,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow2);
    # plot(1:size(vetormaxrunconsolidadow2,1),vetorpercentual,'p-r')
    # ylim([0 110])
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 2')
    
    # subplot(2,2,3)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow3,1),vetormaxrunconsolidadow3(:,3:size(vetormaxrunconsolidadow3,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow3,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow3);
    # plot(1:size(vetormaxrunconsolidadow3,1),vetorpercentual,'p-r')
    # ylim([0 110])
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 3')
    
    # subplot(2,2,4)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow4,1),vetormaxrunconsolidadow4(:,3:size(vetormaxrunconsolidadow4,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow4,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow4);
    # plot(1:size(vetormaxrunconsolidadow4,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 4')
    
    # figure;    
    # subplot(2,2,1)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow5,1),vetormaxrunconsolidadow5(:,3:size(vetormaxrunconsolidadow5,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow5,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow5);
    # plot(1:size(vetormaxrunconsolidadow5,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 5')
    
    # subplot(2,2,2)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow6,1),vetormaxrunconsolidadow6(:,3:size(vetormaxrunconsolidadow6,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow6,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow6);
    # plot(1:size(vetormaxrunconsolidadow6,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 6')
    
    # subplot(2,2,3)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow7,1),vetormaxrunconsolidadow7(:,3:size(vetormaxrunconsolidadow7,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow7,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow7);
    # plot(1:size(vetormaxrunconsolidadow7,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 7')
    
    # subplot(2,2,4)
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow8,1),vetormaxrunconsolidadow8(:,3:size(vetormaxrunconsolidadow8,2)),'ok')
    # ylim([0 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow8,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow8);
    # plot(1:size(vetormaxrunconsolidadow8,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 8')
    
    # figure;
    # subplot(2,1,1);
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow9,1),vetormaxrunconsolidadow9(:,3:size(vetormaxrunconsolidadow9,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow9,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow9);
    # plot(1:size(vetormaxrunconsolidadow9,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 9')
    
    
    # subplot(2,1,2); 
    # yyaxis left
    # plot(1:size(vetormaxrunconsolidadow10,1),vetormaxrunconsolidadow10(:,3:size(vetormaxrunconsolidadow10,2)),'ok')
    # ylim([1 24]);
    # ylabel('Gateway Placement')
    # xlim([0 size(vetormaxrunconsolidadow10,1)+1]);
    # xlabel('Simulation number');
    # yyaxis right
    # vetorpercentual=percentual(vetormaxrunconsolidadow10);
    # plot(1:size(vetormaxrunconsolidadow10,1),vetorpercentual,'p-r')
    # ylim([0 110]) 
    # ylabel('Percentage of coverage area');
    # grid on
    # grid minor
    # title('Monte Carlo Simulation for Pareto Front 10')

