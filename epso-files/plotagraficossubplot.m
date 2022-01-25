global vetorsensibilidade; 
vetorsensibilidade=-107;
figure;    
subplot(2,2,1)
yyaxis left
plot(1:size(vetormaxrunconsolidadow1,1),vetormaxrunconsolidadow1(:,3:size(vetormaxrunconsolidadow1,2)),'ok')
ylim([1 24]);
xlim([0 36]);
vetorpercentual=percentual(vetormaxrunconsolidadow1);
yyaxis right
plot(1:size(vetormaxrunconsolidadow1,1),vetorpercentual,'p-r')
ylim([0 110])
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 1')


subplot(2,2,2)
yyaxis left
plot(1:size(vetormaxrunconsolidadow2,1),vetormaxrunconsolidadow2(:,3:size(vetormaxrunconsolidadow2,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow2);
plot(1:size(vetormaxrunconsolidadow2,1),vetorpercentual,'p-r')
ylim([0 110])
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 2')

subplot(2,2,3)
yyaxis left
plot(1:size(vetormaxrunconsolidadow3,1),vetormaxrunconsolidadow3(:,3:size(vetormaxrunconsolidadow3,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow3);
plot(1:size(vetormaxrunconsolidadow3,1),vetorpercentual,'p-r')
ylim([0 110])
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 3')

subplot(2,2,4)
yyaxis left
plot(1:size(vetormaxrunconsolidadow4,1),vetormaxrunconsolidadow4(:,3:size(vetormaxrunconsolidadow4,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow4);
plot(1:size(vetormaxrunconsolidadow4,1),vetorpercentual,'p-r')
ylim([0 110]) 
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 4')

figure;    
subplot(2,2,1)
yyaxis left
plot(1:size(vetormaxrunconsolidadow5,1),vetormaxrunconsolidadow5(:,3:size(vetormaxrunconsolidadow5,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow5);
plot(1:size(vetormaxrunconsolidadow5,1),vetorpercentual,'p-r')
ylim([0 110]) 

grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 5')

subplot(2,2,2)
yyaxis left
plot(1:size(vetormaxrunconsolidadow6,1),vetormaxrunconsolidadow6(:,3:size(vetormaxrunconsolidadow6,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow6);
plot(1:size(vetormaxrunconsolidadow6,1),vetorpercentual,'p-r')
ylim([0 110]) 

grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 6')

subplot(2,2,3)
yyaxis left
plot(1:size(vetormaxrunconsolidadow7,1),vetormaxrunconsolidadow7(:,3:size(vetormaxrunconsolidadow7,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow7);
plot(1:size(vetormaxrunconsolidadow7,1),vetorpercentual,'p-r')
ylim([0 110]) 
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 7')

subplot(2,2,4)
yyaxis left
plot(1:size(vetormaxrunconsolidadow8,1),vetormaxrunconsolidadow8(:,3:size(vetormaxrunconsolidadow8,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow8);
plot(1:size(vetormaxrunconsolidadow8,1),vetorpercentual,'p-r')
ylim([0 110]) 
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 8')

figure;
subplot(2,1,1);
yyaxis left
plot(1:size(vetormaxrunconsolidadow9,1),vetormaxrunconsolidadow9(:,3:size(vetormaxrunconsolidadow9,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow9);
plot(1:size(vetormaxrunconsolidadow9,1),vetorpercentual,'p-r')
ylim([0 110]) 
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 9')


subplot(2,1,2); 
yyaxis left
plot(1:size(vetormaxrunconsolidadow10,1),vetormaxrunconsolidadow10(:,3:size(vetormaxrunconsolidadow10,2)),'ok')
ylim([1 24]);
xlim([0 36]);
yyaxis right
vetorpercentual=percentual(vetormaxrunconsolidadow10);
plot(1:size(vetormaxrunconsolidadow10,1),vetorpercentual,'p-r')
ylim([0 110]) 
grid on
grid minor
title('Simulações Monte Carlo para Frente de Pareto 10')