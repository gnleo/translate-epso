function [ mutated_Weights ] = MUTATE_WEIGHTS( weights, mutationRate )
% Mutate weights & check weights limits
mutated_Weights = weights;
for i = 1 : length( weights )
    mutated_Weights( i ) = weights( i ) + normrnd( 0, 1 ) * mutationRate;
    if  mutated_Weights( i ) > 1
        mutated_Weights( i ) = 1;
    elseif  mutated_Weights( i ) < 0
        mutated_Weights( i ) = 0;
    end
end
end