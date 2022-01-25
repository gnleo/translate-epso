function [ new_vel ] = COMPUTE_NEW_VEL( D, pos, myBestPos, gbest, vel, Vmin, Vmax, weights, communicationProbability )
% Computes new velocity according to the movement rule

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute inertial term
inertiaTerm = weights( 1 ) * vel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute memory term
memoryTerm = weights( 2 ) * ( myBestPos - pos );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cooperation term
% Sample normally distributed number to perturbate the best position
cooperationTerm = weights( 3 ) * ( gbest * ( 1 + weights( 4 ) * normrnd( 0, 1 ) ) - pos );
communicationProbabilityMatrix = rand( 1, D ) < communicationProbability;
cooperationTerm = cooperationTerm .* communicationProbabilityMatrix;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute velocity
new_vel = inertiaTerm + memoryTerm + cooperationTerm;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check velocity limits
new_vel = ( new_vel > Vmax ) .* Vmax + ( new_vel <= Vmax ) .* new_vel;
new_vel = ( new_vel < Vmin ) .* Vmin + ( new_vel >= Vmin ) .* new_vel;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end