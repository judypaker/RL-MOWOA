%% Cited from NSGA-II All rights reserved.
function f = initialize_variables(NP, M, D, LB, UB)

%% function f = initialize_variables(N, M, D, LB, UB) 
% This function initializes the population. Each individual has the
% following at this stage:
%       * set of decision variables
%       * objective function values
% 
% where,
% NP - Population size
% M - Number of objective functions
% D - Number of decision variables
% LB - Lower bounds of decision variables
% UB - Upper bounds of decision variables

min_val = LB;
max_val = UB;
K = M + D;
f = zeros(NP, K);

%% Initialize each individual in population
% Using stratified sampling (LHS-like) to improve initial coverage
bins = (0:NP-1)'/NP;
X = zeros(NP, D);
for j = 1:D
    u = rand(NP, 1)/NP;
    samples01 = bins + u;
    perm = randperm(NP);
    samples01 = samples01(perm);
    X(:, j) = min_val(j) + samples01*(max_val(j) - min_val(j));
end

for i = 1:NP
    f(i, 1:D) = X(i, 1:D);
    f(i, D + 1: K) = evaluate_objective(f(i, 1:D));
end
