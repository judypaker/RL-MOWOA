function Main_RL_MOWOA()
%% RL-MOWOA Main Script

clc;
clear;
close all;

fprintf('\n');
fprintf('========================================\n');
fprintf('  RL-MOWOA Reinforcement Learning Optimization\n');
fprintf('========================================\n');
fprintf('\n');

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
addpath(genpath(script_dir));
if isempty(which('RL_MOWOA'))
    error('RL_MOWOA function file not found. Please ensure RL_MOWOA.m is located in: %s', script_dir);
end

%% Run
run_rsm_bt1_save_pareto_runs();

end

function run_rsm_bt1_save_pareto_runs()
params = struct();
params.D = 3;
params.M = 2;
params.LB = [0.3, 80, 0];
params.UB = [1, 400, 1];
params.Max_iteration = 1000;
params.SearchAgents_no = 100;
params.ishow = 10;
params.num_runs = 30;
params.Max_evals = 50000;

base_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(base_dir,'results','RSM');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

for run = 1:params.num_runs
    rng(33 + run - 1);
    chromosome = initialize_variables(params.SearchAgents_no, params.M, params.D, params.LB, params.UB);
    intermediate_chromosome = non_domination_sort_mod(chromosome, params.M, params.D);
    Population = replace_chromosome(intermediate_chromosome, params.M, params.D, params.SearchAgents_no);

    rng(33 + run - 1);
    chromosome = initialize_variables(params.SearchAgents_no, params.M, params.D, params.LB, params.UB);
    intermediate_chromosome = non_domination_sort_mod(chromosome, params.M, params.D);
    Population = replace_chromosome(intermediate_chromosome, params.M, params.D, params.SearchAgents_no);

    [rl_mowoa_result, ~] = RL_MOWOA(params.D, params.M, params.LB, params.UB, Population, params.SearchAgents_no, params.Max_iteration, params.ishow, params.Max_evals);
    rl_front = extract_pareto_front(rl_mowoa_result, params.M, params.D);
    out_file = fullfile(out_dir, sprintf('pareto_run_%02d.csv', run));
    writematrix(rl_front, out_file);
end

fprintf('\nSaved 30 independent runs of RSM(BT1) Pareto fronts to directory: %s\n', out_dir);
end

function pareto_front = extract_pareto_front(population, M, D)
%% Extract Pareto Front

K = M + D;
first_front_indices = find(population(:, K+1) == 1);
if isempty(first_front_indices)
    first_front_indices = 1:min(10, size(population, 1));
end

pareto_front = population(first_front_indices, D+1:D+M);

end
