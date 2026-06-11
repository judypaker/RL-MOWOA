classdef RL_Utils < handle
%% RL_Utils - Collection of reinforcement learning utility functions
% Contains functions for state acquisition, performance metric calculation, reward calculation, etc.

methods (Static)
        
function state = get_algorithm_state(Whale_pos, iteration, max_iteration, M, D)
%% Get current algorithm state
% Returns a normalized state vector: [convergence, diversity, progress, quality]

% Get the first non-dominated front
K = M + D;
first_front_indices = find(Whale_pos(:, K+1) == 1);
if isempty(first_front_indices)
    first_front_indices = 1:size(Whale_pos, 1);
end

if ~isempty(first_front_indices)
    first_front = Whale_pos(first_front_indices, D+1:D+M);
else
    first_front = [];
end

% 1. Convergence - based on the standard deviation of objective function values
if ~isempty(first_front) && size(first_front, 1) > 1
    convergence = 1 / (1 + mean(std(first_front, 0, 1)));
else
    convergence = 0.5;
end

% 2. Diversity - based on the distribution of solutions
if ~isempty(first_front)
    diversity = RL_Utils.calculate_diversity_metric(first_front);
else
    diversity = 0;
end

% 3. Iteration progress
progress = iteration / max_iteration;

% 4. Population quality - based on the ratio of non-dominated solutions
quality = length(first_front_indices) / size(Whale_pos, 1);

% Combine into state vector
state = [convergence, diversity, progress, quality];

% Ensure state is within [0, 1] range
state = max(0, min(1, state));

end
        
function diversity = calculate_diversity_metric(front)
%% Calculate diversity metric

if isempty(front) || size(front, 1) <= 1
    diversity = 0;
    return;
end

% Calculate span in objective space, further increasing diversity weight
if size(front, 1) > 1
    ranges = max(front) - min(front);
    % Significantly increase the weight factor for diversity calculation
     diversity = sum(ranges) * 2.5;  % Increased from 2.0 to 2.5
     
     % Enhance additional diversity reward mechanism
     if size(front, 1) > 2
         % Calculate average distance between solutions as additional diversity metric
         distances = pdist(front);
         avg_distance = mean(distances);
         diversity = diversity + avg_distance * 0.8;  % Increased from 0.5 to 0.8
         
         % Add diversity reward based on variance
         obj_variance = sum(var(front));
         diversity = diversity + obj_variance * 0.3;
     end
else
    diversity = 0;
end

end
        
function metrics = calculate_performance_metrics(Whale_pos, M, D)
%% Calculate performance metrics

K = M + D;
first_front_indices = find(Whale_pos(:, K+1) == 1);
if isempty(first_front_indices)
    first_front_indices = 1:min(10, size(Whale_pos, 1));
end

first_front = Whale_pos(first_front_indices, D+1:D+M);

metrics = struct();

% 1. Hypervolume (normalized)
metrics.hypervolume = hv2d_norm(first_front);

% 2. Spacing metric
metrics.spacing = RL_Utils.calculate_spacing_metric(first_front);

% 3. Convergence metric
metrics.convergence = RL_Utils.calculate_convergence_metric(first_front);

% 4. Diversity metric
metrics.diversity = RL_Utils.calculate_diversity_metric(first_front);

% 5. IGD calculation (using GD to ideal point)
metrics.igd = RL_Utils.calculate_gd_to_ideal(first_front);
metrics.spread = metrics_utils.metric_spread(first_front, []);

end
        
function hv = calculate_hypervolume_simple(front)
%% Simplified hypervolume calculation

% Input validation
if isempty(front) || size(front, 1) == 0 || size(front, 2) < 2
    hv = 0;
    return;
end

% Use simplified hypervolume calculation method
% For 2-objective problems, use trapezoidal integration
if size(front, 2) == 2
    % Sort solutions
    [sorted_front, ~] = sortrows(front(:,1:2), 1);
    max1 = max(sorted_front(:,1));
    min1 = min(sorted_front(:,1));
    max2 = max(sorted_front(:,2));
    min2 = min(sorted_front(:,2));
    range1 = max1 - min1; if range1 == 0, range1 = 1; end
    range2 = max2 - min2; if range2 == 0, range2 = 1; end
    if all(sorted_front(:,1) >= 0) && all(sorted_front(:,1) <= 1.05) && all(sorted_front(:,2) >= 0) && all(sorted_front(:,2) <= 1.05)
        ref_point = [1.1, 1.1];
    else
        ref_point = [max1 + 0.1 * range1, max2 + 0.1 * range2];
    end
    hv = 0;
    n = size(sorted_front, 1);
    for i = 1:n
        if i == n
            width = ref_point(1) - sorted_front(i, 1);
        else
            width = sorted_front(i+1, 1) - sorted_front(i, 1);
        end
        height = ref_point(2) - sorted_front(i, 2);
        hv = hv + max(0, width) * max(0, height);
    end
else
    % For more objectives, use a simplified method
    if size(front, 1) == 1
        ref_point = front + 1;  % Simple case for a single point
    else
        range_vals = max(front) - min(front);
        range_vals(range_vals == 0) = 1;  % Avoid division by zero
        ref_point = max(front) + 0.1 * range_vals;
    end
    hv = 0;
    for i = 1:size(front, 1)
        volume = prod(max(0, ref_point - front(i, :)));
        hv = hv + volume;
    end
    hv = hv / max(1, size(front, 1));
end

% Ensure non-negative
hv = max(0, hv);

end
        
function spacing = calculate_spacing_metric(front)
%% Calculate spacing metric

if isempty(front) || size(front, 1) <= 1
    spacing = 0;
    return;
end

% Calculate distance to the nearest neighbor for each solution
distances = [];
for i = 1:size(front, 1)
    min_dist = inf;
    for j = 1:size(front, 1)
        if i ~= j
            dist = norm(front(i, :) - front(j, :));
            min_dist = min(min_dist, dist);
        end
    end
    distances = [distances; min_dist];
end

% Spacing metric is the standard deviation of distances
spacing = std(distances);

end
        
function convergence = calculate_convergence_metric(front)
%% Calculate convergence metric

if isempty(front)
    convergence = inf;
    return;
end

% Simplified convergence metric: average distance to origin
convergence = mean(sqrt(sum(front.^2, 2)));

% Normalize
convergence = 1 / (1 + convergence);

end

function gd = calculate_gd_to_ideal(front)
%% Calculate generation distance to ideal point (normalized)

if isempty(front)
    gd = inf;
    return;
end
ideal = min(front, [], 1);
ranges = max(front, [], 1) - min(front, [], 1);
ranges(ranges == 0) = 1;
diffs = (front - ideal) ./ ranges;
d = sqrt(sum(diffs.^2, 2));
gd = mean(d);
end
        
        
function reward = calculate_rl_reward(current_metrics, iteration, max_iteration)
%% Calculate reinforcement learning reward

% Reward weights (HV priority, IGD secondary, Spacing and Spread auxiliary, low weight for Diversity)
w1_base = 0.80; w6_base = 0.42; w7_base = 0.26; w2 = 0.26; w3 = 0.03; w4_base = 0.02; w5 = 0.06;
prog = iteration / max_iteration;
w1 = w1_base + 0.4 * prog;
w6 = w6_base + 0.2 * prog;
w7 = w7_base + 0.15 * prog;
w4 = w4_base * (1 - prog) + 0.03 * prog;

% Normalize metrics to [0, 1] range
hv_reward = min(1, current_metrics.hypervolume);
hv_reward = hv_reward.^1.22;
sp_reward = 1 / (1 + current_metrics.spacing);
conv_reward = current_metrics.convergence;
% Further enhance sensitivity of diversity reward
div_reward = min(1, current_metrics.diversity / 120);
spread_reward = 1 / (1 + current_metrics.spread);
igd_reward = 1 / (1 + current_metrics.igd);

% Progress reward
progress_reward = iteration / max_iteration;

% Aggregate reward
reward = w1 * hv_reward + w2 * sp_reward + w3 * conv_reward + w4 * div_reward + w5 * progress_reward + w6 * igd_reward + w7 * spread_reward;

% Ensure reward is within reasonable range
reward = max(-1, min(2, reward));

end
        
end

end
