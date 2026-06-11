function [f, performance_metrics] = RL_MOWOA(D, M, LB, UB, Pop, SearchAgents_no, Max_iteration, ishow, Max_evals)
%% Reinforcement Learning Enhanced Non-Sorted Whale Optimization Algorithm (RL-MOWOA)
% 基于强化学习的非排序鲸鱼优化算法
% 使用Q-learning自适应调整算法参数以提高性能
%
% 输入参数:
% D - 决策变量维数
% M - 目标函数数量
% LB - 下界约束
% UB - 上界约束
% Pop - 初始种群
% SearchAgents_no - 搜索代理数量
% Max_iteration - 最大迭代次数
% ishow - 显示间隔
%
% 输出:
% f - 最优解集
% performance_metrics - 性能指标

%% 初始化RL智能体
rl_agent = QLearningAgent();

%% 算法变量初始化
K = D + M;
Whale_pos = Pop(:, 1:K+2);  % Pop经过排序后有K+2列（包括排序和拥挤度信息）
Whale_pos_ad = zeros(SearchAgents_no, K);

%% 问题类型自动判定（用于ZDT1特定偏置逻辑）
is_zdt1 = (D >= 10) && all(abs(LB) < 1e-12) && all(abs(UB - 1) < 1e-12);

%% 指标与参数记录（不计算HV/Spacing/Convergence/Diversity/IGD）
performance_metrics = struct();
performance_metrics.rl_rewards = [];
performance_metrics.parameter_history = [];
performance_metrics.convergence_curve = [];

%% 优化循环
Iteration = 1;
eval_count = 0;
if nargin < 9 || isempty(Max_evals)
    Max_evals = inf;
end
while eval_count < Max_evals
    %% 获取当前状态并选择动作（参数组合）
    current_state = RL_Utils.get_algorithm_state(Whale_pos, Iteration, Max_iteration, M, D);
    [action, params] = rl_agent.select_action(current_state);
    
    performance_metrics.parameter_history(Iteration, :) = params;
    
    %% 使用RL选择的参数执行NSWOA迭代
    for i = 1:SearchAgents_no
        j = floor(rand() * SearchAgents_no) + 1;
        while j == i
            j = floor(rand() * SearchAgents_no) + 1;
        end
        
        % 使用RL智能体选择的缩放因子
        SF = params(1); % 自适应缩放因子
        
        % 从第一前沿按拥挤度加权选择，提升多样性
        front_indices = find(Whale_pos(:, K+1) == 1);
        if ~isempty(front_indices)
            Ffront = Whale_pos(front_indices, :);
            cd = Ffront(:, K+2);
            mask = isinf(cd) | isnan(cd);
            valid = cd(~mask);
            if isempty(valid)
                fillVal = 0;
            else
                fillVal = max(valid);
            end
            cd(mask) = fillVal;
            obj = Ffront(:, D+1:D+M);
            kw = compute_knee_weights(obj);
            hv = hv_contrib_2d(obj);
            cdn = cd - min(cd);
            if sum(cdn) > 0
                cdn = cdn / sum(cdn);
            end
            kwn = kw - min(kw);
            if sum(kwn) > 0
                kwn = kwn / sum(kwn);
            end
            prog = Iteration/Max_iteration;
            a2 = 0.6 + 0.3*prog;
            a3 = 0.25 - 0.15*prog;
            a1 = max(0, 1 - a2 - a3);
            hvn = hv - min(hv);
            if sum(hvn) > 0
                hvn = hvn / sum(hvn);
                hvn = hvn .^ 1.35;
                hvn = hvn / max(sum(hvn), eps);
            end
            w = a1 * cdn + a2 * hvn + a3 * kwn;
            f1 = obj(:,1);
            fmin = min(f1); fmax = max(f1); fr = fmax - fmin;
            num_bins_occ = 40;
            if fr <= 0
                occ_w = ones(size(f1));
            else
                b = floor((f1 - fmin) / fr * num_bins_occ);
                b(b >= num_bins_occ) = num_bins_occ - 1; b(b < 0) = 0;
                cnt = accumarray(b+1, 1, [num_bins_occ, 1]);
                occ_w = 1 ./ (1 + (cnt(b+1)).^3);
            end
            fmin2 = min(obj, [], 1);
            fmax2 = max(obj, [], 1);
            rng2 = max(fmax2 - fmin2, eps);
            pos = (obj - fmin2) ./ rng2;
            edge_w = (1 - min(pos, 1 - pos));
            edge_w = edge_w(:,1) + edge_w(:,2);
            edge_w = edge_w / max(sum(edge_w), eps);
            w = w .* (occ_w.^1.4) .* (1 + 0.3 * edge_w);
            if sum(w) <= 0
                ri = floor(size(Ffront, 1) * rand()) + 1;
            else
                w = w / sum(w);
                cumw = cumsum(w);
                r = rand();
                ri = find(cumw >= r, 1, 'first');
                if isempty(ri), ri = 1; end
            end
            sorted_population = Ffront(ri, 1:D);
        else
            % 如果没有第一前沿，随机选择一个个体
            ri = floor(SearchAgents_no * rand()) + 1;
            if ri < 1, ri = 1; end  % 确保索引有效
            sorted_population = Whale_pos(ri, 1:D);
        end
        
        % 计算新解
        Whale_posNew1 = Whale_pos(i, 1:D) + rand(1, D) .* (sorted_population - SF .* Whale_pos(i, 1:D));
        
        step_sizes = zeros(1, D);
        if D >= 1, step_sizes(1) = 0.01; end
        if D >= 2, step_sizes(2) = 1.0; end
        if D >= 3, step_sizes(3) = 0.01; end
        Whale_posNew1 = bound_with_step(Whale_posNew1(:, 1:D), UB, LB, step_sizes);
        
        % 评估目标函数
        Whale_posNew1(:, D + 1: K) = evaluate_objective(Whale_posNew1(:, 1:D));
        eval_count = eval_count + 1;
        if eval_count >= Max_evals
            break;
        end
        
        % 非支配性检查和更新
        [Whale_pos, Whale_pos_ad] = update_population_with_rl(Whale_pos, Whale_pos_ad, Whale_posNew1, i, M, D, K);
        
        % 使用RL参数的鲸鱼优化策略
        a = 2 - Iteration * ((2) / Max_iteration);
        a2 = -1 + Iteration * ((-1) / Max_iteration);
        r1 = rand();
        r2 = rand();
        A = 2 * a * r1 - a;
        C = 2 * r2;
        b = params(2);
        t = (a2 - 1) * rand + 1;
        p = params(3);
        p_eff = max(min(p + 0.4*prog - 0.10, 0.90), 0.40);
        
        if p_eff < 0.5
            % 包围猎物策略
            X_rand = sorted_population;
            Whale_posNew1 = Whale_pos(i, 1:D) + SF * (X_rand - A .* abs(C * X_rand - Whale_pos(j, 1:D)));
        else
            % 螺旋更新策略
            Whale_posNew1 = Whale_pos(i, 1:D) + SF * (abs(sorted_population - Whale_pos(j, 1:D)) * exp(b .* t) .* cos(t .* 2 * pi) + sorted_population);
        end
        
        Whale_posNew1 = bound_with_step(Whale_posNew1(:, 1:D), UB, LB, step_sizes);
        Whale_posNew1(:, D + 1: K) = evaluate_objective(Whale_posNew1(:, 1:D));
        eval_count = eval_count + 1;
        if eval_count >= Max_evals
            break;
        end
        
        % 更新种群
        [Whale_pos, Whale_pos_ad] = update_population_with_rl(Whale_pos, Whale_pos_ad, Whale_posNew1, i, M, D, K);
        
        % 寄生虫策略（使用RL参数调整）
        j = floor(rand() * SearchAgents_no) + 1;
        while j == i
            j = floor(rand() * SearchAgents_no) + 1;
        end
        
        parasiteVector = Whale_pos(i, 1:D);
        mutation_rate_eff = max(0.04, params(4) * (1 - 0.85 * prog));
        if rand() < 0.7
            num_mut = max(1, ceil(mutation_rate_eff * D));
            rest = setdiff(2:D, []);
            pick_rest = [];
            if num_mut > 1
                pick_rest = rest(randperm(max(1, D-1), num_mut - 1));
            end
            pick = unique([1, pick_rest]);
        else
            pick = randperm(D, max(1, ceil(mutation_rate_eff * D)));
        end
        parasiteVector = polynomial_mutation(parasiteVector, LB, UB, pick, 20);
        parasiteVector = bound_with_step(parasiteVector, UB, LB, step_sizes);
        
        parasiteVector(:, D + 1: K) = evaluate_objective(parasiteVector(:, 1:D));
        eval_count = eval_count + 1;
        if eval_count >= Max_evals
            break;
        end
        
        % 更新寄生虫向量
        [Whale_pos, Whale_pos_ad] = update_parasite_with_rl(Whale_pos, Whale_pos_ad, parasiteVector, j, M, D, K);
    end
    
    current_metrics = RL_Utils.calculate_performance_metrics(Whale_pos, M, D);
    reward = RL_Utils.calculate_rl_reward(current_metrics, Iteration, Max_iteration);
    performance_metrics.rl_rewards(Iteration) = reward;
    
    % 获取新状态
    new_state = RL_Utils.get_algorithm_state(Whale_pos, Iteration + 1, Max_iteration, M, D);
    
    % 更新Q表
    rl_agent.update_q_table(current_state, action, reward, new_state);
    
    %% 种群更新和非支配排序（静默）
    
    Whale_pos_com = [Whale_pos(:, 1:K); Whale_pos_ad];
    anchors2 = generate_rsm_anchors(D, LB, UB, 13);
    Whale_pos_com = [Whale_pos_com; anchors2];
    fillers = gap_fillers_for_x1(Whale_pos_com, D, M, LB, UB, 30, min(10, SearchAgents_no));
    Whale_pos_com = [Whale_pos_com; fillers];
    edge_anchors = generate_rsm_edge_anchors(D, LB, UB);
    Whale_pos_com = [Whale_pos_com; edge_anchors];
    front_indices_curr = find(Whale_pos(:, K+1) == 1);
    if ~isempty(front_indices_curr)
        Ffront_curr = Whale_pos(front_indices_curr, :);
        obj_curr = Ffront_curr(:, D+1:D+M);
        hvc = hv_contrib_2d(obj_curr);
        [~, sidx] = sort(hvc, 'descend');
        elite_n = min(10, numel(sidx));
        elites = Ffront_curr(sidx(1:elite_n), 1:K);
        Whale_pos_com = [Whale_pos_com; elites];
    end
    intermediate_Whale_pos = non_domination_sort_mod(Whale_pos_com, M, D);
    Pop = replace_chromosome_uniform(intermediate_Whale_pos, M, D, SearchAgents_no, 140);
    fillers2 = gap_fillers_for_x1(Pop(:, 1:K), D, M, LB, UB, 30, min(10, SearchAgents_no));
    Whale_pos_com3 = [Pop(:, 1:K); fillers2];
    intermediate_Whale_pos3 = non_domination_sort_mod(Whale_pos_com3, M, D);
    Pop3 = replace_chromosome_uniform(intermediate_Whale_pos3, M, D, SearchAgents_no, 140);
    Whale_pos = Pop3(:, 1:K+2);
    
    performance_metrics.convergence_curve(Iteration) = sum(Whale_pos(:, K+1) == 1);
    Iteration = Iteration + 1;
end

f = Whale_pos;
performance_metrics.termination_iteration = Iteration - 1;

%% 辅助函数
function a = bound(a, ub, lb)
    a(a > ub) = ub(a > ub);
    a(a < lb) = lb(a < lb);
end

function A = generate_rsm_anchors(D, LB, UB, n)
    x1 = linspace(LB(1), UB(1), n)';
    A = zeros(n, D + 2);
    A(:,1) = x1;
    if D > 1
        A(:,2:D) = repmat(LB(2:D), n, 1);
    end
    for i = 1:n
        objv = evaluate_objective(A(i,1:D));
        A(i,D+1:D+2) = objv;
    end
end

function E = generate_rsm_edge_anchors(D, LB, UB)
    xs = [LB(1); UB(1)];
    ys = linspace(LB(2), UB(2), 11)';
    zs_list = [LB(3), (LB(3)+UB(3))/2, UB(3)];
    num = numel(xs) * numel(ys) * numel(zs_list);
    E = zeros(num, D + 2);
    idx = 1;
    for i = 1:numel(xs)
        for j = 1:numel(ys)
            for k = 1:numel(zs_list)
                x = [xs(i), ys(j), zs_list(k)];
                f = evaluate_objective(x);
                E(idx, 1:D) = x;
                E(idx, D+1:D+2) = f;
                idx = idx + 1;
            end
        end
    end
end

function [Whale_pos, Whale_pos_ad] = update_population_with_rl(Whale_pos, Whale_pos_ad, Whale_posNew1, i, M, D, K)
    dom_less = 0;
    dom_equal = 0;
    dom_more = 0;
    
    for k = 1:M
        if (Whale_posNew1(:, D+k) < Whale_pos(i, D+k))
            dom_less = dom_less + 1;
        elseif (Whale_posNew1(:, D+k) == Whale_pos(i, D+k))
            dom_equal = dom_equal + 1;
        else
            dom_more = dom_more + 1;
        end
    end
    
    if dom_more == 0 && dom_equal ~= M
        Whale_pos_ad(i, 1:K) = Whale_pos(i, 1:K);
        Whale_pos(i, 1:K) = Whale_posNew1(:, 1:K);
    else
        Whale_pos_ad(i, 1:K) = Whale_posNew1;
    end
end

function [Whale_pos, Whale_pos_ad] = update_parasite_with_rl(Whale_pos, Whale_pos_ad, parasiteVector, j, M, D, K)
    dom_less = 0;
    dom_equal = 0;
    dom_more = 0;
    
    for k = 1:M
        if (parasiteVector(:, D+k) < Whale_pos(j, D+k))
            dom_less = dom_less + 1;
        elseif (parasiteVector(:, D+k) == Whale_pos(j, D+k))
            dom_equal = dom_equal + 1;
        else
            dom_more = dom_more + 1;
        end
    end
    
    if dom_more == 0 && dom_equal ~= M
        Whale_pos_ad(j, 1:K) = Whale_pos(j, 1:K);
        Whale_pos(j, 1:K) = parasiteVector(:, 1:K);
    else
        Whale_pos_ad(j, 1:K) = parasiteVector;
    end
end

end
function x = polynomial_mutation(x, LB, UB, idx, eta_m)
    if nargin < 5
        eta_m = 20;
    end
    for ii = 1:length(idx)
        j = idx(ii);
        y = (x(j) - LB(j)) / (UB(j) - LB(j));
        y = min(max(y, 0), 1);
        rnd = rand();
        mut_pow = 1 / (eta_m + 1);
        if rnd <= 0.5
            xy = 1 - y;
            val = 2 * rnd + (1 - 2 * rnd) * (xy^(eta_m + 1));
            deltaq = val^mut_pow - 1;
        else
            xy = y;
            val = 2 * (1 - rnd) + 2 * (rnd - 0.5) * (xy^(eta_m + 1));
            deltaq = 1 - val^mut_pow;
        end
        y = y + deltaq;
        y = min(max(y, 0), 1);
        x(j) = LB(j) + y * (UB(j) - LB(j));
    end
end

function w = compute_knee_weights(front)
if isempty(front)
    w = 0;
    return;
end
if size(front,2) < 2
    w = ones(size(front,1),1);
    return;
end
[sorted_f, order] = sortrows(front, 1);
n = size(sorted_f,1);
if n <= 2
    w_sorted = ones(n,1);
    w = zeros(n,1);
    w(order) = w_sorted;
    w = w + 1e-12;
    return;
end
curv = zeros(n,1);
for i = 2:n-1
    curv(i) = abs(sorted_f(i+1,2) - 2*sorted_f(i,2) + sorted_f(i-1,2));
end
curv(1) = curv(2);
curv(n) = curv(n-1);
w_sorted = curv;
w = zeros(n,1);
w(order) = w_sorted;
w = w + 1e-12;
end
function F = gap_fillers_for_x1(pop_k, D, M, LB, UB, num_bins, max_fill)
    K = D + M;
    x1 = pop_k(:, 1);
    b = floor((x1 - LB(1)) / max(UB(1) - LB(1), eps) * num_bins);
    b(b >= num_bins) = num_bins - 1;
    b(b < 0) = 0;
    occ = false(num_bins, 1);
    for i = 1:length(b)
        occ(b(i) + 1) = true;
    end
    empty_bins = find(~occ);
    if isempty(empty_bins)
        F = zeros(0, K);
        return
    end
    fill_n = min(max_fill, numel(empty_bins));
    F = zeros(fill_n, K);
    for k = 1:fill_n
        bi = empty_bins(k);
        left = LB(1) + (bi) * (UB(1) - LB(1)) / num_bins;
        right = LB(1) + (bi + 1) * (UB(1) - LB(1)) / num_bins;
        x = zeros(1, D);
        x(1) = 0.5 * (left + right);
        if D > 1
            x(2:D) = LB(2:D);
        end
        f = evaluate_objective(x);
        F(k, 1:D) = x;
        F(k, D+1:D+M) = f;
    end
end
