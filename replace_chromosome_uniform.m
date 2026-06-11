function f  = replace_chromosome_uniform(intermediate_chromosome, M, D, NP, num_bins)

if nargin < 5
    num_bins = 10;
end

% hv_contrib_2d will be defined as a local function at the end of the file

[~, m] = size(intermediate_chromosome);
f = zeros(NP, m);

sorted_chromosome = sortrows(intermediate_chromosome, M + D + 1);
max_rank = max(intermediate_chromosome(:, M + D + 1));

previous_index = 0;
for i = 1:max_rank
    current_index = find(sorted_chromosome(:, M + D + 1) == i, 1, 'last');
    if current_index > NP
        remaining = NP - previous_index;
        temp_pop = sorted_chromosome(previous_index + 1: current_index, :);
        % Preserve extreme solutions
        objs = temp_pop(:, D + 1:D + M);
        extreme_idx = [];
        for mm = 1:M
            [~, imin] = min(objs(:, mm));
            [~, imax] = max(objs(:, mm));
            extreme_idx = unique([extreme_idx; imin; imax]);
        end
        selected = false(size(temp_pop, 1), 1);
        fill_count = min(remaining, numel(extreme_idx));
        for j = 1:fill_count
            selected(extreme_idx(j)) = true;
            f(previous_index + j, :) = temp_pop(extreme_idx(j), :);
        end
        if fill_count >= remaining
            return;
        end
        start_pos = previous_index + fill_count;
        % Grid-based uniform selection for remaining spots
        objs = temp_pop(:, D + 1:D + M);
        bins_idx = zeros(size(temp_pop, 1), M);
        for mm = 1:M
            fmax = max(objs(:, mm));
            fmin = min(objs(:, mm));
            range_val = fmax - fmin;
            if range_val == 0
                bins_idx(:, mm) = 0;
            else
                normv = (objs(:, mm) - fmin) / range_val;
                b = floor(normv * num_bins);
                b(b >= num_bins) = num_bins - 1;
                b(b < 0) = 0;
                bins_idx(:, mm) = b;
            end
        end
        % Combine into a single cell ID
        cell_id = bins_idx(:, 1);
        for mm = 2:M
            cell_id = cell_id * num_bins + bins_idx(:, mm);
        end

        cd = temp_pop(:, M + D + 2);
        cd(~isfinite(cd)) = 0;
        hvc = hv_contrib_2d(objs);
        % Continue selecting individuals not already taken by extremes
        occ = containers.Map('KeyType', 'double', 'ValueType', 'double');
        norm_range = sqrt(sum((max(objs) - min(objs)).^2));
        if norm_range <= 0, norm_range = 1; end
        for k = 1:(remaining - fill_count)
            % Calculate current occupancy for each cell
            keys = unique(cell_id(~selected));
            min_occ = inf;
            cand_cells = [];
            for kk = 1:numel(keys)
                key = keys(kk);
                if isKey(occ, key)
                    o = occ(key);
                else
                    o = 0;
                end
                if o < min_occ
                    min_occ = o;
                    cand_cells = key;
                elseif o == min_occ
                    cand_cells = [cand_cells, key];
                end
            end
            % Select top-rated unselected individual in candidate cells (HV contribution + min distance to selected)
            cand_idx = find(ismember(cell_id, cand_cells) & ~selected);
            if isempty(cand_idx)
                cand_idx = find(~selected);
            end
            selected_objs = objs(selected, :);
            cand_objs = objs(cand_idx, :);
            dmin = zeros(length(cand_idx), 1);
            if ~isempty(selected_objs)
                for cc = 1:length(cand_idx)
                    v = cand_objs(cc, :);
                    d = sqrt(sum((selected_objs - v).^2, 2));
                    dmin(cc) = min(d);
                end
            end
            dmin = dmin / norm_range;
            hv_local = hvc(cand_idx);
            hv_local = hv_local ./ max(1e-12, max(hv_local));
            threshold = floor((remaining - fill_count)/2);
            if k <= max(1, threshold)
                score = 0.5 * hv_local + 0.5 * dmin;
                [~, local_best] = max(score);
            else
                [~, local_best] = max(dmin);
            end
            pick_idx = cand_idx(local_best);
            selected(pick_idx) = true;
            key = cell_id(pick_idx);
            if isKey(occ, key)
                occ(key) = occ(key) + 1;
            else
                occ(key) = 1;
            end
            f(start_pos + k, :) = temp_pop(pick_idx, :);
        end
        return;
    elseif current_index < NP
        f(previous_index + 1: current_index, :) = sorted_chromosome(previous_index + 1: current_index, :);
    else
        f(previous_index + 1: current_index, :) = sorted_chromosome(previous_index + 1: current_index, :);
        return;
    end
previous_index = current_index;
end
