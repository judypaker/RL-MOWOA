function spacing = spacing_metric(front)
% Spacing metric: std of nearest-neighbor distances in objective space
% Matches the previous implementation
if isempty(front) || size(front, 1) <= 1
    spacing = 0;
    return;
end
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
spacing = std(distances);
