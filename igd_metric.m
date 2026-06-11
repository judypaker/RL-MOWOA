function igd = igd_metric(front, true_pf)
% Inverse Generational Distance (IGD) for 2-objective fronts
% Matches the previous implementation
if isempty(front) || isempty(true_pf)
    igd = inf; 
    return
end
F = front(:,1:2);
R = true_pf(:,1:2);
igd = 0;
for i = 1:size(R,1)
    d = sqrt(sum((F - R(i,:)).^2,2));
    igd = igd + min(d);
end
igd = igd / size(R,1);
