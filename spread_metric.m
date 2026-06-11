function spr = spread_metric(front, true_pf)
% Deb's spread (Delta) metric for 2-objective fronts
% Matches the previous implementation
if isempty(front) || size(front,1) < 2
    spr = NaN; 
    return
end
F = sortrows(front(:,1:2),1);
if nargin < 2 || isempty(true_pf)
    df = 0; 
    dl = 0;
else
    R = sortrows(true_pf(:,1:2),1);
    df = norm(F(1,:) - R(1,:));
    dl = norm(F(end,:) - R(end,:));
end
di = sqrt(sum(diff(F).^2,2));
dm = mean(di);
spr = (df + dl + sum(abs(di - dm))) / max(eps, (df + dl + (size(F,1)-1)*dm));
