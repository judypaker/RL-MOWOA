classdef metrics_utils
methods(Static)
function igd = metric_igd(front, true_pf)
if isempty(front) || isempty(true_pf)
    igd = inf; return
end
F = front(:,1:2);
R = true_pf(:,1:2);
igd = 0;
for i = 1:size(R,1)
    d = sqrt(sum((F - R(i,:)).^2,2));
    igd = igd + min(d);
end
igd = igd / size(R,1);
end
function spr = metric_spread(front, true_pf)
if isempty(front) || size(front,1) < 2
    spr = NaN; return
end
F = sortrows(front(:,1:2),1);
if nargin < 2 || isempty(true_pf)
    df = 0; dl = 0;
else
    R = sortrows(true_pf(:,1:2),1);
    df = norm(F(1,:) - R(1,:));
    dl = norm(F(end,:) - R(end,:));
end
di = sqrt(sum(diff(F).^2,2));
dm = mean(di);
spr = (df + dl + sum(abs(di - dm))) / max(eps, (df + dl + (size(F,1)-1)*dm));
end
end
end
