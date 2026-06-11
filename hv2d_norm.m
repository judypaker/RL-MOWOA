function hvn = hv2d_norm(F)
if isempty(F) || size(F,2) < 2
    hvn = 0;
    return
end
F = F(~any(isnan(F),2),1:2);
minv = min(F,[],1);
maxv = max(F,[],1);
range = max(maxv - minv, eps);
ref = maxv + 0.1 * range;
hv = hv2d(F, ref);
rect = (ref(1) - minv(1)) * (ref(2) - minv(2));
if rect <= 0
    hvn = 0;
else
    hvn = min(max(hv / rect, 0), 1);
end
