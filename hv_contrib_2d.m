function w = hv_contrib_2d(F)
if isempty(F)
    w = 0;
    return
end
F = F(:,1:2);
[Sf, order] = sortrows(F, 1);
n = size(Sf,1);
minv = min(Sf,[],1);
maxv = max(Sf,[],1);
range = max(maxv - minv, eps);
ref = maxv + 0.1 * range;
contrib = zeros(n,1);
best2 = ref(2);
for i=1:n
    f = Sf(i,:);
    if f(2) < best2
        width = ref(1) - f(1);
        height = best2 - f(2);
        contrib(i) = max(0, width) * max(0, height);
        best2 = f(2);
    else
        contrib(i) = 0;
    end
end
w = zeros(n,1);
w(order) = contrib;
w = w + 1e-12;

