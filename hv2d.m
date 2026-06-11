function hv = hv2d(F, r)
if isempty(F)
    hv = 0;
    return
end
F = F(~any(isnan(F),2),:);
F = F(F(:,1)<=r(1) & F(:,2)<=r(2),:);
if isempty(F)
    hv = 0;
    return
end
F = sortrows(F,1);
hv = 0;
f2min = r(2);
for i=1:size(F,1)
    fi = F(i,:);
    if fi(2) < f2min
        hv = hv + (f2min - fi(2)) * (r(1) - fi(1));
        f2min = fi(2);
    end
end

