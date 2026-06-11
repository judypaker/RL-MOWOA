function x = bound_with_step(x, UB, LB, step_sizes)
%% Boundary handling function with step size constraints
% Input:
%   x - Decision variable vector
%   UB - Upper bound vector
%   LB - Lower bound vector
%   step_sizes - Step size vector [wall thickness, density, ratio]
% Output:
%   x - Processed decision variable vector

% Default step size settings
if nargin < 4
    step_sizes = [0.01, 1, 0.01];
end

% Ensure input is a row vector
if size(x, 1) > size(x, 2)
    x = x';
end

% Boundary and step size constraints for each variable
for i = 1:length(x)
    % Boundary reflection (reflect first, then quantize)
    if x(i) < LB(i)
        x(i) = LB(i) + (LB(i) - x(i));
    elseif x(i) > UB(i)
        x(i) = UB(i) - (x(i) - UB(i));
    end

    % Step size quantization
    if step_sizes(i) > 0
        offset = x(i) - LB(i);
        adjusted_offset = round(offset / step_sizes(i)) * step_sizes(i);
        x(i) = LB(i) + adjusted_offset;
    end

    % Final clipping to prevent out-of-bounds after reflection and quantization
    if x(i) > UB(i)
        x(i) = UB(i);
    elseif x(i) < LB(i)
        x(i) = LB(i);
    end
end

end
