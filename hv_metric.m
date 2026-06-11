function hvn = hv_metric(front)
% Normalized hypervolume for 2-objective fronts
% Wrapper to match previous usage; calls hv2d_norm
hvn = hv2d_norm(front);
