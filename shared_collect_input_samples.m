% shared_collect_input_samples.m
%
% Generate the quasi-random Sobol input grids used by both the BBOB and
% CORNN landscape evaluation scripts. Run this once before any data
% collection. Requires MATLAB R2025a or later.
%
% Output files (written to cfg.input_dir):
%   X_D{dim}_S100_R{sid}.csv   shape (100*dim, dim)   float64, range [-1, 1]
%
% See also: shared_consolidate_raw_data.m, cornn_config.m

% Copyright (c) 2026 Mario Andres Munoz Acosta
% School of Computing and Information Systems
% The University of Melbourne
%
% Date: May 2026
%
% This software is licensed under the PolyForm Noncommercial License 1.0.0.
% You may use, copy, modify, and distribute this software for any
% non-commercial purpose. Commercial use is prohibited.
% Full license text: https://polyformproject.org/licenses/noncommercial/1.0.0

cfg = cornn_config();
rng('default');
input_dir = cfg.input_dir;
n_replicates = cfg.n_replicates;
dims = cfg.dimensions;
for jj=1:length(dims)
    d = dims(jj);
    samples_per_file = 100*d;
    P = sobolset(d);
    P = scramble(P,'MatousekAffineOwen');
    X = 2.*net(P,n_replicates*samples_per_file)-1;
    for ii=1:n_replicates
        start_index = (ii-1) * samples_per_file + 1;
        end_index = ii * samples_per_file;
        segment = X(start_index:end_index,:);
        filename = fullfile(input_dir, ['X_D' num2str(d) '_S' num2str(cfg.sample_size) '_R' num2str(ii) '.csv']);
        writematrix(segment, filename);
    end
end