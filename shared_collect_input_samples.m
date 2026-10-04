%SHARED_COLLECT_INPUT_SAMPLES Generate quasi-random Sobol input grids.
%   Generates the Sobol input grids used by both the BBOB and CORNN
%   landscape evaluation scripts. Run this once before any data
%   collection. Requires MATLAB R2025a or later.
%
%   Output files (written to CFG.INPUT_DIR):
%       X_D{dim}_S100_R{sid}.csv   shape (100*dim, dim), range [-1, 1]
%
%   Every file written is printed with its size, and a summary line is
%   printed at the end.
%
%   See also SHARED_CONSOLIDATE_RAW_DATA, CORNN_CONFIG.

%   Copyright (c) 2026 Mario Andres Munoz Acosta
%   School of Computing and Information Systems
%   The University of Melbourne
%
%   This software is licensed under the PolyForm Noncommercial License 1.0.0.
%   You may use, copy, modify, and distribute this software for any
%   non-commercial purpose. Commercial use is prohibited.
%   Full license text: https://polyformproject.org/licenses/noncommercial/1.0.0

cfg          = cornn_config();
rng('default');
input_dir    = cfg.input_dir;
n_replicates = cfg.n_replicates;
dims         = cfg.dimensions;

fprintf('\n=== shared_collect_input_samples.m ===\n');
fprintf('Output directory: %s\n', input_dir);
fprintf('Dimensions: %s | Replicates: %d | Sample size: %d per dimension\n', ...
        mat2str(dims), n_replicates, cfg.sample_size);

n_written = 0;
for jj = 1:length(dims)
    d = dims(jj);
    samples_per_file = 100 * d;
    P = sobolset(d);
    P = scramble(P, 'MatousekAffineOwen');
    X = 2 .* net(P, n_replicates * samples_per_file) - 1;
    for ii = 1:n_replicates
        start_index = (ii - 1) * samples_per_file + 1;
        end_index   = ii * samples_per_file;
        segment     = X(start_index:end_index, :);
        filename    = fullfile(input_dir, ...
            ['X_D' num2str(d) '_S' num2str(cfg.sample_size) '_R' num2str(ii) '.csv']);
        writematrix(segment, filename);
        fprintf('[OK] Wrote %s (%d x %d)\n', filename, size(segment,1), size(segment,2));
        n_written = n_written + 1;
    end
end

fprintf('[OK] Wrote %d input grid files.\n', n_written);
fprintf('\n=== shared_collect_input_samples.m complete ===\n');
