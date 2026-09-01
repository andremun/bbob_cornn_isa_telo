%BBOB_COLLECT_FOPT Extract known optimal values from COCO observer logs.
%   Parses per-function, per-dimension .dat log files produced by a COCO
%   observer run and extracts the known optimal objective value (fopt)
%   for every (function, dimension, instance) combination, writing them
%   to a single CSV consumed by SHARED_CONSOLIDATE_RAW_DATA.
%
%   Input (expected under CFG.BBOB_META_DIR):
%       data_f{fid}/bbobexp_f{fid}_DIM{dim}.dat
%   one file per (function, dimension), each containing one row per
%   instance with columns [.., .., measured, best_measured, ..]; fopt is
%   recovered as column 4 minus column 3 of the numeric (non-comment)
%   rows.
%
%   Output (written to CFG.BBOB_META_DIR):
%       bbob_fopt.csv   columns [xFunction, Dimension, Instance, Fopt]
%
%   NOTE: this is a provenance script documenting how the shipped
%   bbob_fopt.csv was produced. The .dat observer logs it reads are
%   produced by BBOB_COLLECT_META.PY, which must be run first. Most users
%   should obtain bbob_fopt.csv directly from the paper's Figshare data
%   release instead of running this chain -- see README.md, "Step 0".
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

cfg      = cornn_config();
root_dir = cfg.bbob_meta_dir;

if ~isfolder(root_dir)
    error('bbob_collect_fopt:missingDir', ...
        'Expected directory not found: %s', root_dir);
end

fprintf('\n=== bbob_collect_fopt.m ===\n');
fprintf('Reading .dat logs from: %s\n', root_dir);
fprintf('Functions: %s | Dimensions: %s | Instances: %s\n', ...
        mat2str(cfg.bbob_function_ids), mat2str(cfg.dimensions), ...
        mat2str(cfg.bbob_instance_ids));

n_instances_expected = length(cfg.bbob_instance_ids);
fopt = [];
n_files_read = 0;

for ii = cfg.bbob_function_ids
    for dim = cfg.dimensions
        filename = sprintf('bbobexp_f%d_DIM%d.dat', ii, dim);
        filepath = fullfile(root_dir, sprintf('data_f%d', ii), filename);

        fid = fopen(filepath, 'r');
        if fid == -1
            error('bbob_collect_fopt:missingFile', ...
                ['Cannot open file: %s\n' ...
                 'This .dat log is produced by bbob_collect_meta.py -- run ' ...
                 'that script first, or obtain bbob_fopt.csv directly from ' ...
                 'the Figshare data release instead (see README.md, ' ...
                 '"Step 0").'], filepath);
        end

        data = [];
        tline = fgetl(fid);
        while ischar(tline)
            if isempty(tline) || tline(1) ~= '%'
                nums = sscanf(tline, '%f');
                if ~isempty(nums)
                    data = [data; nums']; %#ok<AGROW>
                end
            end
            tline = fgetl(fid);
        end
        fclose(fid);

        if size(data, 1) ~= n_instances_expected
            error('bbob_collect_fopt:unexpectedRowCount', ...
                ['%s: expected %d instance rows, found %d. ' ...
                 'The .dat file may be truncated, or CFG.BBOB_INSTANCE_IDS ' ...
                 'does not match the instances used in the observer run.'], ...
                filepath, n_instances_expected, size(data, 1));
        end

        fopt = vertcat(fopt, [ii .* ones(n_instances_expected, 1), ...
                               dim .* ones(n_instances_expected, 1), ...
                               cfg.bbob_instance_ids(:), ...
                               data(:,4) - data(:,3)]); %#ok<AGROW>
        n_files_read = n_files_read + 1;
    end
    fprintf('[OK] Function f%d: %d dimension(s) processed\n', ii, length(cfg.dimensions));
end

fopt = array2table(fopt, 'VariableNames', {'xFunction','Dimension','Instance','Fopt'});
out_path = fullfile(root_dir, 'bbob_fopt.csv');
writetable(fopt, out_path);
fprintf('[OK] Wrote %s (%d rows from %d .dat files)\n', out_path, height(fopt), n_files_read);

fprintf('\n=== bbob_collect_fopt.m complete ===\n');
