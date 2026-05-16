function cfg = cornn_config()
% CORNN_CONFIG  Central configuration for the BBOB-CORNN ISA project.
%
%   cfg = cornn_config() returns a struct containing all project constants
%   and directory paths. All paths are derived from a single root directory,
%   which can be overridden by the environment variable BBOB_CORNN_ROOT.
%
%   This function mirrors cornn/config.py and is the single source of truth
%   for all MATLAB scripts.
%
%   Directories are created if they do not exist.
%
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

% =============================================================================
% ROOT DIRECTORY  (override via environment variable BBOB_CORNN_ROOT)
% =============================================================================

root_env = getenv('BBOB_CORNN_ROOT');
if ~isempty(root_env)
    cfg.root_dir = root_env;
elseif ispc()
    cfg.root_dir = 'D:/bbob_cornn_isa';
else
    cfg.root_dir = fullfile(getenv('HOME'), 'punim0320', 'bbob_cornn_isa');
end

% =============================================================================
% SUBDIRECTORY LAYOUT  (mirrors cornn/config.py)
% =============================================================================

cfg.input_dir      = fullfile(cfg.root_dir, 'input');

cfg.bbob_raw_dir   = fullfile(cfg.root_dir, 'bbob', 'raw');
cfg.bbob_ela_dir   = fullfile(cfg.root_dir, 'bbob', 'ela');
cfg.bbob_ng_dir    = fullfile(cfg.root_dir, 'bbob', 'nevergrad');
cfg.bbob_adam_dir  = fullfile(cfg.root_dir, 'bbob', 'adam');
cfg.bbob_meta_dir  = fullfile(cfg.root_dir, 'bbob', 'meta');

cfg.cornn_raw_dir  = fullfile(cfg.root_dir, 'cornn', 'raw');
cfg.cornn_ela_dir  = fullfile(cfg.root_dir, 'cornn', 'ela');
cfg.cornn_ng_dir   = fullfile(cfg.root_dir, 'cornn', 'nevergrad');
cfg.cornn_adam_dir = fullfile(cfg.root_dir, 'cornn', 'adam');

cfg.isa_dir        = fullfile(cfg.root_dir, 'isa');

% Create directories if needed
dirs = {cfg.input_dir, cfg.bbob_raw_dir, cfg.bbob_ela_dir, cfg.bbob_ng_dir, ...
        cfg.bbob_adam_dir, cfg.bbob_meta_dir, cfg.cornn_raw_dir, ...
        cfg.cornn_ela_dir, cfg.cornn_ng_dir, cfg.cornn_adam_dir, cfg.isa_dir};
for ii = 1:length(dirs)
    if ~isfolder(dirs{ii})
        mkdir(dirs{ii});
    end
end

% =============================================================================
% EXPERIMENT CONSTANTS  (paper-authoritative values)
% =============================================================================

cfg.dimensions          = [41, 261, 481];
cfg.n_replicates        = 5;
cfg.sample_size         = 100;           % points per dimension

cfg.n_bbob_functions    = 24;
cfg.n_bbob_instances    = 15;

cfg.budget              = 5000;
cfg.n_runs              = 30;
cfg.bounds_lo           = -5.0;
cfg.bounds_hi           =  5.0;

cfg.algorithm_names     = {'Adam','CMA','PSO','RandomSearch','TwoPointsDE'};
cfg.algorithm_header    = [{'Function'}, cfg.algorithm_names];
cfg.n_algorithms        = length(cfg.algorithm_names);

cfg.target_min          = -1.0;
cfg.target_max          =  1.0;
cfg.target_step         =  0.1;
cfg.targets             = cfg.target_max:-cfg.target_step:cfg.target_min;
cfg.n_targets           = length(cfg.targets);   % 21

cfg.epsilon             = 0.0;           % acceptability threshold

end
