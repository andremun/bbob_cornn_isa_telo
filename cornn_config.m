function cfg = cornn_config()
%CORNN_CONFIG Central configuration for the BBOB-CORNN ISA project.
%   CFG = CORNN_CONFIG() returns a struct containing all project constants
%   and directory paths. All paths are derived from a single root
%   directory, which can be overridden via the environment variable
%   BBOB_CORNN_ROOT. This function mirrors cornn/config.py and is the
%   single source of truth for all MATLAB scripts. Missing directories
%   are created automatically, and a message is printed for each one.
%
%   CFG.RUN is a struct of boolean flags controlling which sections of
%   SHARED_GENERATE_INSTANCE_SPACE execute. Each is overridable via an
%   environment variable of the same name, e.g.:
%       setenv('RUN_FIGURES', '0'); cfg = cornn_config();
%
%   CFG.SAMPLE_MODE, when the environment variable SAMPLE_MODE is set to
%   '1', restricts CFG.DIMENSIONS, CFG.N_BBOB_FUNCTIONS,
%   CFG.N_BBOB_INSTANCES, and CFG.N_REPLICATES to a small subset for local
%   verification without a cluster.
%
%   See also SHARED_CONSOLIDATE_RAW_DATA, SHARED_GENERATE_INSTANCE_SPACE.

%   Copyright (c) 2026 Mario Andres Munoz Acosta
%   School of Computing and Information Systems
%   The University of Melbourne
%
%   This software is licensed under the PolyForm Noncommercial License 1.0.0.
%   You may use, copy, modify, and distribute this software for any
%   non-commercial purpose. Commercial use is prohibited.
%   Full license text: https://polyformproject.org/licenses/noncommercial/1.0.0

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

% Create directories if needed. Each creation is printed so a run's log
% always shows where output will land; a silent mkdir here would otherwise
% hide a misconfigured BBOB_CORNN_ROOT until the first write fails.
dirs = {cfg.input_dir, cfg.bbob_raw_dir, cfg.bbob_ela_dir, cfg.bbob_ng_dir, ...
        cfg.bbob_adam_dir, cfg.bbob_meta_dir, cfg.cornn_raw_dir, ...
        cfg.cornn_ela_dir, cfg.cornn_ng_dir, cfg.cornn_adam_dir, cfg.isa_dir};
for ii = 1:length(dirs)
    if ~isfolder(dirs{ii})
        mkdir(dirs{ii});
        fprintf('[INFO] Created directory: %s\n', dirs{ii});
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

% =============================================================================
% SAMPLE MODE  (local reproducibility and badge verification)
% =============================================================================
% Set environment variable SAMPLE_MODE=1 to restrict to a small subset.
% Usage in MATLAB:  setenv('SAMPLE_MODE','1'); cfg = cornn_config();
%
% Sample subset:
%   BBOB:  functions 1 and 8, instances 1-3, dimension 41, 1 replicate
%   CORNN: first function x first architecture, dimension 41, 1 replicate
%
% NOTE: this block must run AFTER the "EXPERIMENT CONSTANTS" block above,
% since it overrides fields that block sets to their full-scale defaults.

cfg.sample_mode = strcmp(getenv('SAMPLE_MODE'), '1');
if cfg.sample_mode
    cfg.dimensions        = 41;
    cfg.n_bbob_functions  = 2;     % functions 1 and 8 only
    cfg.n_bbob_instances  = 3;     % instances 1-3 only
    cfg.n_replicates      = 1;
    fprintf('[INFO] SAMPLE_MODE active: dimensions=41, functions=2, instances=3, replicates=1\n');
end

% Canonical BBOB function/instance ID lists (mirrors cornn/config.py).
% Any script that enumerates BBOB functions or instances manually should
% iterate over these, not over 1:cfg.n_bbob_functions, since the sample
% subset {1, 8} is not a contiguous range and a range-based loop would
% silently process the wrong functions in sample mode.
if cfg.sample_mode
    cfg.bbob_function_ids = [1, 8];
    cfg.bbob_instance_ids = [1, 2, 3];
else
    cfg.bbob_function_ids = 1:cfg.n_bbob_functions;
    cfg.bbob_instance_ids = 1:cfg.n_bbob_instances;
end

% =============================================================================
% SECTION FLAGS  (used by shared_generate_instance_space.m)
% =============================================================================
% Each flag controls one independently-executable section of the ISA
% pipeline and is overridable via an environment variable of the same
% name (e.g. setenv('RUN_FIGURES', '0') to skip figure generation).

cfg.run.load_data  = parse_bool_env('RUN_LOAD_DATA',  true);
cfg.run.projection = parse_bool_env('RUN_PROJECTION', true);
cfg.run.features   = parse_bool_env('RUN_FEATURES',   true);
cfg.run.models     = parse_bool_env('RUN_MODELS',     true);
cfg.run.footprints = parse_bool_env('RUN_FOOTPRINTS', true);
cfg.run.figures    = parse_bool_env('RUN_FIGURES',    true);
cfg.run.tables     = parse_bool_env('RUN_TABLES',     true);

end

% =============================================================================
% LOCAL FUNCTIONS
% =============================================================================
function tf = parse_bool_env(name, default)
%PARSE_BOOL_ENV Read a boolean environment variable with a default.
%   TF = PARSE_BOOL_ENV(NAME, DEFAULT) returns true if the environment
%   variable NAME is set to '1' or 'true' (case-insensitive); false if
%   set to '0' or 'false'; and DEFAULT if the variable is not set.

val = getenv(name);
if isempty(val)
    tf = default;
else
    tf = ismember(lower(val), {'1', 'true'});
end
end
