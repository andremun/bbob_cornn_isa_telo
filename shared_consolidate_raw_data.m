% shared_consolidate_raw_data.m
%
% Reads raw per-run CSV files produced by the nevergrad and Adam runners,
% consolidates them into per-instance matrices, computes ERT-based AUC
% values for the instance space analysis, and processes pflacco ELA features.
%
% Prerequisites: bbob_run_nevergrad.py, cornn_run_nevergrad.py,
%                bbob_run_adam.py, cornn_run_adam.py must have completed.
% Outputs (written to cfg.isa_dir):
%   BBOB_area_under_the_curve.csv
%   CORNN_area_under_the_curve.csv
%   BBOB_CORNN_pflacco.csv
%
% See also: shared_generate_instance_space.m, cornn_config.m

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

%% CONFIGURATION
cfg = cornn_config();

bbob_ng_dir     = cfg.bbob_ng_dir;
bbob_adam_dir   = cfg.bbob_adam_dir;
bbob_consol_dir = fullfile(cfg.bbob_ng_dir,   'consol');
bbob_ela_dir    = cfg.bbob_ela_dir;

cornn_ng_dir     = cfg.cornn_ng_dir;
cornn_adam_dir   = cfg.cornn_adam_dir;
cornn_consol_dir = fullfile(cfg.cornn_ng_dir,  'consol');
cornn_ela_dir    = cfg.cornn_ela_dir;

isa_dir = cfg.isa_dir;

if ~isfolder(bbob_consol_dir),  mkdir(bbob_consol_dir);  end
if ~isfolder(cornn_consol_dir), mkdir(cornn_consol_dir); end

% Algorithm order: Adam first, then the four nevergrad algorithms.
% Must match the consolidation loops below.
algorithm_header  = cfg.algorithm_header;
algorithm_names   = cfg.algorithm_names;
targets           = cfg.targets;

number_algorithms = cfg.n_algorithms;
number_targets    = length(targets);
number_fevals     = cfg.budget;
number_runs       = cfg.n_runs;
max_fevals        = log10(number_runs * number_fevals);

targets4 = reshape(targets, [number_targets, 1, 1, 1]);
fevals4  = reshape(1:number_fevals, [1, number_fevals, 1, 1]);

fopt = table2array(readtable(fullfile(cfg.bbob_meta_dir, 'bbob_fopt.csv')));

%% CONSOLIDATE BBOB NEVERGRAD RAW DATA
filelist = struct2cell(dir(bbob_ng_dir));
filelist = filelist(1,3:end)';
ii = 1;
while ii < length(filelist)
    index_run = strfind(filelist{ii},'_R');
    if isempty(index_run), ii = ii+1; continue; end
    fcn = filelist{ii}(1:index_run(end)-1);
    current_file = fullfile(bbob_consol_dir, [fcn '.csv']);
    disp(fcn);
    index_files = find(contains(filelist, fcn));
    ii = index_files(end) + 1;
    if length(index_files) < number_runs
        disp([' -> ' fcn ' is incomplete']); continue;
    end
    if isfile(current_file), continue; end
    data_all_runs = zeros(number_fevals, number_runs);
    for jj = 1:length(index_files)
        data_single_run = readtable(fullfile(bbob_ng_dir, filelist{index_files(jj)}));
        if isempty(data_single_run), continue; end
        data_all_runs(:,jj) = data_single_run.Fval;
    end
    writetable(array2table(data_all_runs,'VariableNames',string(1:number_runs)), current_file);
end

%% CONSOLIDATE CORNN NEVERGRAD RAW DATA
filelist = struct2cell(dir(cornn_ng_dir));
filelist = filelist(1,3:end)';
ii = 1;
while ii < length(filelist)
    index_run = strfind(filelist{ii},'_R');
    if isempty(index_run), ii = ii+1; continue; end
    fcn = filelist{ii}(1:index_run(end)-1);
    current_train = fullfile(cornn_consol_dir, [fcn '_train.csv']);
    current_test  = fullfile(cornn_consol_dir, [fcn '_test.csv']);
    disp(fcn);
    index_files = find(contains(filelist, fcn));
    ii = index_files(end) + 1;
    if length(index_files) < number_runs
        disp([' -> ' fcn ' is incomplete']); continue;
    end
    if isfile(current_train) && isfile(current_test), continue; end
    data_train = zeros(number_fevals, number_runs);
    data_test  = zeros(number_fevals, number_runs);
    for jj = 1:length(index_files)
        t = readtable(fullfile(cornn_ng_dir, filelist{index_files(jj)}));
        if isempty(t), continue; end
        data_train(:,jj) = t.training_loss;
        data_test(:,jj)  = t.testing_loss;
    end
    writetable(array2table(data_train,'VariableNames',string(1:number_runs)), current_train);
    writetable(array2table(data_test, 'VariableNames',string(1:number_runs)), current_test);
end


%% CONSOLIDATE BBOB ADAM RAW DATA
% Adam raw files: <problem.id>_Adam_R<run>.csv  (column: Fval)
filelist_adam = struct2cell(dir(bbob_adam_dir));
filelist_adam = filelist_adam(1,3:end)';
ii = 1;
while ii <= length(filelist_adam)
    index_run = strfind(filelist_adam{ii}, '_Adam_R');
    if isempty(index_run), ii = ii+1; continue; end
    fcn          = filelist_adam{ii}(1:index_run(end)-1);
    current_file = fullfile(bbob_consol_dir, [fcn '_Adam.csv']);
    disp(fcn);
    index_files = find(contains(filelist_adam, [fcn '_Adam_R']));
    ii = index_files(end) + 1;
    if length(index_files) < number_runs
        disp([' -> ' fcn ' Adam is incomplete']); continue;
    end
    if isfile(current_file), continue; end
    data_all_runs = zeros(number_fevals, number_runs);
    for jj = 1:length(index_files)
        t = readtable(fullfile(bbob_adam_dir, filelist_adam{index_files(jj)}));
        if isempty(t), continue; end
        data_all_runs(:,jj) = t.Fval;
    end
    writetable(array2table(data_all_runs,'VariableNames',string(1:number_runs)), current_file);
    disp(['  -> Written: ' current_file]);
end

%% CONSOLIDATE CORNN ADAM RAW DATA
% Adam raw files: F_<fcn>_<arch>_Adam_R<run>.csv  (columns: training_loss, testing_loss)
filelist_adam = struct2cell(dir(cornn_adam_dir));
filelist_adam = filelist_adam(1,3:end)';
ii = 1;
while ii <= length(filelist_adam)
    index_run = strfind(filelist_adam{ii}, '_Adam_R');
    if isempty(index_run), ii = ii+1; continue; end
    fcn           = filelist_adam{ii}(1:index_run(end)-1);
    current_train = fullfile(cornn_consol_dir, [fcn '_Adam_train.csv']);
    current_test  = fullfile(cornn_consol_dir, [fcn '_Adam_test.csv']);
    disp(fcn);
    index_files = find(contains(filelist_adam, [fcn '_Adam_R']));
    ii = index_files(end) + 1;
    if length(index_files) < number_runs
        disp([' -> ' fcn ' Adam is incomplete']); continue;
    end
    if isfile(current_train) && isfile(current_test), continue; end
    data_train = zeros(number_fevals, number_runs);
    data_test  = zeros(number_fevals, number_runs);
    for jj = 1:length(index_files)
        t = readtable(fullfile(cornn_adam_dir, filelist_adam{index_files(jj)}));
        if isempty(t), continue; end
        data_train(:,jj) = t.training_loss;
        data_test(:,jj)  = t.testing_loss;
    end
    writetable(array2table(data_train,'VariableNames',string(1:number_runs)), current_train);
    writetable(array2table(data_test, 'VariableNames',string(1:number_runs)), current_test);
    disp(['  -> Written: ' current_train ' / ' current_test]);
end

%% PROCESSING BBOB PERFORMANCE TO OBTAIN THE AUC
filelist = struct2cell(dir(bbob_consol_dir));
filelist = filelist(1,3:end)';
number_files = length(filelist);
number_instances_bbob = number_files./number_algorithms;

expected_fevals_bbob = nan(number_targets,number_algorithms,number_instances_bbob);
area_under_the_curve_bbob = cell(number_instances_bbob,number_algorithms+1);
inc = 1;
ii = 1;
while ii<number_files
    index_run = strfind(filelist{ii},'_Adam');
    fcn = filelist{ii}(1:index_run(end)-1);
    area_under_the_curve_bbob{inc,1} = fcn;
    index_files = find(contains(filelist,fcn));
    if length(index_files)<number_algorithms
        disp([' -> ' fcn ' is incomplete']);
        continue
    end

    tokens = regexp(filelist{ii}, 'f(\d+)_i(\d+)_d(\d+)', 'tokens');
    tokens = tokens{1};
    id = fopt(:,1) == str2double(tokens{1}) & ...
         fopt(:,2) == str2double(tokens{3}) & ...
         fopt(:,3) == str2double(tokens{2});

    algo_files = {[fcn '_Adam.csv'],[fcn '_CMA.csv'],[fcn '_PSO.csv'],[fcn '_RandomSearch.csv'],[fcn '_TwoPointsDE.csv']};
    data_all_runs = zeros(number_fevals, number_runs, number_algorithms);
    for jj = 1:number_algorithms
        data_all_runs(:,:,jj) = readmatrix(fullfile(bbob_consol_dir, algo_files{jj}), 'NumHeaderLines',1);
    end
    data_all_runs = data_all_runs - fopt(id,4);

    logdata4 = reshape(log10(data_all_runs), [1, number_fevals, number_runs, number_algorithms]);
    cond = logdata4 <= targets4;
    idx_all = cond .* fevals4;  % [T × F × R × A]
    idx_all(~cond) = inf;
    runs_by_target = squeeze(min(idx_all, [], 2)); % [T × R × A]
    runs_by_target(isinf(runs_by_target)) = number_fevals;
    expected_fevals_bbob(:,:,inc) = squeeze(sum(runs_by_target,2)./sum(runs_by_target<number_fevals,2));
    
    for jj=1:number_algorithms
        unsolved = isinf(expected_fevals_bbob(:,jj,inc));
        if all(unsolved)
            area_under_the_curve_bbob{inc,jj+1} = 0;
        else
            [F,X] = ecdf(log10(expected_fevals_bbob(:,jj,inc)), 'Censoring', unsolved);
            X = [X' max_fevals];
            F = [F' max(F)];
            area_under_the_curve_bbob{inc,jj+1} = trapz(X,F)./max_fevals;
        end
    end
    inc = inc+1;
    ii = index_files(end) + 1;
end

area_under_the_curve_bbob = cell2table(area_under_the_curve_bbob, ...
                                       "VariableNames", algorithm_header);
writetable(area_under_the_curve_bbob, fullfile(isa_dir, 'BBOB_area_under_the_curve.csv'));

%% PROCESSING CORNN PERFORMANCE TO OBTAIN THE AUC
filelist = struct2cell(dir(cornn_consol_dir));
filelist = filelist(1,3:end)';
number_files = length(filelist);
number_instances_corrn = number_files ./ (number_algorithms * 2);

area_under_the_curve_cornn = cell(number_instances_corrn,number_algorithms+1);
expected_fevals_cornn = nan(number_targets,number_algorithms,number_instances_corrn);
inc = 1;
ii = 1;
while ii<number_files
    index_run = strfind(filelist{ii},'_Adam_train');
    if isempty(index_run), ii = ii+1; continue; end
    fcn = filelist{ii}(1:index_run(end)-1);
    algo_suffixes = {'Adam','CMA','PSO','RandomSearch','TwoPointsDE'};
    train_files = cellfun(@(a) [fcn '_' a '_train.csv'], algo_suffixes, 'UniformOutput', false);
    test_files  = cellfun(@(a) [fcn '_' a '_test.csv'],  algo_suffixes, 'UniformOutput', false);
    all_files   = [train_files, test_files];
    missing     = ~cellfun(@(f) isfile(fullfile(cornn_consol_dir,f)), all_files);
    if any(missing)
        disp([' -> ' fcn ' is incomplete']); ii = ii+1; continue;
    end
    index_files = find(contains(filelist, fcn));

    if length(index_files)<number_algorithms
        disp([' -> ' fcn ' is incomplete']);
        continue
    end
    
    for aa = 1:2
        if aa==1
            area_under_the_curve_cornn{inc,1} = [fcn '_test'];
        else
            area_under_the_curve_cornn{inc,1} = [fcn '_train'];
        end
        % Processing training or test data
        if aa == 1
            current_files = test_files;
        else
            current_files = train_files;
        end
        data_all_runs = zeros(number_fevals, number_runs, number_algorithms);
        for jj = 1:number_algorithms
            data_all_runs(:,:,jj) = readmatrix(fullfile(cornn_consol_dir, current_files{jj}), 'NumHeaderLines',1);
        end
        % this minimum is wrong? For CORNN is zero, for BBOB must be extracted from the function
        data_all_runs = data_all_runs - 0;
        
        logdata4 = reshape(log10(data_all_runs), [1, number_fevals, number_runs, number_algorithms]); % [1 × F × R × A]
        cond = logdata4 <= targets4;               % [T × F × R × A]
        idx_all = cond .* fevals4;  % [T × F × R × A]
        idx_all(~cond) = inf;
        runs_by_target = squeeze(min(idx_all, [], 2)); % [T × R × A]
        runs_by_target(isinf(runs_by_target)) = number_fevals;
        expected_fevals_cornn(:,:,inc) = squeeze(sum(runs_by_target,2)./sum(runs_by_target~=number_fevals,2));
        
        for jj=1:number_algorithms
            unsolved = isinf(expected_fevals_cornn(:,jj,inc));
            if all(unsolved)
                area_under_the_curve_cornn{inc,jj+1} = 0;
            else
                [F,X] = ecdf(log10(expected_fevals_cornn(:,jj,inc)), 'Censoring', unsolved);
                X = [X' max_fevals];
                F = [F' max(F)];
                area_under_the_curve_cornn{inc,jj+1} = trapz(X,F)./max_fevals;
            end
        end
        inc = inc+1;
    end
    ii = index_files(end) + 1;
end

area_under_the_curve_cornn = cell2table(area_under_the_curve_cornn, ...
                                        "VariableNames", algorithm_header);
writetable(area_under_the_curve_cornn, fullfile(isa_dir, 'CORNN_area_under_the_curve.csv'));
%%
close all;

expected_fevals = cat(3,expected_fevals_bbob,expected_fevals_cornn);

auc_data = tblvertcat(area_under_the_curve_bbob, area_under_the_curve_cornn);

original_patterns = {'bbob_f0','F0','_i0','_i','_d0','_D0'};
replacement_patterns = {'F','F','_I','_I','_D','_D'};
for ii=1:length(original_patterns)
    auc_data.Function = replace(auc_data.Function, original_patterns{ii}, ...
                                                   replacement_patterns{ii});
end

isbo = false(length(auc_data.Function),3);
isbo(:,1) = contains(auc_data.Function,'_D41') | contains(auc_data.Function,'Net_1');
isbo(:,2) = contains(auc_data.Function,'_D261') | contains(auc_data.Function,'Net_3');
isbo(:,3) = contains(auc_data.Function,'_D481') | contains(auc_data.Function,'Net_5');

figure;
for ii=1:3
    res = [mean(squeeze(all(expected_fevals(:,:,isbo(:,ii))==1,2)),2) ... 
           mean(squeeze(all(isinf(expected_fevals(:,:,isbo(:,ii))),2)),2)];
    subplot(3,1,ii)
    bar(targets, res);
    legend({'Trivial targets','Unsolvable targets'},'Location','northeastoutside');
    ylim([0 1])
end

figure;
for jj = 1:number_algorithms
    current_fevals = squeeze(expected_fevals(:,jj,:));
    [Fax,Xax] = ecdf(log10(current_fevals(:)), 'Censoring', isinf(current_fevals(:)));
    Xax = [Xax' max_fevals 6];
    Fax = [Fax' max(Fax) max(Fax)];
    line(Xax,Fax,'LineWidth',1.5);
end
legend(algorithm_names,"Location","northeast");
xlabel('log_{10}(F_{evals})'); ylabel('Probability reaching target');
axis([-1 6 0 1]); axis square; grid;
set(findall(gcf,'-property','FontSize'),'FontSize',12);
print(gcf, '-dpng', fullfile(isa_dir, 'ecdf_per_algorithm.png'));


%% PROCESS THE PFLACCO DATA

filelist = struct2cell(dir(bbob_ela_dir));
filelist = filelist(1,3:end);
pflacco_data = readtable(fullfile(bbob_ela_dir, filelist{1}), 'TreatAsMissing', 'nan');

warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
for ii=2:length(filelist)
    pflacco_data = tblvertcat(pflacco_data, readtable(fullfile(bbob_ela_dir, filelist{ii}), 'TreatAsMissing', 'nan'));
end

filelist = struct2cell(dir(cornn_ela_dir));
filelist = filelist(1,3:end);
for ii=1:length(filelist)
    pflacco_data = tblvertcat(pflacco_data, readtable(fullfile(cornn_ela_dir, filelist{ii}), 'TreatAsMissing', 'nan'));
end

warning('on', 'MATLAB:table:ModifiedAndSavedVarnames');

idx = contains(pflacco_data.Properties.VariableNames,'runtime') | ...
      contains(pflacco_data.Properties.VariableNames,'fun_evals') | ...
      contains(pflacco_data.Properties.VariableNames,'basic') | ...
      all(ismissing(pflacco_data),1);
pflacco_data(:,idx) = [];

for ii=1:5
    pflacco_data.xFunction = replace(pflacco_data.xFunction,['_R' num2str(ii)],'');
end
instance_list = unique(pflacco_data.xFunction);

inc = 1;
for ii=1:length(instance_list)
    idx = contains(pflacco_data.xFunction,instance_list{ii});
    if ii == 1
        pflacco_avg = varfun(@mean, pflacco_data(idx,2:end), "ErrorHandler", @errorFunc);
        pflacco_avg = addvars(pflacco_avg, {instance_list{ii}}, 'NewVariableNames', 'xFunction', 'Before','mean_ela_distr_skewness');
    else
        pflacco_avg(ii,2:end) = varfun(@mean, pflacco_data(idx,2:end), "ErrorHandler", @errorFunc);
        pflacco_avg(ii,"xFunction") = {instance_list{ii}};
    end
end
pflacco_avg = renamevars(pflacco_avg,pflacco_avg.Properties.VariableNames,pflacco_data.Properties.VariableNames);
pflacco_avg.xFunction = replace(pflacco_avg.xFunction,'_S100','');
pflacco_avg.xFunction = replace(pflacco_avg.xFunction,'ing_loss','');
writetable(pflacco_avg, fullfile(isa_dir, 'BBOB_CORNN_pflacco.csv'));


