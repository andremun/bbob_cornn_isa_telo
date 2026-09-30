%SHARED_GENERATE_INSTANCE_SPACE Full ISA pipeline for BBOB-CORNN.
%   Dimensionality reduction (PCA + t-SNE), feature correlation analysis,
%   footprint estimation (TRACE), and figure/table generation for the
%   BBOB-CORNN instance space analysis.
%
%   Prerequisites:
%       - SHARED_CONSOLIDATE_RAW_DATA must have completed
%       - CORNN_CONFIG.M on the MATLAB path
%       - TRACE, KNNRegressor, tblvertcat, daviolinplot, scriptfcn (bundled)
%       - tsne_d, tsne_p, d2p (van der Maaten t-SNE toolbox, bundled)
%       - MATLAB Statistics and Machine Learning Toolbox (for TSNE)
%
%   Section flags (CFG.RUN.*, overridable via environment variables of the
%   same name -- see CORNN_CONFIG) allow individual sections to be run
%   independently, e.g. to regenerate one figure without recomputing the
%   projection:
%       setenv('RUN_PROJECTION', '0'); shared_generate_instance_space
%
%   All figures and tables are written to CFG.ISA_DIR. Every write prints
%   a confirmation; every skipped section prints why it was skipped.
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

%% SET THE ENVIRONMENT
% TRACE, scriptfcn, KNNRegressor, tblvertcat, and daviolinplot are bundled
% at the repository root -- no addpath needed. The tsne(...) calls below
% use MATLAB's built-in Statistics and Machine Learning Toolbox function;
% no external t-SNE toolbox is required.
scriptfcn;

cfg     = cornn_config();
isa_dir = cfg.isa_dir;

RUN_LOAD_DATA  = cfg.run.load_data;
RUN_PROJECTION = cfg.run.projection;
RUN_FEATURES   = cfg.run.features;
RUN_MODELS     = cfg.run.models;
RUN_FOOTPRINTS = cfg.run.footprints;
RUN_FIGURES    = cfg.run.figures;
RUN_TABLES     = cfg.run.tables;

fprintf('\n=== shared_generate_instance_space.m ===\n');
fprintf('Section flags: load_data=%d projection=%d features=%d models=%d footprints=%d figures=%d tables=%d\n', ...
        RUN_LOAD_DATA, RUN_PROJECTION, RUN_FEATURES, RUN_MODELS, RUN_FOOTPRINTS, RUN_FIGURES, RUN_TABLES);
fprintf('Output directory: %s\n', isa_dir);
if cfg.sample_mode
    fprintf(['[INFO] SAMPLE_MODE is active. The numbers below check that the ' ...
             'pipeline runs end to end. With so few instances, treat them as ' ...
             'a pipeline check, not as a valid instance space analysis.\n']);
end

%% LOAD AND JOIN ELA + AUC DATA
if RUN_LOAD_DATA
    fprintf('\n--- Loading and joining ELA + AUC data ---\n');

    pflacco_data = readtable(fullfile(isa_dir, 'BBOB_CORNN_pflacco.csv'));
    fprintf('[OK] Loaded %d rows from BBOB_CORNN_pflacco.csv\n', height(pflacco_data));

    auc_data = tblvertcat(readtable(fullfile(isa_dir, 'BBOB_area_under_the_curve.csv')), ...
                          readtable(fullfile(isa_dir, 'CORNN_area_under_the_curve.csv')));
    fprintf('[OK] Loaded %d rows from BBOB/CORNN_area_under_the_curve.csv\n', height(auc_data));

    original_patterns    = {'bbob_f0','F0','_i0','_i','_d0','_D0'};
    replacement_patterns = {'F','F','_I','_I','_D','_D'};
    for ii = 1:length(original_patterns)
        auc_data.Function = replace(auc_data.Function, original_patterns{ii}, ...
                                                        replacement_patterns{ii});
    end

    for dd = [41, 261, 481]
        pattern = ['_D' num2str(dd)];
        idx = contains(pflacco_data.xFunction, pattern);
        pflacco_data.xFunction(idx) = erase(pflacco_data.xFunction(idx), pattern);
        pflacco_data.xFunction(idx) = append(pflacco_data.xFunction(idx), pattern);
    end

    meta_data = join(pflacco_data, auc_data, 'LeftKeys', 'xFunction', ...
                                             'RightKeys', 'Function');

    idx = any(ismissing(meta_data));
    meta_data(:, idx) = [];

    out_path = fullfile(isa_dir, 'BBOB_CORNN_metadata.csv');
    writetable(meta_data, out_path);
    fprintf('[OK] Wrote %s (%d rows, %d columns)\n', out_path, height(meta_data), width(meta_data));
else
    require_vars({'meta_data'});
    fprintf('[SKIP] RUN_LOAD_DATA is false; using existing meta_data from workspace.\n');
end

%% GENERATE THE 2-D PROJECTION
if RUN_PROJECTION
    fprintf('\n--- Computing PCA + t-SNE projection ---\n');

    retained_variance = 99.5;
    % Column indices come from matching names, not a fixed numeric range.
    % The pflacco feature set can have a different number of columns,
    % since the missing-value filter above removes a varying number of
    % them. A fixed range does not adapt to this, and can misclassify a
    % feature column as an algorithm, or the reverse.
    performance_idx = find(ismember(meta_data.Properties.VariableNames, cfg.algorithm_names));
    feature_idx     = setdiff(2:size(meta_data, 2), performance_idx);

    feature_names    = meta_data.Properties.VariableNames(feature_idx);
    algorithm_names  = meta_data.Properties.VariableNames(performance_idx);
    epsilon          = cfg.epsilon;

    number_features    = length(feature_idx);
    number_algorithms  = length(performance_idx);
    number_instances   = size(meta_data, 1);

    [X, muX, sigmaX] = zscore(meta_data{:, feature_idx});
    [coeff, score, ~, ~, explained] = pca(X);
    number_components = find(cumsum(explained) > retained_variance, 1);
    rng('default');
    Z = tsne(score(:, 1:number_components));

    fprintf('[OK] Projection complete: %d PCA components retained (%.1f%% variance), Z is %dx2\n', ...
            number_components, retained_variance, size(Z, 1));
else
    require_vars({'X','Z','feature_idx','performance_idx','number_instances', ...
                  'number_algorithms','algorithm_names','feature_names','epsilon'});
    fprintf('[SKIP] RUN_PROJECTION is false; using existing X, Z from workspace.\n');
end

%% IDENTIFY STRONG FEATURES
if RUN_FEATURES
    fprintf('\n--- Identifying strong features ---\n');

    rng('default');
    D = 1 - abs(corr(X));
    % tsne_d, d2p, and tsne_p, bundled at the repo root, are Laurens van
    % der Maaten's original t-SNE toolbox. They are not MATLAB's
    % Statistics and Machine Learning Toolbox tsne() function.
    % tsne(D, 'Distance', 'precomputed') is not an equivalent substitute.
    % It is a different implementation, and some MATLAB releases do not
    % support 'precomputed' as a Distance value at all.
    Lambda = [];
    tsne_failed = false;
    try
        Lambda = tsne_d(D, [], 2, 10);
    catch e
        tsne_failed = true;
        fprintf('[WARN] tsne_d failed (%s)\n', e.message);
    end

    % The final feature selection follows the paper's own choice of four
    % named features. It is not a rule derivable purely from z1/z2
    % correlation. SAMPLE_MODE's small sample size can make several
    % features degenerate, with zero variance across so few, narrowly
    % chosen instances. corr(X) can then contain NaN values, and
    % clustering can leave no valid rows or too few features to match.
    % When this happens, fall back to a fixed placeholder selection.
    % RUN_MODELS and RUN_FOOTPRINTS below only need some valid feature
    % indices to run their own code paths, not a statistically
    % meaningful selection.
    fallback_failed = tsne_failed || isempty(Lambda) || any(isnan(Lambda(:)));
    if ~fallback_failed
        try
            rng('default');
            eva = evalclusters(Lambda, 'kmeans', 'gap', ...
                               'KList',        3:size(X, 2), ... % minimum of three features
                               'Distance',     'sqeuclidean', ...
                               'SearchMethod', 'firstMaxSE');
            clust = bsxfun(@eq, eva.OptimalY, 1:eva.OptimalK);
            features_per_cluster = max(sum(clust, 1));
            clustered_features   = cell(features_per_cluster, eva.OptimalK);
            for ii = 1:eva.OptimalK
                aux = feature_names(clust(:, ii));
                clustered_features(1:length(aux), ii) = aux;
            end
            disp(clustered_features);

            [rho, pval] = corr(X, Z);
            rho(isnan(rho) | (pval > 0.05) | abs(rho) < 0.1) = 0;
            strong = [];
            for ii = 1:2
                [aux, ind] = sort(abs(rho(:, ii) .* clust), 'descend');
                ind(aux == 0) = NaN;
                strong = [strong ind(1, :)]; %#ok<AGROW>
            end
            strong = unique(strong(~isnan(strong)));
            strong_backup = strong;

            % The final narrowing to 4 features follows the paper's own
            % choice, not a derivable rule. See the paper's discussion of
            % nbc_nn_nb_cor, ela_level_mmce_qda_10, nbc_nb_fitness_cor,
            % and disp_ratio_median_02. A position into `strong`, such as
            % strong([7 2 8 5]), only works when `strong`'s order matches
            % one specific run. `strong` is sorted by feature column
            % index, which depends on file-listing order feeding into
            % meta_data's column order. This order has no scientific
            % meaning. Select by name instead, which is robust to a
            % different column order.
            paper_selected_features = {'nbc_nn_nb_cor', 'ela_level_mmce_qda_10', ...
                                        'nbc_nb_fitness_cor', 'disp_ratio_median_02'};
            sel = find(ismember(feature_names(strong), paper_selected_features));
            if length(sel) == length(paper_selected_features)
                strong = strong(sel);
            else
                fprintf(['[WARN] Not all %d of the paper''s named strong features were ' ...
                         'found among this run''s cluster candidates -- keeping the ' ...
                         'first %d candidates instead.\n'], ...
                        length(paper_selected_features), min(4, length(strong)));
                strong = strong(1:min(4, length(strong)));
            end

            fprintf('[OK] Identified %d feature clusters; %d strong features selected for detailed analysis\n', ...
                    eva.OptimalK, length(strong));
        catch e
            fallback_failed = true;
            fprintf('[WARN] Feature clustering failed (%s)\n', e.message);
        end
    end

    if fallback_failed
        n_placeholder = min(4, number_features);
        fprintf(['[WARN] Feature-correlation clustering could not run (too few valid features/' ...
                 'instances, likely SAMPLE_MODE) -- using the first %d features as a placeholder ' ...
                 'selection instead\n'], n_placeholder);
        strong_backup = 1:n_placeholder;
        strong        = strong_backup;
    end
else
    require_vars({'strong','strong_backup'});
    fprintf('[SKIP] RUN_FEATURES is false; using existing strong, strong_backup from workspace.\n');
end

%% BUILD PREDICTION MODELS
if RUN_MODELS
    fprintf('\n--- Building KNN prediction models ---\n');

    [Y, muY, sigmaY] = zscore(meta_data{:, performance_idx}(:));
    Y    = reshape(Y, [number_instances number_algorithms]);
    Yhat = nan(size(Y));
    Ycv  = Yhat;

    models = cell(number_algorithms, 1);
    for ii = 1:number_algorithms
        models{ii}  = KNNRegressor();
        models{ii}  = models{ii}.fitCV(Z, Y(:, ii));
        Yhat(:, ii) = models{ii}.predict(Z) .* sigmaY + muY;
        Ycv(:, ii)  = models{ii}.cvResults.CVPredictions .* sigmaY + muY;
    end

    fprintf('[OK] Fitted %d cross-validated KNN models (one per algorithm)\n', number_algorithms);
else
    require_vars({'Y','Ycv'});
    fprintf('[SKIP] RUN_MODELS is false; using existing Y, Ycv from workspace.\n');
end

%% MEASURE THE FOOTPRINTS
if RUN_FOOTPRINTS
    fprintf('\n--- Estimating TRACE footprints ---\n');

    opts.usesim = true;          % Use the actual or simulated data to calculate the footprints
    opts.PI     = 0.6;           % Purity threshold
    opts.Trace2 = true;          % Use Trace2 instead of TRACE
    opts.prior  = [0.6, 0.4];    % Trace2 prior weighting
    trace_outputs = cell(1, 2);

    Ybin = meta_data{:, performance_idx} > epsilon; % using real values
    [p_performance, p_selection] = max(meta_data{:, performance_idx}, [], 2);
    p_selection(p_performance == 0) = 0;
    beta = mean(Ybin, 2) > 0.51;
    trace_outputs{1} = TRACE(Z, Ybin, p_selection, beta, algorithm_names, opts);

    Ybin_cv = Ycv > epsilon; % Using cross-validated simulation
    [o_performance, o_selection] = max(Ycv, [], 2);
    o_selection(o_performance <= 0) = 0;
    beta = mean(Ybin_cv, 2) > 0.51;
    trace_outputs{2} = TRACE(Z, Ybin_cv, o_selection, beta, algorithm_names, opts);

    fprintf('[OK] Computed footprints for %d algorithms (actual + cross-validated)\n', number_algorithms);
else
    require_vars({'trace_outputs','Ybin','p_selection','o_selection'});
    fprintf('[SKIP] RUN_FOOTPRINTS is false; using existing trace_outputs from workspace.\n');
end

%% DEFINE INSTANCE GROUPS  (shared by RUN_FIGURES and RUN_TABLES)
if RUN_FIGURES || RUN_TABLES
    fprintf('\n--- Defining instance groups ---\n');

    close all;
    isbo041 = contains(meta_data.xFunction, '_D41');
    isbo261 = contains(meta_data.xFunction, '_D261');
    isbo481 = contains(meta_data.xFunction, '_D481');
    istanh  = contains(meta_data.xFunction, 'tanh');
    isrelu  = contains(meta_data.xFunction, 'relu');
    isnet1  = contains(meta_data.xFunction, 'Net_1');
    isnet3  = contains(meta_data.xFunction, 'Net_3');
    isnet5  = contains(meta_data.xFunction, 'Net_5');
    istrn   = contains(meta_data.xFunction, 'train');
    istst   = contains(meta_data.xFunction, 'test');

    groups = [isbo041, isbo261, isbo481, ...
              isrelu & isnet1 & istrn, isrelu & isnet1 & istst, ...
              isrelu & isnet3 & istrn, isrelu & isnet3 & istst, ...
              isrelu & isnet5 & istrn, isrelu & isnet5 & istst, ...
              istanh & isnet1 & istrn, istanh & isnet1 & istst, ...
              istanh & isnet3 & istrn, istanh & isnet3 & istst, ...
              istanh & isnet5 & istrn, istanh & isnet5 & istst];

    group_names = {'BBOB_{D=41}', 'BBOB_{D=261}', 'BBOB_{D=481}', ...
                   'CORNN_{Net1, ReLU, Train}', 'CORNN_{Net1, ReLU, Test}', ...
                   'CORNN_{Net3, ReLU, Train}', 'CORNN_{Net3, ReLU, Test}', ...
                   'CORNN_{Net5, ReLU, Train}', 'CORNN_{Net5, ReLU, Test}', ...
                   'CORNN_{Net1, Tanh, Train}', 'CORNN_{Net1, Tanh, Test}', ...
                   'CORNN_{Net3, Tanh, Train}', 'CORNN_{Net3, Tanh, Test}', ...
                   'CORNN_{Net5, Tanh, Train}', 'CORNN_{Net5, Tanh, Test}'};
    number_groups       = size(groups, 2);
    group_by_dimension  = sum(groups .* (1:number_groups), 2);

    isfunc = false(number_instances, 24);
    for ii = 1:24
        isfunc(:, ii) = contains(meta_data.xFunction, ['F' num2str(ii) '_']);
    end
    group_by_function = sum([isfunc isrelu istanh] .* (1:26), 2);
    group_by_function = categorical(group_by_function);
    % renamecats requires every OLDNAME to already exist as a category.
    % SAMPLE_MODE samples only 1 CORNN architecture, so category 26
    % (Tanh) never occurs. Rename only whichever of 25 and 26 is
    % present. Full mode has both present, so this does not change its
    % behavior.
    old_arch_names = {'25','26'};
    new_arch_names = {'ReLU','Tanh'};
    present = ismember(old_arch_names, categories(group_by_function));
    group_by_function = renamecats(group_by_function, old_arch_names(present), new_arch_names(present));

    fprintf('[OK] Defined %d instance groups\n', number_groups);
else
    fprintf('[SKIP] RUN_FIGURES and RUN_TABLES are both false; skipping group definitions.\n');
end

%% PRODUCE THE FIGURES
if RUN_FIGURES
    fprintf('\n--- Generating figures ---\n');

    for ii = 1:length(strong)
        p = anovan(X(:, strong(ii)), group_by_dimension(:));
        disp(p);
    end

    % SOURCES GRAPH
    figure;
    gscatter(Z(:,1), Z(:,2), group_by_dimension, 'bbbkkkkkkrrrrrr', ...
                                                  '.sx..ssxx..ssxx', ...
                                                  [8 8 8 4 12 4 12 4 12 4 12 4 12 4 12]);
    legend(group_names, 'Location', 'northeastoutside');
    xlabel('z_{1}'); ylabel('z_{2}');
    axis square; grid
    set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12);
    out_path = fullfile(isa_dir, 'bbob_cornn_by_dimension.png');
    print(gcf, '-dpng', out_path);
    fprintf('[OK] Wrote %s\n', out_path);

    figure;
    gscatter(Z(:,1), Z(:,2), group_by_function, [], '.+^v');
    legend(categories(group_by_function), 'Location', 'northeastoutside');
    xlabel('z_{1}'); ylabel('z_{2}');
    axis square; grid
    set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12);
    out_path = fullfile(isa_dir, 'bbob_cornn_by_function.png');
    print(gcf, '-dpng', out_path);
    fprintf('[OK] Wrote %s\n', out_path);

    % FEATURE GRAPHS
    ela = meta_data{:, feature_idx};
    h2  = figure;
    cc  = sum([isbo041|isbo261|isbo481 isrelu istanh] .* (1:3), 2);
    for ii = 1:length(strong)
        figure;
        scatter(Z(:,1), Z(:,2), 8, ela(:, strong(ii)), 'filled');
        colorbar('EastOutside'); axis square; grid on;
        title(replace(feature_names(strong(ii)), "_", " "));
        xlabel('z_{1}'); ylabel('z_{2}');
        set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12);
        out_path = fullfile(isa_dir, ['tsne_' feature_names{strong(ii)} '.png']);
        print(gcf, '-dpng', out_path);
        fprintf('[OK] Wrote %s\n', out_path);

        figure(h2)
        subplot(1, 4, ii);
        daviolinplot(ela(:, strong(ii)), 'groups', cc, 'boxcolors', 'k', 'outliers', 0, ...
                                        'box', 0, 'boxwidth', 0.8, 'scatter', 2, ...
                                        'scattersize', 15, 'jitter', 1, 'scattercolors', 'same', ...
                                        'xtlabels', {'BBOB','ReLU','Tanh'});
        ylabel(replace(feature_names(strong(ii)), "_", " "));
        set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12);
    end
    out_path = fullfile(isa_dir, 'violin_strong_features.png');
    print(gcf, '-dpng', out_path);
    fprintf('[OK] Wrote %s\n', out_path);

    % FOOTPRINT GRAPHS
    for ii = 1:number_algorithms
        figure;
        drawScatter(Z, meta_data{:, performance_idx(ii)}, algorithm_names{ii});
        out_path = fullfile(isa_dir, ['distribution_performance_' algorithm_names{ii} '.png']);
        print(gcf, '-dpng', out_path);
        fprintf('[OK] Wrote %s\n', out_path);

        figure;
        drawGoodBadFootprint(Z, trace_outputs{1}.good{ii}, Ybin(:, ii), algorithm_names{ii});
        legend('Location', 'northeast');
        out_path = fullfile(isa_dir, ['footprint_' algorithm_names{ii} '_eps' num2str(epsilon) '.png']);
        print(gcf, '-dpng', out_path);
        fprintf('[OK] Wrote %s\n', out_path);
    end

    figure;
    drawPortfolioSelections(Z, p_selection, algorithm_names, 'Best algorithm');
    out_path = fullfile(isa_dir, 'distribution_svm_portfolio_actual.png');
    print(gcf, '-dpng', out_path);
    fprintf('[OK] Wrote %s\n', out_path);

    figure;
    drawPortfolioSelections(Z, o_selection, algorithm_names, 'Predicted best algorithm');
    out_path = fullfile(isa_dir, 'distribution_svm_portfolio_predicted.png');
    print(gcf, '-dpng', out_path);
    fprintf('[OK] Wrote %s\n', out_path);

    figure;
    drawPortfolioFootprint(Z, trace_outputs{1}.best, p_selection, algorithm_names);
    out_path = fullfile(isa_dir, 'footprint_portfolio.png');
    print(gcf, '-dpng', out_path);
    fprintf('[OK] Wrote %s\n', out_path);

    fprintf('[OK] Figure generation complete.\n');
else
    fprintf('[SKIP] RUN_FIGURES is false.\n');
end

%% PRODUCE THE SUMMARY TABLES
if RUN_TABLES
    fprintf('\n--- Computing summary tables ---\n');

    % trace_outputs{}.summary is a plain cell array. TRACE.m builds it
    % with cell(nalgos+1,11) and never converts it to a table.
    % writetable always rejects a cell array, so writecell is the
    % correct call here.
    summary = vertcat(trace_outputs{1}.summary, trace_outputs{2}.summary);
    try
        out_path = fullfile(isa_dir, 'footprint_summary.csv');
        writecell(summary, out_path);
        fprintf('[OK] Wrote %s\n', out_path);
    catch e
        fprintf('[WARN] Could not write footprint_summary.csv: %s\n', e.message);
    end

    finds_targets = nan(number_groups, number_algorithms);
    for ii = 1:number_algorithms
        finds_targets(:, ii) = sum(groups & Ybin(:, ii)) ./ sum(groups);
    end
    finds_targets = array2table(finds_targets, ...
                                'VariableNames', algorithm_names, ...
                                'RowNames', group_names);
    try
        out_path = fullfile(isa_dir, 'finds_targets.csv');
        writetable(finds_targets, out_path, 'WriteRowNames', true);
        fprintf('[OK] Wrote %s\n', out_path);
    catch e
        fprintf('[WARN] Could not write finds_targets.csv: %s\n', e.message);
    end

    rho = corr(X(:, strong_backup), [Z Y]);
    rho_features_axes = array2table(rho, 'RowNames', feature_names(strong_backup), ...
                                    'VariableNames', horzcat({'z_{1}','z_{2}'}, algorithm_names));
    try
        out_path = fullfile(isa_dir, 'rho_features_axes.csv');
        writetable(rho_features_axes, out_path, 'WriteRowNames', true);
        fprintf('[OK] Wrote %s\n', out_path);
    catch e
        fprintf('[WARN] Could not write rho_features_axes.csv: %s\n', e.message);
    end

    % REGRESSION FUNCTION COMPARISON (train/test distance) + its violin plot.
    % Kept together since the plot is a direct visualisation of these numbers.
    groups_cornn      = groups(:, 4:end);
    number_groups_cornn = size(groups_cornn, 2);
    number_functions   = sum(groups_cornn(:, 1));

    % train_test_dist takes its row count from group 1 (Net1/ReLU)'s
    % instance count. In full mode, every architecture and activation
    % group has the same count: 54 functions each. In SAMPLE_MODE, only
    % 1 of the 6 groups is populated, with 1 function and 1 architecture.
    % The other 5 groups are empty, so their subtraction has 0 rows, not
    % number_functions. Size each column to its own group pair's actual
    % row count. Unused rows stay at their preallocated NaN value.
    train_test_dist = nan(number_functions, 6);
    inc = 1;
    for ii = 1:2:number_groups_cornn
        train_vals = Z(groups_cornn(:,ii),:);
        test_vals  = Z(groups_cornn(:,ii+1),:);
        n_pairs = min(size(train_vals,1), size(test_vals,1));
        if n_pairs > 0
            train_test_dist(1:n_pairs, inc) = sqrt(sum((train_vals(1:n_pairs,:) - test_vals(1:n_pairs,:)).^2, 2));
        end
        inc = inc + 1;
    end

    try
        out_path = fullfile(isa_dir, 'train_test_distance.csv');
        writematrix(train_test_dist, out_path);
        fprintf('[OK] Wrote %s\n', out_path);
    catch e
        fprintf('[WARN] Could not write train_test_distance.csv: %s\n', e.message);
    end

    if RUN_FIGURES
        % A group with no matching CORNN architecture and activation
        % instances is entirely NaN. SAMPLE_MODE leaves 5 of the 6
        % groups this way. daviolinplot's ksdensity call rejects a group
        % with zero real data points. Drop these groups from the plot
        % instead of letting it crash. Full mode populates all 6 groups.
        valid_groups = ~all(isnan(train_test_dist), 1);
        if any(valid_groups)
            plot_data = train_test_dist(:, valid_groups);
            n_valid   = sum(valid_groups);
            idx = repmat(1:n_valid, [number_functions 1]);
            figure;
            daviolinplot(plot_data(:), 'groups', idx(:), ...
                                        'boxcolors', 'k', 'outliers', 0, ...
                                        'box', 0, 'boxwidth', 0.8, 'scatter', 2, ...
                                        'scattersize', 15, 'jitter', 1, 'scattercolors', 'same');
            ylabel('Distance between train and test');
            all_labels = replace(group_names(4:2:end), ", Train", "");
            xticklabels(all_labels(valid_groups))
            legend off; grid on;
            set(findall(gcf, '-property', 'FontSize'), 'FontSize', 12);
            set(gca, 'Position', [0.1300 0.1100 0.7750 0.6200]);
            out_path = fullfile(isa_dir, 'train_test_distance_by_groups.png');
            print(gcf, '-dpng', out_path);
            fprintf('[OK] Wrote %s\n', out_path);
        else
            fprintf('[WARN] No CORNN architecture/activation group has train/test data -- skipping train_test_distance_by_groups.png\n');
        end
    else
        fprintf('[SKIP] train_test_distance_by_groups.png requires RUN_FIGURES; numeric table still written.\n');
    end

    fprintf('[OK] Table generation complete.\n');
else
    fprintf('[SKIP] RUN_TABLES is false.\n');
end

fprintf('\n=== shared_generate_instance_space.m complete ===\n');


% =============================================================================
% LOCAL FUNCTIONS
% =============================================================================
function require_vars(names)
%REQUIRE_VARS Halt with a clear message if required workspace variables are missing.
%   REQUIRE_VARS(NAMES) checks that every variable name in the cell array
%   NAMES exists in the calling (script) workspace. If any are missing,
%   raises an error naming them and explaining how to fix it, rather than
%   letting execution continue into a much less informative crash several
%   sections later.

missing = names(~cellfun(@(v) evalin('caller', ['exist(''' v ''', ''var'')']) == 1, names));
if ~isempty(missing)
    error('shared_generate_instance_space:missingData', ...
        ['Required variable(s) not found in workspace: %s\n' ...
         'This section was skipped because its RUN_* flag is false, but a ' ...
         'required input from an earlier section is missing. Either enable ' ...
         'that section''s flag (see CONFIGURATION.md), or run it once ' ...
         'manually in this session before skipping it.'], ...
        strjoin(missing, ', '));
end
end
