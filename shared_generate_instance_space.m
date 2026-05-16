% shared_generate_instance_space.m
%
% Full ISA pipeline: dimensionality reduction (PCA + t-SNE), feature
% correlation analysis, footprint estimation (TRACE), and figure generation
% for the BBOB-CORNN instance space analysis.
%
% Prerequisites:
%   - shared_consolidate_raw_data.m must have completed
%   - cornn_config.m on the MATLAB path
%   - ISA Toolbox (TRACE, tSNE_matlab, MATILDA) on the MATLAB path
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

%% SET THE ENVIRONMENT AND GET THE DATA READY
addpath('.\tSNE_matlab\');
addpath('.\MATILDA\');
scriptfcn;

cfg     = cornn_config();
isa_dir = cfg.isa_dir;

pflacco_data = readtable(fullfile(isa_dir, 'BBOB_CORNN_pflacco.csv'));

auc_data = tblvertcat(readtable(fullfile(isa_dir, 'BBOB_area_under_the_curve.csv')), ...
                      readtable(fullfile(isa_dir, 'CORNN_area_under_the_curve.csv')));

original_patterns = {'bbob_f0','F0','_i0','_i','_d0','_D0'};
replacement_patterns = {'F','F','_I','_I','_D','_D'};
for ii=1:length(original_patterns)
    auc_data.Function = replace(auc_data.Function, original_patterns{ii}, ...
                                                   replacement_patterns{ii});
end

for dd = [41, 261, 481]
    pattern = ['_D' num2str(dd)];
    idx = contains(pflacco_data.xFunction, pattern);
    pflacco_data.xFunction(idx) = erase(pflacco_data.xFunction(idx), pattern);
    pflacco_data.xFunction(idx) = append(pflacco_data.xFunction(idx), pattern);
end

meta_data = join(pflacco_data,auc_data,"LeftKeys","xFunction",...
                                       "RightKeys","Function");

idx = any(ismissing(meta_data));
meta_data(:,idx) = [];
writetable(meta_data, fullfile(isa_dir, 'BBOB_CORNN_metadata.csv'));

%% GENERATE THE 2-D PROJECTION
retained_variance = 99.5;
feature_idx = 2:53;
performance_idx = 54:size(meta_data,2);

feature_names = meta_data.Properties.VariableNames(feature_idx);
algorithm_names = meta_data.Properties.VariableNames(performance_idx);
epsilon = cfg.epsilon;

number_features = length(feature_idx);
number_algorithms = length(performance_idx);
number_instances = size(meta_data,1);

[X,muX,sigmaX] = zscore(meta_data{:,feature_idx});
[coeff,score,~,~,explained] = pca(X);
number_components = find(cumsum(explained)>retained_variance,1);
rng('default');
Z = tsne(score(:,1:number_components));

%% IDENTIFY STRONG FEATURES
rng('default');
D = 1-abs(corr(X));
Lambda = tsne_d(D,[],2,10);

rng('default');
eva = evalclusters(Lambda, 'kmeans', 'gap', ...
                           'KList', 3:size(X,2), ... % minimum of three features
                           'Distance', 'sqeuclidean', ...
                           'SearchMethod','firstMaxSE');
clust = bsxfun(@eq, eva.OptimalY, 1:eva.OptimalK);
features_per_cluster = max(sum(clust,1));
clustered_features = cell(features_per_cluster, eva.OptimalK);
for ii=1:eva.OptimalK
    aux = feature_names(clust(:,ii));
    clustered_features(1:length(aux),ii) = aux;
end
disp(clustered_features);

[rho,pval] = corr(X, Z);
rho(isnan(rho) | (pval>0.05) | abs(rho)<0.1) = 0;
strong = [];
for ii=1:2
    [aux,ind] = sort(abs(rho(:,ii).*clust), 'descend');
    ind(aux==0) = NaN;
    strong = [strong ind(1,:)];
end
strong = unique(strong(~isnan(strong)));
strong_backup = strong;
strong = strong([7 2 8 5]); % 7 10 2 8 5

%% BUILD PREDICTION MODELS
[Y,muY,sigmaY] = zscore(meta_data{:,performance_idx}(:));
Y = reshape(Y,[number_instances number_algorithms]);
Yhat = nan(size(Y));
Ycv = Yhat;

models = cell(number_algorithms,1);
for ii=1:number_algorithms
    models{ii} = KNNRegressor();
    models{ii} = models{ii}.fitCV(Z, Y(:,ii));
    Yhat(:,ii) = models{ii}.predict(Z).*sigmaY + muY;
    Ycv(:,ii) = models{ii}.cvResults.CVPredictions.*sigmaY + muY;
end

%% MEASSURE THE FOOTPRINTS
opts.usesim = true;          % Use the actual or simulated data to calculate the footprints
opts.PI = 0.6;               % Purity threshold
opts.Trace2 = true;          % Use Trace2 instead of TRACE
opts.prior = [0.6,0.4];      % Trace2 Prior Weighting
trace_outputs = cell(1,2);

Ybin = meta_data{:,performance_idx}>epsilon; % using real values
[p_performance,p_selection] = max(meta_data{:,performance_idx},[],2);
p_selection(p_performance==0) = 0;
beta = mean(Ybin,2)>0.51;
trace_outputs{1} = TRACE(Z, Ybin, p_selection, beta, algorithm_names, opts);

Ybin_cv = Ycv>epsilon; % Using cross-validated simulation
[o_performance,o_selection] = max(Ycv,[],2);
o_selection(o_performance<=0) = 0;
beta = mean(Ybin_cv,2)>0.51;
trace_outputs{2} = TRACE(Z, Ybin_cv, o_selection, beta, algorithm_names, opts);

%% PRODUCE THE FIGURES AND TABLES

close all;
isbo041 = contains(meta_data.xFunction,'_D41');
isbo261 = contains(meta_data.xFunction,'_D261');
isbo481 = contains(meta_data.xFunction,'_D481');
istanh = contains(meta_data.xFunction,'tanh');
isrelu = contains(meta_data.xFunction,'relu');
isnet1 = contains(meta_data.xFunction,'Net_1');
isnet3 = contains(meta_data.xFunction,'Net_3');
isnet5 = contains(meta_data.xFunction,'Net_5');
istrn = contains(meta_data.xFunction,'train');
istst = contains(meta_data.xFunction,'test');

groups = [isbo041, isbo261, isbo481, ...
          isrelu & isnet1 & istrn, isrelu & isnet1 & istst, ...
          isrelu & isnet3 & istrn, isrelu & isnet3 & istst, ...
          isrelu & isnet5 & istrn, isrelu & isnet5 & istst, ...
          istanh & isnet1 & istrn, istanh & isnet1 & istst, ...
          istanh & isnet3 & istrn, istanh & isnet3 & istst, ...
          istanh & isnet5 & istrn, istanh & isnet5 & istst];

group_names = {'BBOB_{D=41}', 'BBOB_{D=261}', 'BBOB_{D=481}',...
               'CORNN_{Net1, ReLU, Train}','CORNN_{Net1, ReLU, Test}',...
               'CORNN_{Net3, ReLU, Train}','CORNN_{Net3, ReLU, Test}',...
               'CORNN_{Net5, ReLU, Train}','CORNN_{Net5, ReLU, Test}',...
               'CORNN_{Net1, Tanh, Train}','CORNN_{Net1, Tanh, Test}',...
               'CORNN_{Net3, Tanh, Train}','CORNN_{Net3, Tanh, Test}',...
               'CORNN_{Net5, Tanh, Train}','CORNN_{Net5, Tanh, Test}'};
number_groups = size(groups,2);
group_by_dimension = sum(groups.*(1:number_groups),2);


isfunc = false(number_instances,24);
for ii=1:24
    isfunc(:,ii) = contains(meta_data.xFunction,['F' num2str(ii) '_']);
end
group_by_function = sum([isfunc isrelu istanh].*(1:26),2);
group_by_function = categorical(group_by_function);
group_by_function = renamecats(group_by_function,{'25','26'},{'ReLU','Tanh'});

for ii=1:length(strong)
    p = anovan(X(:,strong(ii)),group_by_dimension(:));
    disp(p);
end

% SOURCES GRAPH
figure;
gscatter(Z(:,1),Z(:,2),group_by_dimension,'bbbkkkkkkrrrrrr',...
                                          '.sx..ssxx..ssxx',...
                                          [8 8 8 4 12 4 12 4 12 4 12 4 12 4 12]);
legend(group_names, 'Location','northeastoutside');
xlabel('z_{1}'); ylabel('z_{2}');
axis square; grid
set(findall(gcf,'-property','FontSize'),'FontSize',12);
print(gcf, '-dpng', 'bbob_cornn_by_dimension.png');

figure;
gscatter(Z(:,1),Z(:,2),group_by_function,[],'.+^v');
legend(categories(group_by_function), 'Location','northeastoutside');
xlabel('z_{1}'); ylabel('z_{2}');
axis square; grid
set(findall(gcf,'-property','FontSize'),'FontSize',12);
print(gcf, '-dpng', 'bbob_cornn_by_function.png');


% FEATURE GRAPHS
ela = meta_data{:,feature_idx};
h2 = figure;
cc = sum([isbo041|isbo261|isbo481 isrelu istanh].*(1:3),2);
for ii=1:length(strong)
    figure;
    scatter(Z(:,1),Z(:,2),8,ela(:,strong(ii)),'filled');
    colorbar('EastOutside'); axis square; grid on;
    title(replace(feature_names(strong(ii)),"_"," "));
    xlabel('z_{1}'); ylabel('z_{2}');
    set(findall(gcf,'-property','FontSize'),'FontSize',12);
    print(gcf,'-dpng',['tsne_' feature_names{strong(ii)} '.png']);

    figure(h2)
    subplot(1,4,ii);
    daviolinplot(ela(:,strong(ii)),'groups',cc,'boxcolors','k','outliers',0,...
                                 'box',0,'boxwidth',0.8,'scatter',2,...
                                 'scattersize',15,'jitter',1,'scattercolors','same',...
                                 'xtlabels',{'BBOB','ReLU','Tanh'});
    ylabel(replace(feature_names(strong(ii)),"_"," "));
    set(findall(gcf,'-property','FontSize'),'FontSize',12);
end
print(gcf,'-dpng','violin_strong_features.png');

% FOOTPRINT GRAPHS
for ii=1:number_algorithms
    figure;
    drawScatter(Z, meta_data{:,performance_idx(ii)}, algorithm_names{ii});
    print(gcf,'-dpng',['distribution_performance_' algorithm_names{ii} '.png']);

    figure;
    drawGoodBadFootprint(Z, trace_outputs{1}.good{ii}, Ybin(:,ii), algorithm_names{ii});
    legend("Location","northeast");
    print(gcf,'-dpng',['footprint_' algorithm_names{ii} '_eps' num2str(epsilon) '.png']);
end

figure;
drawPortfolioSelections(Z, p_selection, algorithm_names, 'Best algorithm');
print(gcf,'-dpng', 'distribution_svm_portfolio.png');

figure;
drawPortfolioSelections(Z, o_selection, algorithm_names, 'Predicted best algorithm');
print(gcf,'-dpng', 'distribution_svm_portfolio.png');

figure;
drawPortfolioFootprint(Z, trace_outputs{1}.best, p_selection, algorithm_names);
print(gcf,'-dpng','footprint_portfolio.png');

% SUMMARY TABLES
summary = vertcat(trace_outputs{1}.summary,trace_outputs{2}.summary);

finds_targets = nan(number_groups,number_algorithms);
for ii=1:number_algorithms
    finds_targets(:,ii) = sum(groups & Ybin(:,ii))./sum(groups);
end
finds_targets = array2table(finds_targets,...
                            "VariableNames",algorithm_names,...
                            "RowNames",group_names);

rho = corr(X(:,strong_backup),[Z Y]);
rho_features_axes = array2table(rho,'RowNames',feature_names(strong_backup),'VariableNames',horzcat({'z_{1}','z_{2}'},algorithm_names));


% REGRESSION FUNCTION COMPARISON
groups = groups(:,4:end);
number_groups = size(groups,2);
number_functions = sum(groups(:,1));

train_test_dist = nan(number_functions,6);
inc = 1;
for ii=1:2:number_groups
    train_test_dist(:,inc) = sqrt(sum((Z(groups(:,ii),:) - Z(groups(:,ii+1),:)).^2,2));
    inc = inc+1;
end

idx = repmat(1:6,[number_functions 1]);

figure;
daviolinplot(train_test_dist(:),'groups',idx(:),...
                                'boxcolors','k','outliers',0,...
                                'box',0,'boxwidth',0.8,'scatter',2,...
                                'scattersize',15,'jitter',1,'scattercolors','same');
ylabel('Distance between train and test');
xticklabels(replace(group_names(4:2:end),", Train",""))
legend off; grid on;
set(findall(gcf,'-property','FontSize'),'FontSize',12);
set(gca,"Position",[0.1300 0.1100 0.7750 0.6200]);
print(gcf, '-dpng', 'train_test_distance_by_groups.png');

% gg = [isbo041|isnet1, isbo261|isnet3, isbo481|isnet5];
% aux = zeros(number_instances,1);
% for ii=1:3
%     aux(gg(:,ii)) = zscore(X(gg(:,ii),strong(2)));
% end
% figure;
% scatter(Z(:,1),Z(:,2),8,aux,'filled');
% colorbar('EastOutside'); axis square; grid on;
% title(replace(feature_names(strong(2)),"_"," "));
% xlabel('z_{1}'); ylabel('z_{2}');
% set(findall(gcf,'-property','FontSize'),'FontSize',12);
% print(gcf,'-dpng',['tsne_' feature_names{strong(2)} '.png']);
