classdef KNNRegressor
    %KNNRegressor  Advanced K-Nearest Neighbours regressor
    %   Features:
    %     - Inverse-distance weighting
    %     - Cross-validation for K
    %     - Auto-bestK usage
    %     - Adaptive epsilon with caching
    %     - Multiple K prediction (optimized)
    %     - Reusable KD-tree
    %     - CV plotting with error bars and annotation
    %     - Cross-validated predictions and diagnostics
    %     - Optimized cross-validation (pre-built fold models)
    %
    %   Example:
    %       X_train = rand(500,2);
    %       y_train = sin(sum(X_train,2)*3) + 0.05*randn(500,1);
    %
    %       mdl = KNNRegressor();
    %       mdl = mdl.fitCV(X_train, y_train, [], 'CV', 5, 'Loss','RMSE');
    %       y_pred = mdl.predict(rand(10,2));  % Uses bestK automatically
    %       mdl.plotCVErrors();  % Uses stored CV results
    %       mdl.plotCVDiagnostics();  % Diagnostic plots
    
    properties (Access = private)
        Searcher        % KDTreeSearcher object
        y_train         % Training target values
        nTrain          % Number of training samples
        nFeatures       % Number of features
        X_train         % Stored training features for adaptive epsilon
        cachedEpsilon   % Cached epsilon value for efficiency
    end

    properties
        bestK      % Best K found by CV (if fitCV used)
        isTrained  % Boolean flag indicating if model is trained
        cvResults  % Cross-validation results (if fitCV used)
    end

    methods
        function obj = KNNRegressor()
            %KNNREGRESSOR Constructor
            obj.isTrained = false;
            obj.cvResults = [];
            obj.cachedEpsilon = [];
        end

        function obj = fit(obj, X_train, y_train)
            %FIT  Train KNN by building KD-tree
            if nargin < 3
                error('KNNRegressor:NotEnoughInputs', 'X_train and y_train required.');
            end
            % Validate inputs
            if ~isnumeric(X_train) || ~ismatrix(X_train)
                error('KNNRegressor:InvalidInput', 'X_train must be numeric matrix.');
            end
            if ~isnumeric(y_train) || ~isvector(y_train)
                error('KNNRegressor:InvalidInput', 'y_train must be numeric vector.');
            end
            if size(X_train,1) ~= numel(y_train)
                error('KNNRegressor:DimensionMismatch', 'Rows of X_train must match length of y_train.');
            end
            if size(X_train,1) < 1
                error('KNNRegressor:EmptyData', 'Training data cannot be empty.');
            end

            obj.y_train = y_train(:); % ensure column
            obj.nTrain = size(X_train,1);
            obj.nFeatures = size(X_train,2);
            obj.X_train = X_train;
            obj.Searcher = KDTreeSearcher(X_train);
            obj.isTrained = true;
            
            % Pre-calculate cached epsilon for efficiency
            obj.cachedEpsilon = 1e-9 * mean(range(X_train,1));
            if obj.cachedEpsilon == 0  % Handle constant features
                obj.cachedEpsilon = 1e-9;
            end
        end

        function y_pred = predict(obj, X_test, K, epsilon)
            %PREDICT  Vectorized weighted KNN prediction (optimized for multiple K)
            %
            %   y_pred = predict(obj, X_test) uses obj.bestK if K not provided
            %   Supports multiple K values and returns matrix (rows=test, cols=K)
            
            if nargin < 3 || isempty(K)
                if isempty(obj.bestK)
                    error('KNNRegressor:NoK', 'K not provided and bestK not set. Run fitCV or provide K.');
                end
                K = obj.bestK;
            end
            if nargin < 4 || isempty(epsilon)
                % Use cached epsilon for efficiency
                epsilon = obj.cachedEpsilon;
            end

            if ~obj.isTrained
                error('KNNRegressor:UntrainedModel', 'Call fit before predict.');
            end
            if ~isnumeric(X_test) || ~ismatrix(X_test)
                error('KNNRegressor:InvalidInput', 'X_test must be numeric matrix.');
            end
            if size(X_test,2) ~= obj.nFeatures
                error('KNNRegressor:FeatureMismatch', 'X_test must have %d columns.', obj.nFeatures);
            end
            if ~all(K >= 1 & K <= obj.nTrain & K == round(K))
                error('KNNRegressor:InvalidK', 'All K values must be integers between 1 and %d.', obj.nTrain);
            end
            if ~isscalar(epsilon) || epsilon <= 0
                error('KNNRegressor:InvalidEpsilon', 'Epsilon must be positive scalar.');
            end

            % Ensure K is a row vector
            K = K(:)';

            % Warning for large K values (only once per unique large K)
            if ~isempty(obj.bestK)
                largeK = K(K > 2*obj.bestK);
                if ~isempty(largeK)
                    uniqueLargeK = unique(largeK);
                    if isscalar(uniqueLargeK)
                        warning('KNNRegressor:LargeK', ...
                                'Using K=%d, much larger than CV-selected bestK=%d', uniqueLargeK, obj.bestK);
                    elseif length(uniqueLargeK) > 1
                        warning('KNNRegressor:LargeK', ...
                                'Using K values [%s], much larger than CV-selected bestK=%d', ...
                                num2str(uniqueLargeK), obj.bestK);
                    end
                end
            end

            y_pred = zeros(size(X_test,1), numel(K));
            
            % OPTIMIZED: Get maximum K to minimize knnsearch calls
            maxK = max(K);
            [idx, dist] = knnsearch(obj.Searcher, X_test, 'K', maxK);
            
            for ki = 1:numel(K)
                kVal = K(ki);
                
                % Use subset of distances/indices for this K
                idx_k = idx(:, 1:kVal);
                dist_k = dist(:, 1:kVal);

                w = 1 ./ (dist_k + epsilon);
                w = w ./ sum(w,2);
                y_pred(:,ki) = sum(w .* obj.y_train(idx_k),2);

                exactMask = dist_k(:,1) == 0;
                if any(exactMask)
                    y_pred(exactMask,ki) = obj.y_train(idx_k(exactMask,1));
                end
            end
        end

        function [obj, bestK, cvErrors] = fitCV(obj, X_train, y_train, K_values, varargin)
            %FITCV  Train KNN with cross-validation to select best K (optimized)
            p = inputParser;
            addParameter(p,'CV',5,@(x)isscalar(x)&&x>1&&x==round(x));
            addParameter(p,'Epsilon',[],@(x)isempty(x) || (isscalar(x)&&x>0));
            addParameter(p,'Loss','MSE'); 
            addParameter(p,'Verbose',false,@islogical);
            parse(p,varargin{:});
            nFolds = p.Results.CV;
            epsilon = p.Results.Epsilon;
            lossSpec = p.Results.Loss;
            verbose = p.Results.Verbose;

            n_train = size(X_train,1);
            if n_train < nFolds
                error('KNNRegressor:InsufficientData', 'Need at least %d samples for %d-fold CV.', nFolds, nFolds);
            end

            if nargin < 4 || isempty(K_values)
                K_values = KNNRegressor.generateKRange(n_train);
            end
            if ~isnumeric(K_values) || any(K_values < 1) || any(K_values > n_train) || any(K_values ~= round(K_values))
                error('KNNRegressor:InvalidKValues','K_values must be integers between 1 and %d.', n_train);
            end

            % Select loss function
            isR2 = false;
            if isa(lossSpec,'function_handle')
                lossFun = lossSpec;
            else
                switch upper(lossSpec)
                    case 'MSE'; lossFun=@(y,yhat) mean((y-yhat).^2);
                    case 'RMSE'; lossFun=@(y,yhat) sqrt(mean((y-yhat).^2));
                    case 'MAE'; lossFun=@(y,yhat) mean(abs(y-yhat));
                    case 'MEDAE'; lossFun=@(y,yhat) median(abs(y-yhat));
                    case 'R2'; lossFun=@(y,yhat) 1 - sum((y-yhat).^2)/sum((y-mean(y)).^2); isR2=true;
                    otherwise; error('KNNRegressor:InvalidLoss','Unknown loss: %s',lossSpec);
                end
            end

            cvp = cvpartition(n_train,'KFold',nFolds);
            meanErrs = zeros(numel(K_values),1);
            stdErrs = zeros(numel(K_values),1);

            % OPTIMIZATION: Pre-build fold models for efficiency
            foldModels = cell(nFolds, 1);
            foldEpsilons = zeros(nFolds, 1);
            
            if verbose, fprintf('Pre-building fold models...\n'); end
            for f = 1:nFolds
                trainIdx = training(cvp, f);
                foldModels{f} = KNNRegressor();
                foldModels{f} = foldModels{f}.fit(X_train(trainIdx,:), y_train(trainIdx));
                
                if isempty(epsilon)
                    foldEpsilons(f) = 1e-9 * mean(range(X_train(trainIdx,:),1));
                    if foldEpsilons(f) == 0
                        foldEpsilons(f) = 1e-9;
                    end
                else
                    foldEpsilons(f) = epsilon;
                end
            end

            if verbose
                fprintf('Running %d-fold CV for %d K values...\n', nFolds, numel(K_values));
            end

            for ki = 1:numel(K_values)
                kVal = K_values(ki);
                foldErrs = zeros(nFolds,1);

                for f = 1:nFolds
                    testIdx = test(cvp, f);
                    yhat = foldModels{f}.predict(X_train(testIdx,:), kVal, foldEpsilons(f));
                    foldErrs(f) = lossFun(y_train(testIdx), yhat);
                end

                meanErrs(ki) = mean(foldErrs);
                stdErrs(ki) = std(foldErrs);

                if verbose
                    fprintf('K=%d: %.4f ± %.4f\n', kVal, meanErrs(ki), stdErrs(ki));
                end
            end

            % Best K selection
            if isR2
                [~,bestIdx] = max(meanErrs);
            else
                [~,bestIdx] = min(meanErrs);
            end
            bestK = K_values(bestIdx);
            if verbose, fprintf('Best K: %d\n', bestK); end

            % OPTIMIZATION: Generate cross-validated predictions for best K (reuse fold models)
            cvPredictions = nan(n_train, 1);
            if verbose, fprintf('Generating CV predictions for best K...\n'); end
            
            for f = 1:nFolds
                testIdx = test(cvp, f);
                cvPredictions(testIdx) = foldModels{f}.predict(X_train(testIdx,:), bestK, foldEpsilons(f));
            end

            % Fit full model
            obj = obj.fit(X_train, y_train);
            obj.bestK = bestK;

            % Return CV info
            cvErrors.K_values = K_values;
            cvErrors.Errors = meanErrs;
            cvErrors.StdErrors = stdErrs;
            cvErrors.BestK = bestK;
            cvErrors.Loss = lossSpec;
            cvErrors.CVPredictions = cvPredictions;
            cvErrors.CVResiduals = y_train - cvPredictions;
            
            % Store CV results in the object
            obj.cvResults = cvErrors;
        end

        function [cvPred, cvResid] = getCVPredictions(obj)
            %GETCVPREDICTIONS  Get cross-validated predictions and residuals
            %   [cvPred, cvResid] = getCVPredictions(obj)
            %   Returns cross-validated predictions and residuals using bestK
            
            if isempty(obj.cvResults)
                error('KNNRegressor:NoCVResults', ...
                      'No CV results available. Run fitCV first.');
            end
            
            if ~isfield(obj.cvResults, 'CVPredictions')
                error('KNNRegressor:NoCVPredictions', ...
                      'CV predictions not available. This may be an older CV result.');
            end
            
            cvPred = obj.cvResults.CVPredictions;
            cvResid = obj.cvResults.CVResiduals;
        end

        function plotCVDiagnostics(obj, varargin)
            %PLOTCVDIAGNOSTICS  Plot diagnostic plots for cross-validated predictions
            %   plotCVDiagnostics(obj) creates diagnostic plots
            
            p = inputParser;
            addParameter(p, 'ShowResiduals', true, @islogical);
            addParameter(p, 'ShowPredVsActual', true, @islogical);
            parse(p, varargin{:});
            
            [cvPred, cvResid] = obj.getCVPredictions();
            y_actual = obj.y_train;
            
            if p.Results.ShowPredVsActual && p.Results.ShowResiduals
                figure;
                subplot(1,2,1);
            elseif p.Results.ShowPredVsActual || p.Results.ShowResiduals
                figure;
            end
            
            % Predicted vs Actual plot
            if p.Results.ShowPredVsActual
                scatter(y_actual, cvPred, 50, 'filled', 'Alpha', 0.6);
                hold on;
                
                % Perfect prediction line
                minVal = min([y_actual; cvPred]);
                maxVal = max([y_actual; cvPred]);
                plot([minVal, maxVal], [minVal, maxVal], 'r--', 'LineWidth', 2);
                
                xlabel('Actual Values');
                ylabel('Cross-Validated Predictions');
                title(sprintf('CV Predictions vs Actual (K=%d)', obj.bestK));
                grid on;
                axis equal;
                legend('Data', 'Perfect Fit', 'Location', 'best');
                hold off;
            end
            
            % Residuals plot
            if p.Results.ShowResiduals
                if p.Results.ShowPredVsActual
                    subplot(1,2,2);
                end
                
                scatter(cvPred, cvResid, 50, 'filled', 'Alpha', 0.6);
                hold on;
                yline(0, 'r--', 'LineWidth', 2);
                
                xlabel('Cross-Validated Predictions');
                ylabel('Residuals');
                title(sprintf('CV Residuals vs Predictions (K=%d)', obj.bestK));
                grid on;
                hold off;
            end
        end

        function plotCVErrors(obj, cvErrors, varargin)
            %PLOTCVERRORS  Plot CV error vs K with optional error bars
            %   plotCVErrors(obj) - uses stored CV results from fitCV
            %   plotCVErrors(obj, cvErrors) - uses provided CV results
            
            % If no cvErrors provided, use stored results
            if nargin < 2 || isempty(cvErrors)
                if isempty(obj.cvResults)
                    error('KNNRegressor:NoCVResults', ...
                          'No CV results available. Run fitCV first or provide cvErrors.');
                end
                cvErrors = obj.cvResults;
            end
            
            p = inputParser;
            addParameter(p,'Title','CV Error vs K',@(x) ischar(x)||isstring(x));
            addParameter(p,'ShowErrorBars',true,@islogical);
            parse(p,varargin{:});
            plotTitle = char(p.Results.Title);
            showErrorBars = p.Results.ShowErrorBars;

            if ~isstruct(cvErrors) || ~isfield(cvErrors,'K_values') || ~isfield(cvErrors,'Errors')
                error('KNNRegressor:InvalidCVErrors','cvErrors must have K_values and Errors');
            end
            if length(cvErrors.K_values) ~= length(cvErrors.Errors)
                error('KNNRegressor:InvalidCVErrors','K_values and Errors must match in length');
            end

            figure;
            legendEntries = {};
            if showErrorBars && isfield(cvErrors,'StdErrors') && ...
               length(cvErrors.StdErrors)==length(cvErrors.Errors)
                errorbar(cvErrors.K_values, cvErrors.Errors, cvErrors.StdErrors,'-o','LineWidth',1.5,'MarkerSize',6);
                legendEntries{end+1} = 'CV Error';
            else
                plot(cvErrors.K_values, cvErrors.Errors,'-o','LineWidth',1.5,'MarkerSize',6);
                legendEntries{end+1} = 'CV Error';
            end
            
            xlabel('K (Number of Neighbors)');
            if isfield(cvErrors, 'Loss')
                ylabel(sprintf('Cross-Validation %s', cvErrors.Loss));
            else
                ylabel('Cross-Validation Error');
            end
            title(plotTitle);
            grid on;

            % Highlight best K
            if isfield(cvErrors,'BestK')
                bestIdx = find(cvErrors.K_values==cvErrors.BestK,1);
                hold on;
                if ~isempty(bestIdx)
                    plot(cvErrors.BestK, cvErrors.Errors(bestIdx),'ro','MarkerSize',10,'LineWidth',2);
                    legendEntries{end+1} = sprintf('Best K=%d', cvErrors.BestK);
                    legend(legendEntries, 'Location', 'best');
                else
                    % Annotation if best K not in range
                    text(0.02,0.98,sprintf('Best K=%d (not shown)',cvErrors.BestK),...
                         'Units','normalized','VerticalAlignment','top',...
                         'BackgroundColor','white','EdgeColor','black');
                end
                hold off;
            end
        end

        function disp(obj)
            %DISP Custom display method
            if obj.isTrained
                fprintf('KNNRegressor: %d samples, %d features\n', obj.nTrain, obj.nFeatures);
                if ~isempty(obj.bestK)
                    fprintf('Best K (from CV): %d\n', obj.bestK);
                end
                if ~isempty(obj.cvResults)
                    fprintf('CV results available:\n');
                    fprintf('  - Use plotCVErrors() to visualize K selection\n');
                    if isfield(obj.cvResults, 'CVPredictions')
                        fprintf('  - Use getCVPredictions() to get CV predictions\n');
                        fprintf('  - Use plotCVDiagnostics() for diagnostic plots\n');
                    end
                end
            else
                fprintf('KNNRegressor: Untrained\n');
            end
        end
    end

    methods (Static)
        function K_values = generateKRange(n_train)
            %GENERATEKRANGE  Reasonable K candidates
            %   Prefers odd K values for consistency with KNN best practices
            
            if n_train < 10
                % For small datasets, still test most values but prefer odd K
                if n_train <= 5
                    % Very small: test all K values
                    K_values = 1:n_train;
                else
                    % Small but not tiny: favor odd K, include a few even K
                    oddK = 1:2:n_train;  % All odd K values
                    evenK = 2:2:min(6, n_train);  % Just a few even K (2,4,6)
                    K_values = unique([oddK evenK]);
                end
                return;
            end

            smallK = [1, 3:2:min(11,n_train)];
            sqrtN = round(sqrt(n_train));
            midK = round(linspace(max(5,smallK(end)+1), min(sqrtN,n_train), 5));
            maxLargeK = max(ceil(0.2*n_train), midK(end)+5);
            if maxLargeK<=n_train && midK(end)<n_train
                largeK = round(linspace(midK(end)+1, min(maxLargeK,n_train), min(5,n_train-midK(end))));
            else
                largeK = [];
            end
            K_values = unique([smallK midK largeK]);
            K_values = K_values(K_values>=1 & K_values<=n_train);
        end
    end
end