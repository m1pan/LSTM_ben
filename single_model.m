%% Battery degradation predictions using a GRU-LSTM neural net
% Modified for single generalised model training with curve visualisation
% Original: Benjamin Warmington
% Modified for benchmarking: [Your name]

close all;

% ====== CONFIGURATION ======
% Select which batteries to hold out for testing
test_battery_indices = [3];%[3,8,11,16];%[2, 9, 16];  % Example: test on batteries 2, 9, 16

% Target features to use
target_Feature = [1:7, 12:21];
desired_features = target_Feature;

% Hyperparameters
window_size = 7;
ensemble_data_share = 0.6;
numSamples = 10;  % Ensemble size

% Neural network architecture
gru1_cell_num = 224;
DOL1 = 0.1;
lstm_cell_num = 200;
DOL2 = 0.1;
gru2_cell_num = 192;
DOL3 = 0.1;
epochs = 900;

% LKP array for different forecast starting points
LKP_array = [7, 10, 13, 16, 19];

% Plotting options
plot_curves = true;  % Set to false to skip curve plots
save_figures = true; % Set to true to save figures as PNG

% ====== PREPARE TRAINING DATA (NO BIASING) ======
Train_data = raw_data;

% Remove test batteries and their bootstrapped copies
indices_to_remove = sort([test_battery_indices, test_battery_indices + 16], 'descend');
for idx = indices_to_remove
    if idx <= length(Train_data)
        Train_data(idx) = [];
    end
end

% remove first battery
Train_data(1) = [];

num_train_batteries = length(Train_data);
fprintf('Training on %d batteries (including bootstrapped copies)\n', num_train_batteries);

% ====== NORMALISATION ======
num_features = length(Train_data(1).Features(1, desired_features));
Tot_Feature_Lists = [];

for b = 1:num_train_batteries
    Tot_Feature_Lists = [Tot_Feature_Lists; Train_data(b).Features(:, desired_features)];
end

[~, mu, sigma] = zscore(Tot_Feature_Lists);

% ====== BUILD TRAINING WINDOWS (NO BIASING) ======
X_cell = {};
Y_cell = {};
projection = 1;

for b = 1:num_train_batteries
    num_rpts = length(Train_data(b).Features(:, 1));
    max_window_start = num_rpts - (window_size + projection - 1);
    
    if max_window_start > 0
        for w_start = 1:max_window_start
            w_end = w_start + window_size - 1;
            
            X_Data_stan = (Train_data(b).Features(w_start:w_end, desired_features) - mu) ./ sigma;
            Y_Data_stan = (Train_data(b).Features(w_start+projection:w_end+projection, desired_features) - mu) ./ sigma;
            
            X_cell{end+1, 1} = X_Data_stan;
            Y_cell{end+1, 1} = Y_Data_stan;
        end
    end
end

fprintf('Total training windows: %d\n', length(X_cell));

% ====== TRAIN SINGLE ENSEMBLE (ONE TIME) ======
numChannels = size(X_cell{1}, 2);

fprintf('Training ensemble of %d networks...\n', numSamples);
net = {};

for i = 1:numSamples
    fprintf('  Training network %d/%d\n', i, numSamples);
    
    % Random subset for this ensemble member
    randArray = rand(length(X_cell), 1);
    [~, idx] = sort(randArray);
    rand_select = idx(1:ceil(length(idx) * ensemble_data_share));
    XTrain = X_cell(rand_select);
    TTrain = Y_cell(rand_select);
    
    layers = [
        sequenceInputLayer(numChannels, Normalization="none")
        gruLayer(gru1_cell_num)
        dropoutLayer(DOL1)
        lstmLayer(lstm_cell_num)
        dropoutLayer(DOL2)
        gruLayer(gru2_cell_num)
        dropoutLayer(DOL3)
        fullyConnectedLayer(numChannels)];
    
    options = trainingOptions("adam", ...
        MaxEpochs=epochs, ...
        SequencePaddingDirection="left", ...
        Shuffle="every-epoch", ...
        Verbose=false);
    
    net{end+1} = trainnet(XTrain, TTrain, layers, "mse", options);
end

fprintf('Ensemble training complete.\n\n');

% ====== BENCHMARK ON ALL TEST BATTERIES ======
% Store results
results = struct();
results.battery_id = [];
results.LKP = [];
results.MAPE = [];
results.RMSE = [];
results.R_squared = [];
results.cross_corr = [];

% Store predictions for plotting
predictions = struct();
predictions.battery_id = [];
predictions.LKP = [];
predictions.ground_truth = {};
predictions.mean_pred = {};
predictions.std_pred = {};
predictions.conf_upper = {};
predictions.conf_lower = {};

% Battery metadata for plot titles
Temperature = [10, 10, 10, 25, 25, 40, 40, 40, 10, 10, 10, 25, 25, 40, 40, 40];
Cycle_Type_Names = {'Drive Cycle', 'Drive Cycle', 'Drive Cycle', 'Drive Cycle', ...
                    'Drive Cycle', 'Drive Cycle', 'Drive Cycle', 'Drive Cycle', ...
                    'Constant Load', 'Constant Load', 'Constant Load', 'Constant Load', ...
                    'Constant Load', 'Constant Load', 'Constant Load', 'Constant Load'};

for k = test_battery_indices
    fprintf('Testing on battery %d...\n', k);
    Test_data = raw_data(k);
    GT_Trace = Test_data.Features(:, target_Feature);
    
    % X-axis: energy throughput (feature 21 if available, otherwise index)
    if size(raw_data(k).Features, 2) >= 21
        x_axis = raw_data(k).Features(1:size(GT_Trace,1), 21) / 1000;  % Convert to kWh
        x_label = 'Total Energy Throughput (kWh)';
    else
        x_axis = 1:size(GT_Trace, 1);
        x_label = 'Data Point Index';
    end
    
    for p = 1:length(LKP_array)
        LKP = LKP_array(p);
        
        if LKP >= length(GT_Trace(:, 1))
            continue;
        end
        
        % Prepare test input
        X_Test_stan = (Test_data.Features(1:LKP, desired_features) - mu) ./ sigma;
        
        % Initialize storage for ensemble predictions
        stan_SP = zeros(length(GT_Trace(:, 1)), numChannels, numSamples);
        
        for i = 1:numSamples
            stan_SP(1:LKP, :, i) = X_Test_stan;
        end
        
        % Autoregressive prediction
        for i = 1:numSamples
            for g = 1:(length(GT_Trace(:, 1)) - LKP)
                X_window = stan_SP(LKP - window_size + g:LKP + g - 1, :, i);
                YTest = predict(net{i}, X_window);
                stan_SP(LKP + g, :, i) = YTest(end, :);
            end
        end
        
        % Calculate ensemble statistics (denormalised)
        meanPrediction = zeros(length(GT_Trace(:, 1)), numChannels);
        stdPrediction = zeros(length(GT_Trace(:, 1)), numChannels);
        
        for t = 1:length(stan_SP(:, 1, 1))
            for c = 1:numChannels
                samples = squeeze(stan_SP(t, c, :) .* sigma(c) + mu(c));
                meanPrediction(t, c) = mean(samples);
                stdPrediction(t, c) = std(samples);
            end
        end
        
        % Confidence intervals (95%)
        confUpper = meanPrediction + 1.96 * stdPrediction;
        confLower = meanPrediction - 1.96 * stdPrediction;
        
        % Calculate metrics (on capacity - feature 1)
        actual = GT_Trace(LKP:end, 1);
        predicted = meanPrediction(LKP:end, 1);
        
        % MAPE
        actual_norm = (actual - mu(1)) ./ sigma(1);
        pred_norm = (predicted - mu(1)) ./ sigma(1);
        mape = mean(abs((actual_norm - pred_norm) ./ (actual_norm + eps))) * 100;
        
        % RMSE
        rmse = sqrt(mean((actual - predicted).^2));
        
        % R-squared
        ss_res = sum((actual - predicted).^2);
        ss_tot = sum((actual - mean(actual)).^2);
        r_squared = 1 - (ss_res / ss_tot);
        
        % Cross-correlation
        GT_norm_cc = (actual - mean(actual)) / std(actual);
        Pred_norm_cc = (predicted - mean(predicted)) / std(predicted);
        cross_corr = max(xcorr(GT_norm_cc, Pred_norm_cc, 'normalized'));
        
        % Store results
        results.battery_id(end+1) = k;
        results.LKP(end+1) = LKP;
        results.MAPE(end+1) = mape;
        results.RMSE(end+1) = rmse;
        results.R_squared(end+1) = r_squared;
        results.cross_corr(end+1) = cross_corr;
        
        % Store predictions for plotting
        pred_idx = length(predictions.battery_id) + 1;
        predictions.battery_id(pred_idx) = k;
        predictions.LKP(pred_idx) = LKP;
        predictions.ground_truth{pred_idx} = GT_Trace(:, 1);
        predictions.mean_pred{pred_idx} = meanPrediction(:, 1);
        predictions.std_pred{pred_idx} = stdPrediction(:, 1);
        predictions.conf_upper{pred_idx} = confUpper(:, 1);
        predictions.conf_lower{pred_idx} = confLower(:, 1);
        predictions.x_axis{pred_idx} = x_axis;
        predictions.x_label{pred_idx} = x_label;
        
        fprintf('  LKP=%d: MAPE=%.2f%%, R²=%.4f, RMSE=%.2f\n', LKP, mape, r_squared, rmse);
    end
end

% ====== DEGRADATION CURVE PLOTS ======
if plot_curves
    fprintf('\nGenerating degradation curve plots...\n');
    
    % Plot 1: Grid of all batteries, one LKP per battery (e.g., LKP=10)
    selected_LKP = 10;  % Choose which LKP to show in summary grid
    
    figure('Position', [50, 50, 1400, 900], 'Name', 'Degradation Curves Summary');
    num_test = length(test_battery_indices);
    cols = ceil(sqrt(num_test));
    rows = ceil(num_test / cols);
    
    for i = 1:num_test
        k = test_battery_indices(i);
        
        % Find prediction for this battery at selected LKP
        idx = find(predictions.battery_id == k & predictions.LKP == selected_LKP, 1);
        
        if isempty(idx)
            % If selected LKP not available, use first available
            idx = find(predictions.battery_id == k, 1);
        end
        
        if ~isempty(idx)
            subplot(rows, cols, i);
            hold on;
            
            x = predictions.x_axis{idx};
            gt = predictions.ground_truth{idx};
            pred = predictions.mean_pred{idx};
            conf_up = predictions.conf_upper{idx};
            conf_lo = predictions.conf_lower{idx};
            lkp = predictions.LKP(idx);
            
            % Shaded confidence interval (only for predicted region)
            x_pred = x(lkp:end);
            fill([x_pred; flipud(x_pred)], ...
                 [conf_up(lkp:end); flipud(conf_lo(lkp:end))], ...
                 [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
            
            % Ground truth
            plot(x, gt, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth');
            
            % Prediction
            plot(x(lkp:end), pred(lkp:end), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
            
            % Known data portion
            plot(x(1:lkp), gt(1:lkp), 'g-', 'LineWidth', 2, 'DisplayName', 'Known Data');
            
            % LKP marker
            plot(x(lkp), gt(lkp), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'g', ...
                 'DisplayName', 'Last Known Point');
            
            % Find metrics for title
            res_idx = find(results.battery_id == k & results.LKP == lkp, 1);
            if ~isempty(res_idx)
                title_str = sprintf('Battery %d (%d°C, %s)\nMAPE=%.1f%%, R²=%.3f', ...
                    k, Temperature(k), Cycle_Type_Names{k}, ...
                    results.MAPE(res_idx), results.R_squared(res_idx));
            else
                title_str = sprintf('Battery %d (%d°C, %s)', k, Temperature(k), Cycle_Type_Names{k});
            end
            
            title(title_str, 'FontSize', 9);
            xlabel(predictions.x_label{idx}, 'FontSize', 8);
            ylabel('Capacity (mAh)', 'FontSize', 8);
            grid on;
            hold off;
        end
    end
    
    sgtitle(sprintf('Degradation Predictions (LKP = %d) - Generalised Model', selected_LKP), ...
            'FontSize', 12, 'FontWeight', 'bold');
    
    if save_figures
        saveas(gcf, 'degradation_curves_summary.png');
        fprintf('  Saved: degradation_curves_summary.png\n');
    end
    
    % Plot 2: Detailed view for each battery showing all LKPs
    for k = test_battery_indices
        figure('Position', [100, 100, 1200, 800], 'Name', sprintf('Battery %d Detailed', k));
        
        % Find all predictions for this battery
        battery_idx = find(predictions.battery_id == k);
        num_lkps = length(battery_idx);
        
        cols_detail = min(num_lkps, 3);
        rows_detail = ceil(num_lkps / cols_detail);
        
        for j = 1:num_lkps
            idx = battery_idx(j);
            
            subplot(rows_detail, cols_detail, j);
            hold on;
            
            x = predictions.x_axis{idx};
            gt = predictions.ground_truth{idx};
            pred = predictions.mean_pred{idx};
            conf_up = predictions.conf_upper{idx};
            conf_lo = predictions.conf_lower{idx};
            lkp = predictions.LKP(idx);
            
            % Shaded confidence interval
            x_pred = x(lkp:end);
            fill([x_pred; flipud(x_pred)], ...
                 [conf_up(lkp:end); flipud(conf_lo(lkp:end))], ...
                 [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.5, ...
                 'DisplayName', '95% Confidence');
            
            % Ground truth
            plot(x, gt, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth (RPT)');
            
            % Prediction
            plot(x(lkp:end), pred(lkp:end), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
            
            % Confidence bounds
            plot(x(lkp:end), conf_up(lkp:end), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
            plot(x(lkp:end), conf_lo(lkp:end), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
            
            % Known data
            plot(x(1:lkp), gt(1:lkp), 'g-', 'LineWidth', 2, 'DisplayName', 'Known Data');
            
            % LKP marker
            xline(x(lkp), 'm--', 'LineWidth', 1.5, 'DisplayName', 'LKP');
            plot(x(lkp), gt(lkp), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g', ...
                 'HandleVisibility', 'off');
            
            % Metrics
            res_idx = find(results.battery_id == k & results.LKP == lkp, 1);
            if ~isempty(res_idx)
                title_str = sprintf('LKP = %d\nMAPE = %.2f%%, R² = %.4f, RMSE = %.1f', ...
                    lkp, results.MAPE(res_idx), results.R_squared(res_idx), results.RMSE(res_idx));
            else
                title_str = sprintf('LKP = %d', lkp);
            end
            
            title(title_str, 'FontSize', 10);
            xlabel(predictions.x_label{idx});
            ylabel('Capacity (mAh)');
            grid on;
            
            if j == 1
                legend('Location', 'southwest', 'FontSize', 8);
            end
            
            hold off;
        end
        
        sgtitle(sprintf('Battery %d: %d°C, %s - Predictions at Different LKPs', ...
                k, Temperature(k), Cycle_Type_Names{k}), ...
                'FontSize', 12, 'FontWeight', 'bold');
        
        if save_figures
            saveas(gcf, sprintf('battery_%d_detailed.png', k));
            fprintf('  Saved: battery_%d_detailed.png\n', k);
        end
    end
    
    % Plot 3: Overlay comparison - all batteries on one plot (single LKP)
    figure('Position', [100, 100, 1000, 600], 'Name', 'All Batteries Overlay');
    hold on;
    
    colors = lines(length(test_battery_indices));
    legend_entries = {};
    
    for i = 1:length(test_battery_indices)
        k = test_battery_indices(i);
        idx = find(predictions.battery_id == k & predictions.LKP == selected_LKP, 1);
        
        if ~isempty(idx)
            x = predictions.x_axis{idx};
            gt = predictions.ground_truth{idx};
            pred = predictions.mean_pred{idx};
            lkp = predictions.LKP(idx);
            
            % Normalise to initial capacity for comparison
            gt_norm = gt / gt(1) * 100;
            pred_norm = pred / gt(1) * 100;
            
            % Ground truth (solid)
            plot(x, gt_norm, '-', 'Color', colors(i,:), 'LineWidth', 1.5);
            
            % Prediction (dashed)
            plot(x(lkp:end), pred_norm(lkp:end), '--', 'Color', colors(i,:), 'LineWidth', 1.5);
            
            legend_entries{end+1} = sprintf('Batt %d GT', k);
            legend_entries{end+1} = sprintf('Batt %d Pred', k);
        end
    end
    
    xlabel('Total Energy Throughput (kWh)');
    ylabel('Capacity Retention (%)');
    title(sprintf('Normalised Degradation Curves (LKP = %d)', selected_LKP));
    legend(legend_entries, 'Location', 'eastoutside', 'FontSize', 8);
    grid on;
    hold off;
    
    if save_figures
        saveas(gcf, 'all_batteries_overlay.png');
        fprintf('  Saved: all_batteries_overlay.png\n');
    end
    
    % Plot 4: Error analysis - prediction error over time
    figure('Position', [100, 100, 1200, 500], 'Name', 'Prediction Error Analysis');
    
    subplot(1, 2, 1);
    hold on;
    for i = 1:length(test_battery_indices)
        k = test_battery_indices(i);
        idx = find(predictions.battery_id == k & predictions.LKP == selected_LKP, 1);
        
        if ~isempty(idx)
            gt = predictions.ground_truth{idx};
            pred = predictions.mean_pred{idx};
            lkp = predictions.LKP(idx);
            
            % Percentage error over prediction horizon
            error_pct = abs(gt(lkp:end) - pred(lkp:end)) ./ gt(lkp:end) * 100;
            prediction_horizon = 0:(length(error_pct)-1);
            
            plot(prediction_horizon, error_pct, '-', 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('Battery %d', k));
        end
    end
    xlabel('Steps Ahead from LKP');
    ylabel('Absolute Percentage Error (%)');
    title('Error Growth Over Prediction Horizon');
    legend('Location', 'northwest');
    grid on;
    hold off;
    
    subplot(1, 2, 2);
    hold on;
    for i = 1:length(test_battery_indices)
        k = test_battery_indices(i);
        idx = find(predictions.battery_id == k & predictions.LKP == selected_LKP, 1);
        
        if ~isempty(idx)
            gt = predictions.ground_truth{idx};
            pred = predictions.mean_pred{idx};
            std_p = predictions.std_pred{idx};
            lkp = predictions.LKP(idx);
            
            % Normalised error (in std units)
            norm_error = (gt(lkp:end) - pred(lkp:end)) ./ (std_p(lkp:end) + eps);
            prediction_horizon = 0:(length(norm_error)-1);
            
            plot(prediction_horizon, norm_error, '-', 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('Battery %d', k));
        end
    end
    yline([-1.96, 1.96], 'r--', 'LineWidth', 1, 'HandleVisibility', 'off');
    xlabel('Steps Ahead from LKP');
    ylabel('Normalised Error (σ units)');
    title('Error Relative to Predicted Uncertainty');
    legend('Location', 'northwest');
    grid on;
    hold off;
    
    sgtitle(sprintf('Prediction Error Analysis (LKP = %d)', selected_LKP));
    
    if save_figures
        saveas(gcf, 'error_analysis.png');
        fprintf('  Saved: error_analysis.png\n');
    end
end

% ====== SUMMARY STATISTICS ======
fprintf('\n====== BENCHMARK SUMMARY ======\n');
fprintf('Total test cases: %d\n', length(results.MAPE));
fprintf('Mean MAPE: %.2f%% (std: %.2f%%)\n', mean(results.MAPE), std(results.MAPE));
fprintf('Mean R²: %.4f (std: %.4f)\n', mean(results.R_squared), std(results.R_squared));
fprintf('Mean RMSE: %.2f (std: %.2f)\n', mean(results.RMSE), std(results.RMSE));

fprintf('\nResults by LKP:\n');
for lkp = LKP_array
    idx = results.LKP == lkp;
    if any(idx)
        fprintf('  LKP=%2d: MAPE=%5.2f%% (±%5.2f), R²=%.4f (±%.4f), n=%d\n', ...
            lkp, mean(results.MAPE(idx)), std(results.MAPE(idx)), ...
            mean(results.R_squared(idx)), std(results.R_squared(idx)), sum(idx));
    end
end

fprintf('\nResults by Battery:\n');
for k = test_battery_indices
    idx = results.battery_id == k;
    if any(idx)
        fprintf('  Battery %2d (%2d°C, %s): MAPE=%5.2f%% (±%5.2f), R²=%.4f\n', ...
            k, Temperature(k), Cycle_Type_Names{k}(1:5), ...
            mean(results.MAPE(idx)), std(results.MAPE(idx)), mean(results.R_squared(idx)));
    end
end

% ====== METRICS HEATMAPS ======
figure('Position', [100, 100, 1200, 500], 'Name', 'Metrics Summary');

subplot(1, 2, 1);
mape_matrix = nan(length(test_battery_indices), length(LKP_array));
for i = 1:length(test_battery_indices)
    for j = 1:length(LKP_array)
        idx = results.battery_id == test_battery_indices(i) & results.LKP == LKP_array(j);
        if any(idx)
            mape_matrix(i, j) = results.MAPE(idx);
        end
    end
end
heatmap(LKP_array, test_battery_indices, mape_matrix);
xlabel('Last Known Point');
ylabel('Battery ID');
title('MAPE (%)');
colormap(flipud(hot));

subplot(1, 2, 2);
r2_matrix = nan(length(test_battery_indices), length(LKP_array));
for i = 1:length(test_battery_indices)
    for j = 1:length(LKP_array)
        idx = results.battery_id == test_battery_indices(i) & results.LKP == LKP_array(j);
        if any(idx)
            r2_matrix(i, j) = results.R_squared(idx);
        end
    end
end
heatmap(LKP_array, test_battery_indices, r2_matrix);
xlabel('Last Known Point');
ylabel('Battery ID');
title('R²');

sgtitle('Generalised Model Benchmark Metrics');

if save_figures
    saveas(gcf, 'metrics_heatmaps.png');
    fprintf('  Saved: metrics_heatmaps.png\n');
end

fprintf('\nPlotting complete.\n');