%% Battery degradation predictions using a GRU-LSTM neural net
% Modified for single generalised model training
% Original: Benjamin Warmington
% Modified for benchmarking: [Your name]

close all;

% ====== CONFIGURATION ======
% Select which batteries to hold out for testing (leave-one-out or hold-out set)
test_battery_indices = [2, 9, 16];  % Example: test on batteries 2, 9, 16
% Or for leave-one-out on all: test_battery_indices = 1:16;

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

% ====== PREPARE TRAINING DATA (NO BIASING) ======
% Use all batteries EXCEPT the test set for training
Train_data = raw_data;

% Remove test batteries and their bootstrapped copies
% Sort in descending order to avoid index shifting issues
indices_to_remove = sort([test_battery_indices, test_battery_indices + 16], 'descend');
for idx = indices_to_remove
    if idx <= length(Train_data)
        Train_data(idx) = [];
    end
end

% Also remove battery 1 if it's an outlier (as Ben noted)
% Uncomment if needed:
% Train_data([1, 17]) = [];

num_train_batteries = length(Train_data);
fprintf('Training on %d batteries (including bootstrapped copies)\n', num_train_batteries);

% ====== NORMALISATION ======
% Build normalisation parameters from ALL training data
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
numObservations = numel(X_cell);
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
    
    % Network architecture (same as Ben's)
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

for k = test_battery_indices
    fprintf('Testing on battery %d...\n', k);
    Test_data = raw_data(k);
    GT_Trace = Test_data.Features(:, target_Feature);
    
    for p = 1:length(LKP_array)
        LKP = LKP_array(p);
        
        % Skip if LKP exceeds available data
        if LKP >= length(GT_Trace(:, 1))
            continue;
        end
        
        % Prepare test input
        X_Test_stan = (Test_data.Features(1:LKP, desired_features) - mu) ./ sigma;
        
        % Initialize storage for ensemble predictions
        stan_SP = zeros(length(GT_Trace(:, 1)), numChannels, numSamples);
        
        % Initialize all samples with known data
        for i = 1:numSamples
            stan_SP(1:LKP, :, i) = X_Test_stan;
        end
        
        % Autoregressive prediction with each ensemble member
        for i = 1:numSamples
            for g = 1:(length(GT_Trace(:, 1)) - LKP)
                X_window = stan_SP(LKP - window_size + g:LKP + g - 1, :, i);
                YTest = predict(net{i}, X_window);
                stan_SP(LKP + g, :, i) = YTest(end, :);
            end
        end
        
        % Calculate ensemble mean prediction (denormalised)
        meanPrediction = zeros(length(GT_Trace(:, 1)), numChannels);
        stdPrediction = zeros(length(GT_Trace(:, 1)), numChannels);
        
        for t = 1:length(stan_SP(:, 1, 1))
            for c = 1:numChannels
                samples = squeeze(stan_SP(t, c, :) .* sigma(c) + mu(c));
                meanPrediction(t, c) = mean(samples);
                stdPrediction(t, c) = std(samples);
            end
        end
        
        % Calculate metrics (on capacity - feature 1)
        actual = GT_Trace(LKP:end, 1);
        predicted = meanPrediction(LKP:end, 1);
        
        % MAPE (on normalised data as Ben does)
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
        GT_norm = (actual - mean(actual)) / std(actual);
        Pred_norm_cc = (predicted - mean(predicted)) / std(predicted);
        cross_corr = max(xcorr(GT_norm, Pred_norm_cc, 'normalized'));
        
        % Store results
        results.battery_id(end+1) = k;
        results.LKP(end+1) = LKP;
        results.MAPE(end+1) = mape;
        results.RMSE(end+1) = rmse;
        results.R_squared(end+1) = r_squared;
        results.cross_corr(end+1) = cross_corr;
        
        fprintf('  LKP=%d: MAPE=%.2f%%, R²=%.4f, RMSE=%.2f\n', LKP, mape, r_squared, rmse);
    end
end

% ====== SUMMARY STATISTICS ======
fprintf('\n====== BENCHMARK SUMMARY ======\n');
fprintf('Total test cases: %d\n', length(results.MAPE));
fprintf('Mean MAPE: %.2f%% (std: %.2f%%)\n', mean(results.MAPE), std(results.MAPE));
fprintf('Mean R²: %.4f (std: %.4f)\n', mean(results.R_squared), std(results.R_squared));
fprintf('Mean RMSE: %.2f (std: %.2f)\n', mean(results.RMSE), std(results.RMSE));

% Results by LKP
fprintf('\nResults by Last Known Point:\n');
for lkp = LKP_array
    idx = results.LKP == lkp;
    if any(idx)
        fprintf('  LKP=%d: MAPE=%.2f%% (±%.2f), R²=%.4f (±%.4f)\n', ...
            lkp, mean(results.MAPE(idx)), std(results.MAPE(idx)), ...
            mean(results.R_squared(idx)), std(results.R_squared(idx)));
    end
end

% Results by battery
fprintf('\nResults by Battery:\n');
for k = test_battery_indices
    idx = results.battery_id == k;
    if any(idx)
        fprintf('  Battery %d: MAPE=%.2f%% (±%.2f), R²=%.4f\n', ...
            k, mean(results.MAPE(idx)), std(results.MAPE(idx)), mean(results.R_squared(idx)));
    end
end

% ====== VISUALISATION ======
figure('Position', [100, 100, 1200, 800]);

% Heatmap of MAPE by battery and LKP
subplot(2, 2, 1);
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
title('MAPE (%) by Battery and LKP');
colormap(flipud(hot));

% Heatmap of R² by battery and LKP
subplot(2, 2, 2);
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
title('R² by Battery and LKP');

% Box plot of MAPE by LKP
subplot(2, 2, 3);
boxplot(results.MAPE, results.LKP);
xlabel('Last Known Point');
ylabel('MAPE (%)');
title('MAPE Distribution by LKP');

% Box plot of MAPE by battery
subplot(2, 2, 4);
boxplot(results.MAPE, results.battery_id);
xlabel('Battery ID');
ylabel('MAPE (%)');
title('MAPE Distribution by Battery');

sgtitle('Generalised Model Benchmark Results (No Biasing)');

%%
