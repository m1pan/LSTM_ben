%% NASA Random Walk Battery Data Loader
% This script loads NASA RW battery data and extracts features
% compatible with the GRU-LSTM degradation prediction model.
%
% Supports both dataset variants:
% 1. RW_ChargeDischarge_RT (batteries RW9, RW10, RW11, RW12)
%    - Reference cycles every 1500 RW steps
% 2. RW_UniformDischargeVariableCharge_RT (batteries RW1, RW2, RW7, RW8)
%    - Reference cycles every 50 RW cycles
%
% Usage:
%   1. Set the data_dir to your NASA dataset folder
%   2. Set the battery_labels for your dataset variant
%   3. Run the script
%   4. Use the resulting raw_data struct with LTST_multivariate3.m

close all;
clear all;

%% Configuration - MODIFY THESE FOR YOUR DATASET

% Path to the folder containing RW*.mat files
data_dir = '/Users/michael/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ME4/FYP/Datasets/nasa/11. Randomized Battery Usage Data Set/Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab';

% Battery labels - choose one set based on your dataset:
% For RW_UniformDischargeVariableCharge_RT:
battery_labels = ["RW1", "RW2", "RW7", "RW8"];
% For RW_ChargeDischarge_RT:
% battery_labels = ["RW9", "RW10", "RW11", "RW12"];

% Reference discharge comment string
ref_discharge_comment = 'reference discharge';

% All batteries are at room temperature (~25C) in NASA dataset
Temperature = [25, 25, 25, 25]';

% All batteries use the same cycling protocol (use 1 for all)
Cycle_Type = [1, 1, 1, 1]';

% Energy granularity in Wh 
% NOTE: NASA dataset has less data than Kirkaldy, so use smaller granularity
% RW1 has ~7.8 kWh total, so 200 Wh gives ~39 points
Point_x_diff = 200;

%% Load and process each battery

num_batteries = length(battery_labels);
raw_data = struct();

for j = 1:num_batteries
    
    fprintf('Processing battery %s (%d/%d)...\n', battery_labels(j), j, num_batteries);
    
    % Load the .mat file
    mat_file = fullfile(data_dir, strcat(battery_labels(j), '.mat'));
    loaded = load(mat_file);
    battery_data = loaded.data;
    
    % Extract step array - note the specific indexing for MATLAB structs from scipy
    steps = battery_data.step;
    num_steps = length(steps);
    
    %% Find reference discharge cycles and extract capacity
    ref_discharge_idx = [];
    ref_discharge_capacity = [];
    ref_discharge_time = [];
    ref_discharge_energy_throughput = [];
    
    cumulative_energy = 0;  % Track total energy throughput
    
    for i = 1:num_steps
        step = steps(i);
        
        % Get comment - it's a char array in the .mat file
        comment = step.comment;
        if iscell(comment)
            comment = comment{1};
        end
        comment = strtrim(comment);
        
        % Get data arrays
        current = step.current(:);
        voltage = step.voltage(:);
        time = step.relativeTime(:);
        
        % Calculate energy for this step (for tracking cumulative)
        if ~isempty(current) && ~isempty(voltage) && ~isempty(time)
            % Energy = integral of |current * voltage| dt (in Wh)
            if length(time) > 1
                dt = diff(time) / 3600;  % Convert seconds to hours
                power = abs(current(1:end-1) .* voltage(1:end-1));
                step_energy = sum(power .* dt);
                cumulative_energy = cumulative_energy + step_energy;
            end
        end
        
        % Check if this is a reference discharge
        if strcmp(comment, ref_discharge_comment)
            ref_discharge_idx(end+1) = i;
            
            % Calculate discharge capacity (Ah) by integrating current
            if length(time) > 1
                dt = diff(time) / 3600;  % Convert seconds to hours
                capacity_ah = sum(abs(current(1:end-1)) .* dt);
                capacity_mah = capacity_ah * 1000;  % Convert to mAh
            else
                capacity_mah = NaN;
            end
            
            ref_discharge_capacity(end+1) = capacity_mah;
            ref_discharge_time(end+1) = step.time(1);  % Time at start of step
            ref_discharge_energy_throughput(end+1) = cumulative_energy;
        end
    end
    
    fprintf('  Found %d reference discharge cycles\n', length(ref_discharge_idx));
    fprintf('  Capacity range: %.1f - %.1f mAh\n', ...
        max(ref_discharge_capacity), min(ref_discharge_capacity));
    
    %% Extract additional features from reference cycles
    % For each reference discharge, also get temperature and voltage stats
    
    ref_avg_temp = zeros(size(ref_discharge_capacity));
    ref_avg_voltage = zeros(size(ref_discharge_capacity));
    ref_min_voltage = zeros(size(ref_discharge_capacity));
    ref_max_voltage = zeros(size(ref_discharge_capacity));
    ref_duration = zeros(size(ref_discharge_capacity));
    
    for k = 1:length(ref_discharge_idx)
        idx = ref_discharge_idx(k);
        step = steps(idx);
        
        temp_data = step.temperature(:);
        volt_data = step.voltage(:);
        time_data = step.relativeTime(:);
        
        ref_avg_temp(k) = mean(temp_data);
        ref_avg_voltage(k) = mean(volt_data);
        ref_min_voltage(k) = min(volt_data);
        ref_max_voltage(k) = max(volt_data);
        ref_duration(k) = time_data(end) / 3600;  % Hours
    end
    
    %% Create feature matrix similar to Kirkaldy
    % Interpolate to regular energy throughput grid
    
    if isempty(ref_discharge_energy_throughput)
        warning('No reference discharge cycles found for battery %s', battery_labels(j));
        continue;
    end
    
    % Create x-axis based on energy throughput
    max_energy = max(ref_discharge_energy_throughput);
    Points_x = 0:Point_x_diff:max_energy;
    Tot_points = length(Points_x);
    
    if Tot_points < 3
        warning('Not enough data points for battery %s', battery_labels(j));
        continue;
    end
    
    % Interpolate features onto regular grid
    Cap = interp1(ref_discharge_energy_throughput, ref_discharge_capacity, Points_x, 'linear', 'extrap');
    Temp = interp1(ref_discharge_energy_throughput, ref_avg_temp, Points_x, 'linear', 'extrap');
    Avg_V = interp1(ref_discharge_energy_throughput, ref_avg_voltage, Points_x, 'linear', 'extrap');
    Min_V = interp1(ref_discharge_energy_throughput, ref_min_voltage, Points_x, 'linear', 'extrap');
    Max_V = interp1(ref_discharge_energy_throughput, ref_max_voltage, Points_x, 'linear', 'extrap');
    Duration = interp1(ref_discharge_energy_throughput, ref_duration, Points_x, 'linear', 'extrap');
    
    % Calculate gradients (rate of change)
    Cap_Grad = gradient(Cap) ./ Point_x_diff;
    Temp_Grad = gradient(Temp) ./ Point_x_diff;
    V_Grad = gradient(Avg_V) ./ Point_x_diff;
    Duration_Grad = gradient(Duration) ./ Point_x_diff;
    
    %% Build feature matrix
    % Structure similar to Kirkaldy but with available NASA features
    
    num_points = length(Points_x) - 1;  % Lose one point for gradients
    
    for i = 1:num_points
        % Feature 1: Capacity (target variable)
        raw_data(j).Features(i, 1) = Cap(i);
        % Feature 2: Capacity gradient
        raw_data(j).Features(i, 2) = Cap_Grad(i);
        % Feature 3: Average temperature
        raw_data(j).Features(i, 3) = Temp(i);
        % Feature 4: Temperature gradient
        raw_data(j).Features(i, 4) = Temp_Grad(i);
        % Feature 5: Average discharge voltage
        raw_data(j).Features(i, 5) = Avg_V(i);
        % Feature 6: Voltage gradient
        raw_data(j).Features(i, 6) = V_Grad(i);
        % Feature 7: Min voltage during discharge
        raw_data(j).Features(i, 7) = Min_V(i);
        % Feature 8: Max voltage during discharge
        raw_data(j).Features(i, 8) = Max_V(i);
        % Feature 9: Discharge duration (related to internal resistance)
        raw_data(j).Features(i, 9) = Duration(i);
        % Feature 10: Duration gradient
        raw_data(j).Features(i, 10) = Duration_Grad(i);
        % Feature 11: Voltage range (max - min)
        raw_data(j).Features(i, 11) = Max_V(i) - Min_V(i);
        % Feature 12: Energy throughput
        raw_data(j).Features(i, 12) = Points_x(i);
        % Feature 13: Sequence counter
        raw_data(j).Features(i, 13) = i - 1;
    end
    
    % Store classifiers
    raw_data(j).Classifiers(1) = Temperature(j);
    raw_data(j).Classifiers(2) = Cycle_Type(j);
    
    fprintf('  Created %d feature vectors with %d features\n', ...
        size(raw_data(j).Features, 1), size(raw_data(j).Features, 2));
end

%% Create augmented copies (bootstrapping)
% Same as in Kirkaldy Data_Structure.m

fprintf('\nCreating augmented copies...\n');

for j = num_batteries+1:2*num_batteries
    
    orig_idx = j - num_batteries;
    mult = (rand() - 0.5) * 0.03 + 1;  % Random multiplier 0.985 to 1.015
    
    raw_data(j).Features = raw_data(orig_idx).Features * mult;
    raw_data(j).Classifiers = raw_data(orig_idx).Classifiers;
    
end

fprintf('Total batteries (including augmented): %d\n', length(raw_data));

%% Summary statistics

fprintf('\n=== Dataset Summary ===\n');
for j = 1:num_batteries
    fprintf('Battery %s:\n', battery_labels(j));
    fprintf('  Data points: %d\n', size(raw_data(j).Features, 1));
    fprintf('  Initial capacity: %.1f mAh\n', raw_data(j).Features(1, 1));
    fprintf('  Final capacity: %.1f mAh\n', raw_data(j).Features(end, 1));
    fprintf('  Capacity fade: %.1f%%\n', ...
        100 * (1 - raw_data(j).Features(end, 1) / raw_data(j).Features(1, 1)));
end

%% Plot capacity degradation curves

figure;
hold on;
colors = lines(num_batteries);
for j = 1:num_batteries
    x = raw_data(j).Features(:, 12) / 1000;  % Energy throughput in kWh
    y = raw_data(j).Features(:, 1);  % Capacity in mAh
    plot(x, y, 'Color', colors(j,:), 'LineWidth', 1.5, 'DisplayName', char(battery_labels(j)));
end
xlabel('Energy Throughput (kWh)');
ylabel('Capacity (mAh)');
title('NASA RW Battery Degradation');
legend('Location', 'southwest');
grid on;

%% Save processed data

save_file = fullfile(data_dir, 'nasa_rw_processed.mat');
save(save_file, 'raw_data', 'battery_labels', 'Temperature', 'Cycle_Type');
fprintf('\nSaved processed data to: %s\n', save_file);

%% Note about using with LTST_multivariate3.m
fprintf('\n=== To use with LTST_multivariate3.m ===\n');
fprintf('1. Load the processed data: load(''%s'')\n', save_file);
fprintf('2. Modify target_Feature in LTST_multivariate3.m to: [1:13]\n');
fprintf('   (or select specific features you want to use)\n');
fprintf('3. Adjust the test battery index k as needed\n');
fprintf('4. Note: This dataset has fewer batteries than Kirkaldy,\n');
fprintf('   so bias parameters (alpha, beta) may need adjustment.\n');
