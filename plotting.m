%% Plot Degradation Traces for Batteries 1-3
% Run this script AFTER running Data_Structure.m
% This will plot the capacity degradation curves for the first 3 batteries

% Define battery indices to plot
batteries_to_plot = [1:16];

% Get temperature and cycle type labels
temp_labels = {'10°C', '10°C', '10°C', '25°C', '25°C', '40°C', '40°C', '40°C', ...
               '10°C', '10°C', '10°C', '25°C', '25°C', '40°C', '40°C', '40°C'};
cycle_labels = {'Drive Cycle', 'Drive Cycle', 'Drive Cycle', 'Drive Cycle', ...
                'Drive Cycle', 'Drive Cycle', 'Drive Cycle', 'Drive Cycle', ...
                'Constant Load', 'Constant Load', 'Constant Load', 'Constant Load', ...
                'Constant Load', 'Constant Load', 'Constant Load', 'Constant Load'};
cell_labels = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'};

% Define colors for each battery
colors = lines(length(batteries_to_plot));

%% Figure 1: RPT Capacity Degradation
figure('Name', 'RPT Capacity Degradation - Batteries 1-3', 'Position', [100, 100, 900, 600]);

hold on;
legend_entries = {};

for idx = 1:length(batteries_to_plot)
    b = batteries_to_plot(idx);
    
    % Extract data
    % Feature 1 = RPT Capacity (mAh)
    % Feature 21 = Energy throughput (Wh)
    capacity = raw_data(b).Features(:, 1);
    energy_throughput = raw_data(b).Features(:, 21) / 1000; % Convert to kWh
    
    % Plot
    plot(energy_throughput, capacity, 'LineWidth', 2, 'Color', colors(idx, :));
    
    % Build legend entry
    legend_entries{idx} = sprintf('Cell %s (%s, %s)', ...
        cell_labels{b}, temp_labels{b}, cycle_labels{b});
end

hold off;

% Formatting
xlabel('Total Energy Throughput (kWh)', 'FontSize', 12);
ylabel('RPT Capacity (mAh)', 'FontSize', 12);
title('Battery Degradation Traces - RPT Capacity', 'FontSize', 14);
legend(legend_entries, 'Location', 'southwest', 'FontSize', 10);
grid on;
box on;

%% Figure 2: Cycling Capacity Degradation
figure('Name', 'Cycling Capacity Degradation - Batteries 1-3', 'Position', [150, 150, 900, 600]);

hold on;
legend_entries = {};

for idx = 1:length(batteries_to_plot)
    b = batteries_to_plot(idx);
    
    % Extract data
    % Feature 12 = Cycling Capacity (Ah) - note: multiply by 1000 to get mAh
    % Feature 21 = Energy throughput (Wh)
    cycling_capacity = raw_data(b).Features(:, 12) * 1000; % Convert to mAh
    energy_throughput = raw_data(b).Features(:, 21) / 1000; % Convert to kWh
    
    % Plot
    plot(energy_throughput, cycling_capacity, 'LineWidth', 2, 'Color', colors(idx, :));
    
    % Build legend entry
    legend_entries{idx} = sprintf('Cell %s (%s, %s)', ...
        cell_labels{b}, temp_labels{b}, cycle_labels{b});
end

hold off;

% Formatting
xlabel('Total Energy Throughput (kWh)', 'FontSize', 12);
ylabel('Cycling Capacity (mAh)', 'FontSize', 12);
title('Battery Degradation Traces - Cycling Capacity', 'FontSize', 14);
legend(legend_entries, 'Location', 'southwest', 'FontSize', 10);
grid on;
box on;

%% Figure 3: Combined Plot (RPT and Cycling)
figure('Name', 'Combined Degradation - Batteries 1-3', 'Position', [200, 200, 1000, 700]);

hold on;
legend_entries = {};
line_styles = {'-', '--'}; % Solid for RPT, dashed for Cycling

for idx = 1:length(batteries_to_plot)
    b = batteries_to_plot(idx);
    
    % Extract data
    rpt_capacity = raw_data(b).Features(:, 1);
    cycling_capacity = raw_data(b).Features(:, 12) * 1000;
    energy_throughput = raw_data(b).Features(:, 21) / 1000;
    
    % Plot RPT capacity (solid line)
    plot(energy_throughput, rpt_capacity, '-', 'LineWidth', 2, 'Color', colors(idx, :));
    
    % Plot Cycling capacity (dashed line)
    plot(energy_throughput, cycling_capacity, '--', 'LineWidth', 1.5, 'Color', colors(idx, :));
end

hold off;

% Create custom legend
legend_entries = {};
for idx = 1:length(batteries_to_plot)
    b = batteries_to_plot(idx);
    legend_entries{end+1} = sprintf('Cell %s RPT (%s)', cell_labels{b}, temp_labels{b});
    legend_entries{end+1} = sprintf('Cell %s Cycling (%s)', cell_labels{b}, temp_labels{b});
end

xlabel('Total Energy Throughput (kWh)', 'FontSize', 12);
ylabel('Capacity (mAh)', 'FontSize', 12);
title('Battery Degradation Traces - RPT vs Cycling Capacity', 'FontSize', 14);
legend(legend_entries, 'Location', 'southwest', 'FontSize', 9);
grid on;
box on;

%% Figure 4: Subplots for Each Battery
figure('Name', 'Individual Battery Degradation', 'Position', [250, 100, 1200, 400]);

for idx = 1:length(batteries_to_plot)
    b = batteries_to_plot(idx);
    
    subplot(1, 3, idx);
    hold on;
    
    % Extract data
    rpt_capacity = raw_data(b).Features(:, 1);
    cycling_capacity = raw_data(b).Features(:, 12) * 1000;
    energy_throughput = raw_data(b).Features(:, 21) / 1000;
    
    % Plot both traces
    plot(energy_throughput, rpt_capacity, 'b-', 'LineWidth', 2);
    plot(energy_throughput, cycling_capacity, 'r--', 'LineWidth', 1.5);
    
    hold off;
    
    xlabel('Energy Throughput (kWh)', 'FontSize', 10);
    ylabel('Capacity (mAh)', 'FontSize', 10);
    title(sprintf('Cell %s (%s, %s)', cell_labels{b}, temp_labels{b}, cycle_labels{b}), 'FontSize', 11);
    legend({'RPT Capacity', 'Cycling Capacity'}, 'Location', 'southwest', 'FontSize', 8);
    grid on;
    box on;
end

%% Print Summary Statistics
fprintf('\n=== Battery Degradation Summary (Batteries 1-3) ===\n\n');

for idx = 1:length(batteries_to_plot)
    b = batteries_to_plot(idx);
    
    rpt_capacity = raw_data(b).Features(:, 1);
    energy_throughput = raw_data(b).Features(:, 21) / 1000;
    
    initial_cap = rpt_capacity(1);
    final_cap = rpt_capacity(end);
    total_energy = energy_throughput(end);
    capacity_fade = (initial_cap - final_cap) / initial_cap * 100;
    
    fprintf('Cell %s (%s, %s):\n', cell_labels{b}, temp_labels{b}, cycle_labels{b});
    fprintf('  Initial Capacity: %.1f mAh\n', initial_cap);
    fprintf('  Final Capacity:   %.1f mAh\n', final_cap);
    fprintf('  Capacity Fade:    %.2f%%\n', capacity_fade);
    fprintf('  Total Energy:     %.1f kWh\n', total_energy);
    fprintf('  Data Points:      %d\n\n', length(rpt_capacity));
end

fprintf('Plots generated successfully!\n');