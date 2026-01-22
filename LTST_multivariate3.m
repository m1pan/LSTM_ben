%% Battery degradation predictions using a GRU-LSTM neural net
% Benjamin Warmington, yp20984@bristol.ac.uk or
% benjaminwarmington@hotmail.co.uk
% 31/03/2025

% This is the main code, but relies on first running `Data_Structure'. Data
% structure absolutely should be made a fuction to unify the code, however
% its not atm. Check `Data_Structure' for details on how the struct
% `raw_data' is built. 

% READ ME; 
% This code relies on downloading the matlab package `Deep learning
% toolbox'. Matlab's 2024 updates to its deep learning framework have
% greatly improved its functionality, here we will be using two types of RNN
% layer, the GRU and the LSTM. Check the companion report for details.

% The param goodness (W) and gd (k) loops are what I've been using to check
% metrics, if you want to check a hyperparameter, make it some function of
% W, and use the W for loop. If you want to see a specific battery set k to
% the battery number you want.

close all;

% Arrays for comparing average metrics
param_goodness = zeros(9, 1);
param_goodness2 = zeros(9, 1);
param_goodness3 = zeros(9, 1);

for W = 1:1:1

    % Arrays for storing individual battery metrics at different forecast
    % starting points
    gd1 = zeros(length(raw_data(:)), 5);
    gd2 = zeros(length(raw_data(:)), 5);
    gd3 = zeros(length(raw_data(:)), 5);


    % For looping through or selecting a battery
    for k = [2]%[2, 9, 16] %12:12%:2:14%[2, 4, 9, 11]%length(raw_data(:))/2

        % ------ THESE ARE VARIABLE PARAMETERS ---------
        
        % Target feature allows selecting which features you want to use
        % from raw data. See that file for details.
        target_Feature = [1:13]%[1:7, 12:21];%[1:2, 7:15];%[1, 2, 4, 10:12];
        desired_features = target_Feature;

        % Hyperparameter for sliding window size
        window_size = 7;
        % Temperature bias
        alpha = 3;
        % Cycle type bias
        beta = 2;
        % Different forecasting start points.
        LKP_array = [7, 10, 13, 16, 19];
        % Amount of total data used to train each ensemble
        ensemble_data_share = 0.6;
        % Ensemble size
        numSamples = 7;

        % Hyperparams of the neural net, cell numbers effect network
        % memory, Drop out layers prevent overfitting. Epochs are the
        % amount of times the network is trained on all of its dataa
        gru1_cell_num = 224;
        DOL1 = 0.05;
        lstm_cell_num = 200;
        DOL2 = 0.05;
        gru2_cell_num = 200;
        DOL3 = 0.05;
        epochs = 700;


        % ------ CHANGING THINGS FROM HERE WILL EFFECT THE CODE --------
        

        % This sets which data is used as the test set.
        Test_data = raw_data(k);

        projection = 1;
        
        % Getting the test data classifiers
        Temp_bias = Test_data.Classifiers(1);
        Cycle_bias = Test_data.Classifiers(2);

        % Setting the training data as all the raw data...
        Train_data = raw_data;
        % And then deleting the test data (and the bootstrapped data that
        % came from the testing data)
        % I'm also deleting the first data set as its a massive outlier,
        % this isn't actually strictly necessary.
        % Train_data(k+16) = [];
        % Train_data(1+16) = [];
        Train_data(k+4) = [];
        % Train_data(k) = [];
        % Train_data(1) = [];

        num_batteries = length(Train_data(:));


        % Initialising data stacks
        X_flattened = [];
        y_flattened = [];
        battery_ids = [];
        window_starts = [];
        train_temperatures = [];
        train_types = [];
        X_cell = {};
        Y_cell = {};
     
        % Duplicating data sets using biases.
        dupearray = zeros(1, num_batteries);
        for i = 1:num_batteries
            if Train_data(i).Classifiers(1) == Temp_bias 
                a = alpha;
            else
                a = 1;
            end

            if Train_data(i).Classifiers(2) == Cycle_bias
                b = beta;
            else
                b = 1;
            end
            dupearray(1, i) = b*a;
        end

        Train_data = duplicateStruct(Train_data, dupearray);

        % This is for normalising the data, essentially making massive
        % arrays of everything in a given feature across all of the training
        % data.

        num_features = length(Train_data(1).Features(1, desired_features));

        Tot_Feature_Lists = zeros(0, num_features);

        for b = 1:num_batteries

            Tot_Feature_Lists = [Tot_Feature_Lists; Train_data(b).Features(:, desired_features)];

        end

        % We use z score normalisation
        [X_train_std, mu, sigma] = zscore(Tot_Feature_Lists);

        % This just produces a correlation matrix so you can check if
        % anythings too linearly dependant...
        % corr_mat = corrcoef(X_train_std);
        % %Check cocorrelation of features
        % figure(1)
        % heatmap(corr_mat);
        %



        % Min-max scaling to [0,1] range - this is for if you want to
        % normalise between 0 and 1. zscore typically works better
        Min =  min(Tot_Feature_Lists);
        Range = max(Tot_Feature_Lists) - min(Tot_Feature_Lists);
        X_minmax = (Tot_Feature_Lists - Min) ./ Range;


        % Uncomment this if you want to convert to 0 to 1 normalisation
        %--------------------
        % mu = Min;
        % sigma = Range;
        %--------------------

        % This builds the data stack out of moving windows from all the
        % battery experiment sets.

        for b = 1:num_batteries

            num_rpts = length(Train_data(b).Features(:, 1));
            max_window_start_1 = num_rpts - (window_size + projection - 1);

            if max_window_start_1 > 0
                for w_start = 1:max_window_start_1

                    w_end = w_start + window_size - 1;

                    X_Data_stan = (Train_data(b).Features(w_start:w_end, desired_features) - mu)./sigma;
                    Y_Data_stan =  (Train_data(b).Features(w_start+projection:w_end+projection, target_Feature) - mu)./sigma;
              
                    X_cell{end+1, 1} = X_Data_stan;
                    Y_cell{end+1, 1} = Y_Data_stan;
                    window_starts = [window_starts; w_start];
                end

            end

        end

        % Getting the size of the input data

        numObservations = numel(X_cell);
        numResponses = size(Y_cell{1}, 2);
        numChannels = size(X_cell{1},2);

        % This initialises the cell we'll store all the neural nets in. It
        % should probably be called `nets' lol.
%%
        net = {};

        for i = 1:numSamples
            
            i
            % Selecting `ensemble data share' amount of the data randomly
            % for each new neural net.
            randArray = rand(length(X_cell), 1);
            [~, idx] = sort(randArray);
            rand_select = idx(1:ceil(length(idx)*ensemble_data_share));
            XTrain = X_cell(rand_select);
            TTrain = Y_cell(rand_select);


            % The network layer stack. We've prenormalized the data, so
            % it's set to none.
            layers = [
                sequenceInputLayer(numChannels, Normalization= "none")%, NormalizationDimension= "channel")
                gruLayer(gru1_cell_num)%, 'OutputMode','sequence')
                dropoutLayer(DOL1)
                lstmLayer(lstm_cell_num)%, 'OutputMode', 'sequence')
                dropoutLayer(DOL2)
                lstmLayer(lstm_cell_num)%, 'OutputMode', 'sequence')
                dropoutLayer(DOL2)
                gruLayer(gru2_cell_num)%, 'OutputMode', 'sequence')
                dropoutLayer(DOL3)
                fullyConnectedLayer(numChannels)];

            % Network options -  if you want to see training plots
            % uncomment the plots option. it needs to be above verbose
            % though.
                options = trainingOptions("adam", ...
                MaxEpochs=epochs, ...
                SequencePaddingDirection="left", ...
                Shuffle="every-epoch", ... 
                Verbose=true);
                % Plots="training-progress", ...
                % Verbose=true);


            net{end+1} = trainnet(XTrain,TTrain,layers,"mse",options);

        end

      
%%
        close all
        
        figure(1)
        tiledlayout(2, 2)
        % Testing all the LKP array on the net ensemble.
        for p = 1:length(LKP_array)-1
    %%
            LKP = LKP_array(p);

            % Sort out test data first, will be putting multiple models into this
            Xtest = Test_data(1).Features(1:LKP, target_Feature);
            GT_Trace = Test_data(1).Features(:, target_Feature);
            X_Test_stan = (Test_data(1).Features(1:LKP, target_Feature)- mu)./sigma;
            X_test_fin = X_Test_stan;

            

            % Initialize storage for multiple samples
            stan_SP1 = zeros(length(X_test_fin(:, 1)), numChannels, numSamples);

            % Initialize all samples with the same test data
            for i = 1:numSamples
                stan_SP1(:, :, i) = X_test_fin;
            end

            % Predict using all the networks in turn
            for i = 1:numSamples

                % Run autoregressive prediction
                for g = 1:length(GT_Trace(:, 1)) - LKP
                    X_window = stan_SP1(LKP - window_size + g:LKP + g - 1, :, i);

                    YTest = predict(net{i}, X_window);

                    stan_SP1(LKP + g, :, i) = YTest(end, :);
                end
            end

            % Calculate statistics across all samples
            meanPrediction = zeros(length(stan_SP1(:, 1, 1)), numChannels);
            stdPrediction = zeros(length(stan_SP1(:, 1, 1)), numChannels);

            % For each time step and channel, compute mean and std across all samples
            for t = 1:length(stan_SP1(:, 1, 1))
                for c = 1:numChannels
                    % Get all samples for this time point and channel
                    samples = squeeze(stan_SP1(t, c, :).*sigma(c)+mu(c));

                    % Calculate statistics
                    meanPrediction(t, c) = mean(samples);
                    stdPrediction(t, c) = std(samples);
                end
            end

            % This enforces monotonicity on the signal, if you want that...
            mon_pred = enforceMonotonicDecrease(meanPrediction);


            
            % Calculate confidence intervals (95%)
            confInterval95Lower = meanPrediction - 1.96 * stdPrediction;
            confInterval95Upper = meanPrediction + 1.96 * stdPrediction;

            monfInterval95Lower = mon_pred - 1.96 * stdPrediction;
            monfInterval95Upper = mon_pred + 1.96 * stdPrediction;

            %saved_predictions = (stan_SP1(:, 1).*Range(1)+Min(1));
            %saved_predictions2 = (stan_SP2(:, 1).*Range(1)+Min(1));


            % total_predict = saved_predictions(1:sig_L).*(multiplier(end:-1:1)) + saved_predictions2(1:sig_L).*multiplier(1:1:end);


          
            % Plotting for the predictions and confidence intervals. This
            % is currently set up to show RPT capacity and cycling
            % capacity. The second set of plots are for feature 8, which is
            % cycling capacity now, but might change if you change the
            % target feature array.  
           
            nexttile
            hold on


            plot(GT_Trace(:, 1), 'b');
            plot(GT_Trace(:, 8)*1000, 'g');
            plot(LKP:length(meanPrediction(:, 1)),meanPrediction(LKP:end, 1), 'r');
            plot(LKP:length(meanPrediction(:, 1)), confInterval95Upper(LKP:end, 1), 'r--');
            plot(LKP:length(meanPrediction(:, 1)), confInterval95Lower(LKP:end, 1), 'r--');
           
            %plot(LKP:length(meanPrediction(:, 1)), meanPrediction(LKP:end, 1) - meanPrediction(LKP:end, 12), 'y');
            % plot(LKP:length(meanPrediction(:, 1)), confInterval95Upper(LKP:end, 8)*1000, 'y--');
            % plot(LKP:length(meanPrediction(:, 1)), confInterval95Lower(LKP:end, 8)*1000, 'y--');


            plot(LKP:length(meanPrediction(:, 1)), meanPrediction(LKP:end, 8)*1000, 'y');
            plot(LKP:length(meanPrediction(:, 1)), confInterval95Upper(LKP:end, 8)*1000, 'y--');
            plot(LKP:length(meanPrediction(:, 1)), confInterval95Lower(LKP:end, 8)*1000, 'y--');

            % plot(meanPrediction(:, 11)*1000, 'b');
            % plot(confInterval95Upper(:, 11)*1000, 'b--');
            % plot(confInterval95Lower(:, 11)*1000, 'b--');

            plot(LKP*ones(2), [min(meanPrediction(:, 1)) max(meanPrediction(:, 1))], 'mo-')


            xlabel("Total Energy throughput (kWh)")
            ylabel("Capacity (mAh)")
            legend(["Ground Truth Capacity (RPT)" "Ground Truth Capacity (Cycling)" "Predicted RPT Capacity" "RPT Cap Confidence Bounds"...
                "" "Predicted Cycling Capacity" "Cycling Cap Confidence Bounds" "" "Last known point"],Location="southwest")
          
            % Calculations for metrics - cross correlation was functionally
            % pointless, so should change to something else...

            GT_norm = (GT_Trace(LKP:end, 1) - mean(GT_Trace(LKP:end, 1))/std(GT_Trace(LKP:end, 1)));
            Pred_norm = (meanPrediction(LKP:end, 1) - mean(meanPrediction(LKP:end, 1))/std(meanPrediction(LKP:end, 1)));


            gd1(k, p) = max(xcorr(GT_norm, Pred_norm, 'normalized'));
            gd2(k, p) = mean_absolute_percentage_error((GT_Trace(LKP:end, 1)-mu(1))./sigma(1), (meanPrediction(LKP:end, 1)-mu(1))./sigma(1));
            gd3(k, p) = calculate_r_squared(GT_Trace(LKP:end, 1), meanPrediction(LKP:end, 1));
            k
%%
        end
 

    end

    param_goodness(W) = sum(gd1(:, 1))/length(gd1(:, 1));
    param_goodness2(W) = sum(gd2(:, 1))/length(gd2(:, 1));
    param_goodness3(W) = sum(gd3(:, 1))/length(gd3(:, 1));

    W

end


% This is the monotonic prediction function.
function monotonicPredictions = enforceMonotonicDecrease(predictions)
    % For a degradation curve that should be non-increasing
    % with downward shift of remaining values when upticks occur
    monotonicPredictions = predictions;
    
    for i = 2:length(predictions)
        if monotonicPredictions(i) > monotonicPredictions(i-1)
            % Calculate the shift amount needed
            shiftAmount = monotonicPredictions(i) - monotonicPredictions(i-1);
            
            % Apply this shift to current and all remaining points
            monotonicPredictions(i:end) = monotonicPredictions(i:end) - shiftAmount;
        end
    end
end

% This is the duplication function for the biases
function duplicatedStruct = duplicateStruct(originalStruct, duplicateArray)
    % Check input sizes match
    if length(originalStruct) ~= length(duplicateArray)
        error('The struct and array must have the same length');
    end
    
    % Initialize cell array to store duplicated structs
    duplicatedCells = cell(1, length(originalStruct));
    
    % Iterate through each struct element
    for i = 1:length(originalStruct)
        % Replicate the current struct element based on the corresponding array value
        duplicatedCells{i} = repmat(originalStruct(i), 1, duplicateArray(i));
    end
    
    % Combine the cell array into a single struct array
    duplicatedStruct = [duplicatedCells{:}];
end
% Mean absolute percentage error function
function mape = mean_absolute_percentage_error(actual, predicted)
    % Check input sizes match
    if length(actual) ~= length(predicted)
        error('Actual and predicted curves must be the same length');
    end
    
    % Calculate absolute percentage error
    % Add small epsilon to avoid division by zero
    ape = abs((actual - predicted) ./ (actual + eps)) * 100;
    
    % Calculate mean
    mape = mean(ape);
end
% r squared function
function r_squared = calculate_r_squared(y_true, y_pred)
    % Calculate R-squared (coefficient of determination) between predicted and true values
    %
    % Inputs:
    %   y_true - Vector of observed/true values
    %   y_pred - Vector of predicted values from your model
    %
    % Output:
    %   r_squared - The coefficient of determination (R^2)
    
    % Ensure inputs are column vectors
    y_true = y_true(:);
    y_pred = y_pred(:);
    
    % Calculate the mean of the observed values
    y_mean = mean(y_true);
    
    % Calculate total sum of squares (proportional to variance of data)
    SS_total = sum((y_true - y_mean).^2);
    
    % Calculate residual sum of squares
    SS_residual = sum((y_true - y_pred).^2);
    
    % Calculate R-squared
    r_squared = 1 - (SS_residual / SS_total);
    
    % Handle edge case where all true values are identical
    if SS_total == 0
        fprintf('Warning: All true values are identical. R-squared is undefined.\n');
        r_squared = NaN;
    end
end