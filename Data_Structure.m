%% Feature Generation and organisation.
close all
clear all


% Initialise the raw data struct.


raw_data = struct();


% READ ME: I've effectively put all the experimental data into the same folder and
% titled it the same with A-P. This included the slightly silly step of
% changing all the titles to 'Expt 4', even though half of them are
% actually expt 5. A more sensible approach would be to literally just
% title them with a letter or something. Regardless, the label array is how
% individual data sets are called. 
% The temperature array and cycle type I manually enter. This just requires
% knowledge of the experiments you're using. 

label = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"];
Temperature = [10, 10, 10, 25, 25, 40, 40, 40, 10, 10, 10 25, 25, 40, 40, 40]';

% 1 is drive cycle, 2 is 100% CC cycling
Cycle_Type = [1, 1, 1, 1, 1, 1, 1, 1, 2,  2, 2, 2, 2, 2, 2, 2];

% This is the Watt hour granularity you're setting. It determines how many
% point you have, but more points means you'll be inferring more between RPTs, so
% this is a balance. Cycling data is high enough fidelity that it's not
% affected in quality though.
Point_x_diff = 1000;

    num_batteries = length(label);

% Loop through all of the batteries...
for j = 1:1:num_batteries
 

    % Calling the cycle data and putting it into struct (A) and the RPT
    % data goes into (B)
    FileName = strcat("/Users/michael/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ME4/FYP/Datasets/kirkaldy/Expt 4 - Drive Cycle Aging (Control)/Summary Data/Ageing Sets Summary/Summary per Cycle/expt 4 - cell ", label(j), " - cycle_data.csv");
    FileName2 = strcat("/Users/michael/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ME4/FYP/Datasets/kirkaldy/Expt 4 - Drive Cycle Aging (Control)/Summary Data/Performance Summary/Expt 4 - cell ", label(j), " (", string(Temperature(j)), "degC) - Processed Data.csv");
    A = importdata(FileName);
    B = importdata(FileName2);

    % We need to create an x axis so gradients can be compared properly
    % between different data sets

    num_RPT = length(B.data(:, 1));

    RPT_x = B.data(:, 4);
    Points_x = 0:Point_x_diff:max(RPT_x);
    Tot_points = length(Points_x);

    % Interping all the data features onto the new x axis. I'm using linear
    % rather than spline as there's a risk with spline that you add
    % information
    
    Cap = interp1(RPT_x, B.data(:, 5), Points_x, "spline", "extrap");
    PE_Cap =  interp1(RPT_x, B.data(:, 9), Points_x, "spline", "extrap");
    NE_Cap =  interp1(RPT_x, B.data(:, 10), Points_x, "spline", "extrap");
    LAM  =  interp1(RPT_x, B.data(:, 15), Points_x, "spline", "extrap");
    LLI = interp1(RPT_x, B.data(:, 19), Points_x, "spline", "extrap"); 
    Electrode_offset = interp1(RPT_x, B.data(:, 13), Points_x, "spline", "extrap");

    % The function `extract_RPT_features' actually just finds the gradient
    % of a signal with respect to another signal, and requires you to give
    % it the number of points you want back... This requires losing a point
    % off the end, you should probably take the gradient from around a
    % point, rather than between two points in future.
    Cap_Grad = extract_RPT_features(Points_x, Cap, Tot_points-1);
    PE_Cap_Grad = extract_RPT_features(Points_x, PE_Cap, Tot_points-1);
    NE_Cap_Grad = extract_RPT_features(Points_x, NE_Cap, Tot_points-1);
    LLI_Grad = extract_RPT_features(Points_x, LLI, Tot_points-1);
    LAM_Grad = extract_RPT_features(Points_x, LAM, Tot_points-1);

    %% Cycling features


     % Cycling data features
     % This is just going through the cycling data and picking out
     % features. The cycling data is aggressively cleaned first, as we're
     % only interested in overall trends at this fidelity. If we were
     % looking at only cycling data you'd want to smooth it less.

    Fit_Data = cleandata(A.data(:, :));  
    Fit_tot_en_through = zeros(length(Fit_Data(:, 1)), 1);
    Fit_tot_en_through(1) = Fit_Data(1, 2)*Fit_Data(1, 7) + Fit_Data(1, 3)*Fit_Data(1, 12);
    for i = 2:1:length(Fit_tot_en_through)
        Fit_tot_en_through(i) = Fit_tot_en_through(i-1) + Fit_Data(i, 7)*Fit_Data(i, 2) + Fit_Data(i, 3)*Fit_Data(i, 12);
    end  
    Fit_Data_x = round(Fit_tot_en_through, 4); 

    % The smoothdatas are a choice on smoothness of your cycling data you
    % sample from - with a large granularity like this you want it to be
    % pretty smooth, but with higher fidelity final points you could maybe
    % capture more by making the data less buttery. You may notice I start
    % the smoothing at point 3, this is to avoid the couple of weird spurious
    % points some of the traces start with, that cause some issues. 
    % Have a play with the types of smoothing available.
    
    Fit_Data_Temp = Fit_Data(:, 6);
    Fit_Data_Cap = Fit_Data(:, 3);
    Fit_Data_Volt_dis = Fit_Data(:, 7);
    Fit_Data_Volt_cha = Fit_Data(:, 12);
    extra_clean_Fit = cleandata([Fit_Data_x, Fit_Data_Cap]);
    ECF_Volt_dis = cleandata([Fit_Data_x, Fit_Data_Volt_dis]);
    ECF_Volt_cha = cleandata([Fit_Data_x, Fit_Data_Volt_cha]);
    ECF_Temp = cleandata([Fit_Data_x, Fit_Data_Temp]);
    Fit_Data_x = extra_clean_Fit(3:end, 1);
    Fit_Data_Cap = smoothdata(extra_clean_Fit(3:end, 2), "sgolay", 300);
    %Fit_Data_Cap = movmean(extra_clean_Fit(3:end, 2), 500, "Endpoints","shrink");
    Fit_Data_Volt_dis = smoothdata(ECF_Volt_dis(3:end, 2), "sgolay", 300);
    Fit_Data_Volt_cha = smoothdata(ECF_Volt_cha(3:end, 2), "sgolay", 300);
    Fit_Data_Temp = smoothdata(ECF_Temp(3:end, 2), "sgolay", 300);
   
    dv = zeros(length(Fit_Data_Volt_cha), 1);
    dq = zeros(length(Fit_Data_Volt_cha), 1);
    dq_dv = zeros(length(Fit_Data_Volt_cha), 1);

    for i = 2:1:length(Fit_Data_Volt_dis)
        dv(i) = (Fit_Data_Volt_dis(i) - Fit_Data_Volt_dis(i-1))/(Fit_Data_x(i) - Fit_Data_x(i-1));
        dq(i) = (Fit_Data_Cap(i) - Fit_Data_Cap(i-1))/(Fit_Data_x(i) - Fit_Data_x(i-1));
        dq_dv(i) = dq(i)* dv(i);
    end



    % Making cycling features per RPT
    % The data is already smoothed, so we can sample the smoothed curves
    % rather than find averages or interp or anything for our cycling
    % features
   

    av_en_through = zeros(length(Points_x), 1);
    av_cap = zeros(length(Points_x), 1);
    cap_varience = zeros(length(Points_x), 1);
    av_volt_dis = zeros(length(Points_x), 1);
    av_volt_cha = zeros(length(Points_x), 1);
    av_dq = zeros(length(Points_x), 1);
    av_dv = zeros(length(Points_x), 1);
    av_dq_dv = zeros(length(Points_x), 1);
    av_temp = zeros(length(Points_x), 1);

    % --------------- This is out of data, see bottom
    % Setting the first point for all the data. I'm using the fifth cycle as the
    % first point because some of the cycle data has spurious data points at the start.
    % The rest will be set basically
    % using a combination of a counter and checking the energy through
    % for the cycling data, when the energy through passes a multiple of
    % 1000 watt hours we take a sample. This could be done way easier using
    % the interp function now that I think about it, so whoever reads this
    % might want to make that change... I dont think it's super important
    % though. 

    % Update - I added the interp lol, it was marginally important


    mark = Point_x_diff;
    RPT_reg = [];
    c = 1;

    type = 1;

    if type == 1

        av_cap = interp1(Fit_Data_x,Fit_Data_Cap, Points_x, "linear", "extrap");
        av_volt_dis = interp1(Fit_Data_x,Fit_Data_Volt_dis, Points_x, "linear", "extrap");
        av_volt_cha = interp1(Fit_Data_x,Fit_Data_Volt_cha, Points_x, "linear", "extrap");
        av_dq = interp1(Fit_Data_x, dq, Points_x, "linear", "extrap");
        av_dv = interp1(Fit_Data_x, dv, Points_x, "linear", "extrap");
        av_dq_dv = interp1(Fit_Data_x, dq_dv, Points_x, "linear", "extrap");
        av_temp = interp1(Fit_Data_x, Fit_Data_Temp, Points_x, "linear", "extrap");
        av_en_through = 0:1000:length(Points_x(1, :))*1000;


         for i = 1:1:length(Fit_Data_x)-1
            if Fit_Data_x(i+1) > mark*c && Fit_Data_x(i) < mark*c
                RPT_reg = i;
                cap_varience(c) = max(Fit_Data_Cap(RPT_reg - 20:RPT_reg)) - min(Fit_Data_Cap(RPT_reg -20:RPT_reg));
                         c = c+1;
            end
         end
         cap_varience(c) = max(Fit_Data_Cap(RPT_reg(end):end)) - min(Fit_Data_Cap(RPT_reg(end):end));
   
    else
    

        av_en_through(c) = 0;%Fit_Data_x(5);
        av_volt_dis(c) = Fit_Data_Volt_dis(5);
        av_volt_cha(c) = Fit_Data_Volt_cha(5);
        av_dq(c) = dq(5);
        av_dv(c) = dv(5);
        av_dq_dv(c) = dq_dv(5);
        av_temp(c) = Fit_Data_Temp(5);
        av_cap(c) = Fit_Data_Cap(5);
    
        for i = 1:1:length(Fit_Data_x)-1
            if Fit_Data_x(i+1) > mark*c && Fit_Data_x(i) < mark*c
                RPT_reg(end + 1) = i;
                
                av_en_through(c+1) = 1000*c;%Fit_Data_x(i);
                av_cap(c+1) = Fit_Data_Cap(i);
                cap_varience(c) = max(extra_clean_Fit(RPT_reg(end) - 20:RPT_reg(end)),2) - min(extra_clean_Fit(RPT_reg(end) - 20:RPT_reg(end)),2);
                av_volt_dis(c+1) = Fit_Data_Volt_dis(i);
                av_volt_cha(c+1) = Fit_Data_Volt_cha(i);
                av_dq(c+1) = dq(i);
                av_dv(c+1) = dv(i);
                av_dq_dv(c+1) = dq_dv(i);
                av_temp(c+1) = Fit_Data_Temp(i);
                c = c+1;
            end
        end
        
        cap_varience(c) = max(Fit_Data_Cap(RPT_reg(end):end)) - min(Fit_Data_Cap(RPT_reg(end):end));

    end
    % Constructing the raw data struct. There will be a full struct entry
    % in raw data for each battery. This will be split into Features (like capacity)
    % and Classifiers (like cycling type).

    % One may notice the shortening of all of the data features by a point,
    % this is because the gradients are one point shorter than the
    % non-gradient features. If you took the gradients around a point
    % rather than between points this wouldn't be an issue, so defo make
    % that change to limit data loss. In general you should bring the data
    % as much in line with itself as poss.

    for i = 1:1:length(Points_x)-1

        % Target Variable
        raw_data(j).Features(i, 1) = Cap(i);
        raw_data(j).Features(i, 2) = Cap_Grad(i);
        % PE_Cap
        raw_data(j).Features(i, 3) = PE_Cap(i);
        raw_data(j).Features(i, 4) = PE_Cap_Grad(i);
        % NE_Cap
        raw_data(j).Features(i, 5) = NE_Cap(i);
        raw_data(j).Features(i, 6) = NE_Cap_Grad(i);
        % Electrode offset
        raw_data(j).Features(i, 7) = Electrode_offset(i);
        % LAM
        raw_data(j).Features(i, 8) = LAM(i);
        raw_data(j).Features(i, 9) = LAM_Grad(i);
        % % LLI
        raw_data(j).Features(i, 10) = LLI(i);
        raw_data(j).Features(i, 11) = LLI_Grad(i);
        % Average cap data across RPT set
        raw_data(j).Features(i, 12) = av_cap(i);
        raw_data(j).Features(i, 13) = cap_varience(i);
         % Average voltage data across RPT set
        raw_data(j).Features(i, 14) = av_volt_dis(i);
        raw_data(j).Features(i, 15) = av_volt_cha(i);
        % Difference between RPT and Cycling data
        raw_data(j).Features(i, 16) = Cap(i) - (av_cap(i)*1000);
        % dV, dQ, dV*dQ
        raw_data(j).Features(i, 17) = av_dq(i+1);
        raw_data(j).Features(i, 18) = av_dv(i+1);
        raw_data(j).Features(i, 19) = av_dq_dv(i+1);
        % temperature
        raw_data(j).Features(i, 20) = av_temp(i);
        % Energy through
        raw_data(j).Features(i, 21) = av_en_through(i);
        % Sequence Counter
         raw_data(j).Features(i, 22) = i-1;

        
    end
    raw_data(j).Classifiers(1) = Temperature(j);
    raw_data(j).Classifiers(2) = Cycle_Type(j);
    
end

% README!!; This next for loop basically duplicates the data adding a small random
% multiplier. It's just bootstrapping to increase the volume of data to train with.
% It's really important that when testing you delete the data bootstrapped
% off the test data. This is hardcoded in the model code, but making a note
% of it here as well.

for j = num_batteries+1:1:2*num_batteries


    FileName = strcat("/Users/michael/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ME4/FYP/Datasets/kirkaldy/Expt 4 - Drive Cycle Aging (Control)/Summary Data/Ageing Sets Summary/Summary per Cycle/expt 4 - cell ", label(j - num_batteries), " - cycle_data.csv");
    FileName2 = strcat("/Users/michael/Library/CloudStorage/OneDrive-ImperialCollegeLondon/ME4/FYP/Datasets/kirkaldy/Expt 4 - Drive Cycle Aging (Control)/Summary Data/Performance Summary/Expt 4 - cell ", label(j - num_batteries), " (", string(Temperature(j - num_batteries)), "degC) - Processed Data.csv");
    A = importdata(FileName);
    B = importdata(FileName2);


    num_RPT = length(B.data(:, 1));

    RPT_x = B.data(:, 4);
    Points_x = 0:Point_x_diff:max(RPT_x);
    Tot_points = length(Points_x);


    mult = (rand()-0.5)*0.03 + 1;

    for i = 1:1:length(Points_x)-1

        raw_data(j).Features(i, :) = raw_data(j - num_batteries).Features(i, :)*mult;

    end

    raw_data(j).Classifiers(1) = Temperature(j - num_batteries);
    raw_data(j).Classifiers(2) = Cycle_Type(j- num_batteries);

end

% This function finds the gradient between two points. It's a mediocre
% function, I'd recommend writing a new one if you find the time.

function extracted_features = extract_RPT_features(X_Data, Y_Data, N_Features)

    extracted_features = zeros(1, N_Features);

    if length(Y_Data) > 1

        %extracted_features(end) = (Y_Data(end) - Y_Data(end-1))/(X_Data(end) - X_Data(end-1));
    
        for i = 2:1:N_Features+1
    
               extracted_features(i-1) = (Y_Data(i) - Y_Data(i-1))/(X_Data(i) - X_Data(i-1));
    
        end
    
    else

        extracted_features(1) = nan;
    
    end


end

% This function removes nan values, major outliers, duplicate points,
% generally just makes the data more legible. Adjust as necessary for how
% you want to treat the data. 

function DataWin = cleandata(DataRough)




%% Data Cleaning
% removal of NaN values  (here using forward imputation)
% and removal of repeat time values

DataRough(any(isnan(DataRough), 2), :) = [];

[~, uniqueIdx] = unique(DataRough(:,1), 'first');

% Keep only the first occurrence of each time value
DataRough = DataRough(sort(uniqueIdx), :);



Data = zeros(size(DataRough));

for i = 1:1:length(DataRough(:, 1))
    
    for j = 1:1:length(DataRough(1, :))

        if isnan(DataRough(i, j)) == true

            Data(i, j) = Data(i-1, j);

        else

            Data(i, j) = DataRough(i, j);

        end
        % 
        % if Data(i, j) == 0
        % 
        %     Data(i, j) = 0.001;
        % 
        % end

    end

end


% Can see there's lots of outliers - detect and fill the outliers (set to
% interpolated value scaled median devaitions) - this is a 1 dimensional winnowing method - we can check
% outliers for correlations later.

DataWin = zeros(size(DataRough));

for j =  1:1:length(Data(1, :))


   % DataWin(:, j) = filloutliers(Data(:, j), "linear", "median");
    DataWin(:, j) = filloutliers(Data(:, j),"linear","movmedian",10);

end


end