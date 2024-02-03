% TRAINING & MODEL GENERATION SCRIPT - SCRIPT #1

% Split training and test data using prebuilt csv
data=readtable('ANFIS_Train_1.csv');
dataArr=table2array(data);
dataout=dataArr(:,1);
datain=dataArr(:,2:end);

% running subtractive clustering and generate the fis model
cluster_infl_range = 4 : 10;
cluster_infl_range = cluster_infl_range/10;

for i = cluster_infl_range
    
    % ANFIS model parameters
    epoch = 20;
    % 
    
    fis_name = sprintf('fisFile%.1f', i);
    fuzout_name = sprintf('fuzOut%.1f.csv',i);
    trainRMSE_name = sprintf('trainRMSE-result%.1f.txt',i);

    fisOpt = genfisOptions("SubtractiveClustering",...
    "ClusterInfluenceRange",i);
    fis = genfis(datain,dataout,fisOpt);
   
    % ANFIS model
    fis = anfis([datain,dataout],fis,epoch);
    % 

    % print out the fis model
    writeFIS(fis,fis_name)

    % evaluate the fis model
    fuzout = evalfis(fis,datain);

    % printing out the predictions
    T = array2table(fuzout);
    writetable(T, fuzout_name);
    
    % calculate RMSE
    trnRMSE = norm(fuzout-dataout)/sqrt(length(fuzout));

    % print out the RMSE
    fileID=fopen(trainRMSE_name,'wt');
    fprintf(fileID, '%e\n', trnRMSE);
    fclose(fileID);

end