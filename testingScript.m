% TESTING SCRIPT - SCRIPT #2

% generate cluster distance
cluster_infl_range = 4 : 10;
cluster_infl_range = cluster_infl_range/10;

% second script for the test
for i = cluster_infl_range
    fis_name = sprintf('fisFile%.1f', i);
    test_fuzout_name = sprintf('testfuzOut%.1f.csv',i);
    testRMSE_name = sprintf('testRMSE-result%.1f.txt',i);

    fis = readfis(fis_name);

    % reading test data file
    data=readtable('ANFIS_Test_1.csv');
    dataArr=table2array(data);
    valdatain=dataArr(:,2:end);
    valdataout=dataArr(:,1);

    % evaluate our fis model with test data
    valfuzout = evalfis(fis,valdatain);
    valRMSE = norm(valfuzout-valdataout)/sqrt(length(valfuzout));

    % printing out the predictions
    T = array2table(valfuzout);
    writetable(T, test_fuzout_name);

    % print out the RMSE
    fileID=fopen(testRMSE_name,'wt');
    fprintf(fileID, '%e\n', valRMSE);
    fclose(fileID);
end