function pred_pm2d5 = pm2d5_pred_model(train_data, test_data, problem_name, var_level)
  
    % Downsampling the train data into minutely average
    tt = table2timetable(train_data);
    train_data_minutely = retime(tt,'minutely','mean');
    train_data_minutely = fillmissing(train_data_minutely,'linear');
    Y_train= train_data_minutely.pm2d5;
    Y_train= Y_train(1:end-1,:);
    X_train= [juliandate(train_data_minutely.time), train_data_minutely.hmd, train_data_minutely.spd, train_data_minutely.tmp, train_data_minutely.lat, train_data_minutely.lon];
    X_train= X_train(1:end-1,:); 
    N_days = length(Y_train)/1440;
    
    
    %normalize X features of train data
    for ii=1:6
        X_train(:,ii)= normalize(X_train(:,ii),'range');
    end
   

    %Preprocessing Y_train to remove any Gaussian noise
    if var_level == 5
        Y_train= imgaussfilt(Y_train, 0.5*std(Y_train));
    elseif var_level == 10
        Y_train= imgaussfilt(Y_train, 1*std(Y_train));
    end
    
    % Reshaping the Y_train matrix into daily values for SVD analysis
    daily_matrix=reshape(Y_train,[1440,N_days])';
 
    %SVD to remove any residual noises
    [U,S,V] = svd(daily_matrix,'econ');

    % Daily matrix reconstruction with SVD and explained ratios
    singular_vals = diag(S);
    V_T=V';
    explained_ratio=0;
    n=0;
    while explained_ratio< 0.98
        n=n+1; 
        explained_ratio = sum(singular_vals(1:n).^2)/sum(singular_vals.^2);
    end
    Re_dailymatrix = U(:,1:n)*S(1:n,1:n)*V_T(1:n,:);
    
    Y_1=[];
    for ii= 1:N_days
        Y_1= cat(2,Y_1, Re_dailymatrix(ii,:));
    end
  
    Y_train= Y_1'; % Y_train now free of any Gaussian and Residual noises

    %Test Data precprocessing
    test_data = fillmissing(test_data,'linear');
    X_test= [juliandate(test_data.time), test_data.hmd, test_data.spd, test_data.tmp, test_data.lat, test_data.lon];

    %normalize X features of test data
    for ii=1:6
        X_test(:,ii)= normalize(X_test(:,ii),'range');
    end
    
    % Determine predicted Y values by either Gaussian method or Ridge
    % Regression based on the problem type
    if ismember(problem_name, {'long_term','short_term'})
        grpmdl = fitrgp(X_train,Y_train);
        y_pred = predict(grpmdl,X_test);
    else
        lambda =0.1;
        dim = size(X_train,2);
        beta = inv((X_train'*X_train)+lambda*eye(dim))*X_train'*Y_train;
        y_pred = X_test*beta;
    end
        
    % Zeroing out the negative values in the final return pred_pm2d5 array
    pred_pm2d5= (abs(y_pred)+y_pred)./2; 
    

end
