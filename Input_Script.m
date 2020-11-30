clear
clc
% Taking Problem name and var_level input
fprintf(2,'\n WELCOME! PLEASE FOLLOW THE INSTRUCTIONS BELOW:\n');
problem_name= input('\n Enter a valid problem name: ','s');
problem_choices= {'long_term','short_term','interpolation'};

while ~ismember(lower(problem_name), problem_choices)
  problem_name = input('\n ERROR! Please enter a valid choice: ', 's');
end
problem_name = lower(problem_name);
% Taking var_level input for Gaussian noise
var_level= input('\n Enter a valid var_level of Gaussian noise(0, 5, or 10): ');
var_choices= [0,5,10];

while ~ismember(var_level, var_choices)
  var_level = input('\n ERROR! Please enter a valid choice: ');
end
fprintf(2,'\n THANK YOU. PLEASE WAIT FOR THE RESULTS.\n');

% Loading the selected train and test data
train_data = load(['train_data_',problem_name,'_',num2str(var_level),'_var.mat']).train_data;
test_data = load(['test_data_',problem_name,'_',num2str(var_level),'_var.mat']).test_data;

% Running the model function
pred_pm2d5 = pm2d5_pred_model(train_data, test_data,problem_name,var_level);

%Saving predictions
save([problem_name,'_',num2str(var_level),'.mat'],'pred_pm2d5');
 

% %plotting
% [~,idx] = sort(datenum(test_data.time), 1, 'ascend');
% pred_pm2d5 = pred_pm2d5(idx);
% figure
% plot(sort(test_data.time),pred_pm2d5) 