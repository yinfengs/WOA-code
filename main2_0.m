
warning off
close all
clear
clc


dataLength = 400;%

load('data1.mat');
res = [[input_train'; input_test'], [output_train'; output_test']];
res(:, end) = round(res(:, end));



% SMOTE
N = 50;                      
minority_class = [1,5];           
k = 100;                      
res = smote(res, k, minority_class, N);


windowSize = 50;             
stepSize = 25;                 
[res, dataLength] = smallWindowSegmentation(res, windowSize, stepSize);

num_class = length(unique(res(:, end)));   
num_dim = size(res, 2) - 1;                
num_size = 0.9;                            



temp = randperm(dataLength);

trainingSet = dataLength * num_size;%
P_train = res(temp(1 : trainingSet), 1 : num_dim)';
T_train = res(temp(1 : trainingSet), num_dim + 1)';%
M = size(P_train, 2);%

P_test = res(temp(trainingSet + 1 : end), 1 : num_dim)';%
T_test = res(temp(trainingSet + 1 : end), num_dim + 1)';%
N = size(P_test, 2);%

%% 
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);%

t_train = categorical(T_train)';
t_test = categorical(T_test)';%


P_train = double(reshape(P_train, num_dim, 1, 1, M));
P_test = double(reshape(P_test, num_dim, 1, 1, N));


for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1} = P_test(:, :, 1, i);
end


SearchAgents_no = 8;                      
Max_iteration = 5;                         
dim = 3;                                   
lb = [1e-4, 10, 1e-5];                     
ub = [1e-1, 30, 1e-2];                    
fitness = @(x)fical(x, num_dim, num_class, p_train, t_train, T_train);

% [Best_score, Best_pos, curve] = WOA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness);
% [Best_score, Best_pos, curve] = GA1(SearchAgents_no, Max_iteration, lb, ub, dim, fitness);
[Best_score, Best_pos, curve] = PSO(SearchAgents_no, Max_iteration, lb, ub, dim, fitness);
Best_pos(1, 2) = round(Best_pos(1, 2));
best_hd = Best_pos(1, 2)                
best_lr = Best_pos(1, 1)                
best_l2 = Best_pos(1, 3)                

lgraph = layerGraph();                                                    
tempLayers = [
    sequenceInputLayer([num_dim, 1, 1], 'Name', 'sequence'),               
    sequenceFoldingLayer('Name', 'seqfold')                                
];
lgraph = addLayers(lgraph, tempLayers);                                 

tempLayers = [
    convolution2dLayer([2, 1], 16, 'Name', 'conv_1', 'Padding', 'same')    
    reluLayer('Name', 'relu_1')                                            
    convolution2dLayer([2, 1], 32, 'Name', 'conv_2', 'Padding', 'same')    
    reluLayer('Name', 'relu_2')                                            
];
lgraph = addLayers(lgraph, tempLayers);                                    

tempLayers = [
    sequenceUnfoldingLayer('Name', 'sequnfold')                            
    flattenLayer('Name', 'flatten')                                        
    gruLayer(best_hd, 'Name', 'gru', 'OutputMode', 'last')                 
    fullyConnectedLayer(num_class, 'Name', 'fc')                          
    softmaxLayer('Name', 'softmax')                                        
    classificationLayer('Name', 'classification')                          
];
lgraph = addLayers(lgraph, tempLayers);                                   


lgraph = connectLayers(lgraph, 'seqfold/out', 'conv_1');                  
lgraph = connectLayers(lgraph, 'seqfold/miniBatchSize', 'sequnfold/miniBatchSize'); 
                                                                          
lgraph = connectLayers(lgraph, 'relu_2', 'sequnfold/in');                


options = trainingOptions('adam', ...        
    'MaxEpochs', 300, ...                   
    'MiniBatchSize', 400, ...                
    'InitialLearnRate', best_lr, ...        
    'L2Regularization', best_l2, ...         
    'LearnRateSchedule', 'piecewise', ...   
    'LearnRateDropFactor', 0.1, ...         
    'LearnRateDropPeriod', 200, ...          
    'Shuffle', 'every-epoch', ...            
    'ValidationPatience', Inf, ...          
    'Plots', 'training-progress', ...       
    'Verbose', false);


[net, info] = trainNetwork(p_train, t_train, lgraph, options);


t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test);


T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');


error1 = sum((T_sim1 == T_train)) / M * 100;
error2 = sum((T_sim2 == T_test)) / N * 00;

figure
plot(curve, 'linewidth', 1.5);
title('WOA')
xlabel('The number of iterations')
ylabel('Fitness')

analyzeNetwork(lgraph)


[T_train, index_1] = sort(T_train);
[T_test, index_2] = sort(T_test);

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);


figure 
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';