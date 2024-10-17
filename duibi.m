clear all 
clc
 
SearchAgents_no=30; % Number of sEOrch agents
Function_name='F9'; % Name of the test function
Max_iteration=500; % Maximum number of iterations
Run_no=30;         % Number of independent runs 
 
[lb,ub,dim,fobj]=funinfo(Function_name);
 
WOA_scores = zeros(Run_no, 1);
GA_scores = zeros(Run_no, 1);
PSO_scores = zeros(Run_no, 1);
 
 
for run = 1:Run_no
   
    %% WOA 
    [Best_score1,Best_pos1, WOA_cg_curve ] = WOA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
    WOA_scores(run,1) = Best_score1;
 
 
    %% PSO    
    [Best_score2,Best_pos2,PSO_cg_curve] = PSO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj); % Call PSO
    PSO_scores(run,1) = Best_score2;
   
    
   
 
 
    %% GA   
    [Best_score4,Best_pos4, GA_cg_curve ] = GA1(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
    GA_scores(run,1) = Best_score4;
   
    
end
 
fprintf ('Best solution obtained by WOA in run %d: %s\n', run, num2str(Best_pos1,'%e  '));
display(['The best optimal value of the objective function found by WOA in run ', num2str(run), ' for ', Function_name, ' is: ', num2str(Best_score1)]);
 
fprintf ('Best solution obtained by PSO: %s\n', num2str(Best_pos2,'%e  '));
display(['The best optimal value of the objective funciton found by PSO  for ' [num2str(Function_name)],'  is : ', num2str(Best_score2)]);

fprintf ('Best solution obtained by GA in run %d: %s\n', run, num2str(Best_pos4,'%e  '));
display(['The best optimal value of the objective function found by GA in run ', num2str(run), ' for ', Function_name, ' is: ', num2str(Best_score4)]);
 
 
figure;
t = 1:Max_iteration;
semilogy( t, WOA_cg_curve, 'm-p', t, PSO_cg_curve, 'b-d', ...
    t, GA_cg_curve, 'g-s',   ...
    'linewidth', 1.5, 'MarkerSize', 8, 'MarkerIndices', 1:25:Max_iteration);
title(Function_name)
xlabel('Iteration');
ylabel('Best fitness obtained so far');
axis fill
grid on
box on
legend('WOA','PSO','GA');