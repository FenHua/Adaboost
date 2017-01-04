%% 基于Adaboost 实现目标的分类
%step 1: reading data from the file
file_data=load('Ionosphere.txt');
Data=file_data(:,1:end-1)';
Labels=file_data(:,end)';
Labels=Labels*2-1;
MaxIter=100;%boosting iterations
%step 2:splitting data to training and control set
TrainData=Data(:,1:2:end);
TrainLabels=Labels(1:2:end);
ControlData=Data(:,2:2:end);
ControlLabels=Labels(2:2:end);
%step 3:constructing weak learner
weak_learner=tree_node_w(3);% pass the number of tree splits to the constructor
%step 4:training with Gentle AdaBoost
[RLearners RWeights]=RealAdaBoost(weak_learner,TrainData,TrainLabels,MaxIter);
%step 5:evaluating on control set
ResultR=sign(Classify(RLearners,RWeights,ControlData));
%step 6:calculating error
ErrorR=sum(ControlLabels~=ResultR)/length(ControlLabels);