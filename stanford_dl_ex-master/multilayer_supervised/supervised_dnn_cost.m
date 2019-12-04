function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
m = size(data, 2);
%% forward prop
%%% YOUR CODE HERE %%%
for l = 1:numHidden+1
    if(l == 1) 
        hAct{l}.z = stack{l}.W * data;  %stack{l}.W 256 * 784    data 784 * 10000
    else
        hAct{l}.z = stack{l}.W * hAct{l-1}.a;
    end 
    hAct{l}.z = bsxfun(@plus, hAct{l}.z, stack{l}.b);
    hAct{l}.a = sigmoid(hAct{l}.z);
end

pred_temp = exp(hAct{numHidden+1}.z);
pred_prob = bsxfun(@rdivide, pred_temp, sum(pred_temp, 1));
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
I = sub2ind(size(pred_prob), labels', 1 : size(pred_prob, 2));  
ceCost = -sum(log(pred_prob(I))); 

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
d = zeros(size(pred_prob));
d(I) = 1;
error = pred_prob - d;
for l = numHidden+1: -1 : 1	
    if(l == numHidden+1)
        hAct{l}.delta = error;
    else
        hAct{l}.delta = (stack{l+1}.W' * hAct{l+1}.delta) .* (hAct{l}.a .* (1-hAct{l}.a));
    end
    
	if(l == 1)
		gradStack{l}.W = hAct{l}.delta * data';
	else 
		gradStack{l}.W = hAct{l}.delta * hAct{l-1}.a';
    end
    gradStack{l}.b = sum(hAct{l}.delta,2);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for l = 1:numHidden+1
    wCost = wCost+ sum(sum(stack{l}.W.^2));   %  网络参数W的累计和,正则项损失
end
cost = (1/m)*ceCost + .5 * ei.lambda * wCost; % 带正则项的损失
for l = numHidden+1: -1 : 1
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



