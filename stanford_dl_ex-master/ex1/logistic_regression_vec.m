function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  
  m=size(X,2);
  n=size(X,1);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
    hs = zeros(1, m);
    %disp(1 / (1 + exp(-(theta' * X(:,1)))));
    %fprintf("====")
    for i = 1:m   
        hs(i) = 1 / (1 + exp(-(theta' * X(:,i))));  %theta' 1 * n  X(:, i) n * 1
        disp(1 / (1 + exp(-(theta' * X(:,i)))));
        fprintf('%d', i);
        f = f - (y(i) * log(hs(i)) + (1 - y(i)) * log(1 - hs(i)));
    end
    g = (hs - y) * X';     %(hs - y): 1 * m   X': m * n   g: 1 * n
    




