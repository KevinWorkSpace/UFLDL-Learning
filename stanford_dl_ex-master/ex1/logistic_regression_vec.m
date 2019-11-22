function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  %m=size(X,2);
  
  % initialize objective value and gradient.
  %f = 0;
  %g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
   %f = 0.5 .* (theta' * X - y) * (theta' * X - y)';
   %g = X * (theta' * X - y)';   %X: n * m, (theta' * X - y)': m * 1,  g: 1 * n
   f = 0.5 .* (theta' * X - y) * (theta' * X - y)';
% g = g';
   g = (theta' * X - y) * X';
   g = g';


