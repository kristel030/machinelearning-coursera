function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% ======== Implement J 

% helpers
h = sigmoid(X*theta);
%size(theta)
theta_without0 = theta(2:length(theta), :);
%size(theta_without0)

% compute J in two parts
base_J = (-y'*log(h) - (1-y)' * log(1-h)) / m;
%size(base_J)
regularized_J = lambda / (2*m) * theta_without0' * theta_without0;
%size(regularized_J)
J = base_J + regularized_J;

% ======== Implement gradient
% compuate gradient in two parts
%base_grad = X' * (sigmoid(X*theta)-y) / m;

base_grad = X' * (sigmoid(X*theta)-y) / m;
regularized_grad = (lambda/m) * theta;
grad = base_grad + regularized_grad;

% X = [ones(m, 1), data(:,1)]
grad0 = sum (h-y) / m;
grad(1,1) = grad0;

% =============================================================

end
