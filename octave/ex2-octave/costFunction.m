function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% ======== Implement J 

h = sigmoid(X*theta);

% for debugging purposes
% size(y)
% size(y')
% size(h)

% component based implementation
% part1 = -y'*log(h);
% part2 = (1-y)';
% part3 = log(1-h);ex
% J = (part1 - part2 * part3) ./ m;

% single line implementation
J = (-y'*log(h) - (1-y)' * log(1-h)) / m;

% ======== Implement gradient

grad = X' * (sigmoid(X*theta)-y) / m;

% =============================================================

end
