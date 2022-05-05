function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
				 
printDim(X, 'X')
printDim(y, 'y')
printDim(Theta1, 'Theta1')
printDim(Theta2, 'Theta2')

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

J_base = 0;
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

printDim(Delta1, 'Delta1')
printDim(Delta2, 'Delta2')

% remove bias unit weights from the Theta matrices
Theta1_ = Theta1(:, 2:end);
printDim(Theta1_, 'Theta1_')
Theta2_ = Theta2(:, 2:end);
printDim(Theta2_, 'Theta2_')

% iterate over the training examples
for i = 1:m
  % forward propagation: calculate a1, a2 and a3 (=h) for example i
  x_i = X(i, :);
  a1_i = x_i;
  size(a1_i);

  a1_i = [1, a1_i]; % Add a 1 for the bias unit
  z2_i = a1_i * Theta1';
  a2_i = sigmoid(z2_i);
  size(a2_i);

  a2_i = [1, a2_i]; % Add a 1 for the bias unit
  z3_i = a2_i * Theta2';
  a3_i = sigmoid(z3_i);

  h_i = a3_i;
  size(h_i);
  y_i = (1:num_labels);
  y_i = y_i == y(i);
  size(y_i);
  
  % calculate the costs for this training example and add it to the total costs (J_base)
  %J_i = (-y_i'*log(h_i) - (1-y_i)' * log(1-h_i));
  J_i = (-y_i*log(h_i)' - (1-y_i) * log(1-h_i)');
  J_base = J_base + J_i;
  
  % backward propagation: calculate d3 and d2 for example i
  d3_i = a3_i - y_i;
  d2_i = (Theta2_' * d3_i')' .* sigmoidGradient(z2_i);
  
  % accumulate the gradient for example i
  Delta2 = Delta2 + d3_i' * a2_i;
  
  Delta1 = Delta1 + d2_i' * a1_i;

endfor


size(Theta1);
size(Theta1_);

size(Theta2);
size(Theta2_);

J = (J_base / m) + (lambda * (sum(sum(Theta1_ .^ 2)) + sum(sum(Theta2_ .^ 2))) / (2*m) ) ;

% calculate gradient *without* regularization
%Theta1_grad = Delta1 / m;
%Theta2_grad = Delta2 / m;

% calculate gradient *with* regularization
Theta1_grad = (Delta1 + lambda * Theta1) / m; 
Theta1_grad(:,1) = Delta1(:,1) / m;
Theta2_grad = (Delta2 + lambda * Theta2) / m;
Theta2_grad(:,1) = Delta2(:,1) / m; 


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.




%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
