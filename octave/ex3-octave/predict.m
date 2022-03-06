function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
fprintf('\nDimensions X: %f %f\n', size(X, 1), size(X, 2));
fprintf("\nDimensions Theta1\': %f %f\n", size(Theta1', 1), size(Theta1', 2));

A2 = sigmoid(X * Theta1');

fprintf('\nDimensions A2 - before: %f %f\n', size(A2, 1), size(A2, 2));
% Add ones to the A2 data matrix
A2 = [ones(m, 1) A2];
fprintf('\nDimensions A2 - after: %f %f\n', size(A2, 1), size(A2, 2));
fprintf("\nDimensions Theta2\': %f %f\n", size(Theta2', 1), size(Theta2', 2));
A3 = sigmoid(A2 * Theta2');

fprintf('\nDimensions A3: %f %f\n', size(A3, 1), size(A3, 2));
[~,p] = max(A3, [], 2);
fprintf('\nDimensions p: %f %f\n', size(p, 1), size(p, 2));

p;

% =========================================================================

end
