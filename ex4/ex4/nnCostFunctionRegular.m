function [J grad] = nnCostFunctionRegular(nn_params, ...
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
A1 = [ones(m, 1) X];   %5000x401
Z2 = A1 * Theta1';     %5000x25
A2 = sigmoid(Z2);     %5000x25
A2 = [ones(m, 1) A2]; %5000x26
Z3 = A2 * Theta2';    %5000x10
A3 = sigmoid(Z3);     %5000x10

Y = zeros(m, num_labels); % 5000x10
for i = 1 : m
	Y(i, y(i)) = 1;
end

Jm = (Y .* log(A3) + (1-Y) .* log(1-A3));  %5000x10

J = -1/m * sum(sum(Jm));


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
delta3 = A3 - Y; %5000x10

%        5000x10   10x(26-1)                    5000x25    
delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(Z2); %5000x25
Delta2 = delta3' * A2;      %10x5000  5000x26 -> 10x26
Delta1 = delta2' * A1;      %25x5000  5000x401 -> 25x401
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;
% Part 3: Implement regularization with the cost function and gradients.

%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
londa = 1000
tmpTheta1 = Theta1;
tmpTheta2 = Theta2;
tmpTheta1(:,1) = 0; %no regularation for bias
tmpTheta2(:,2) = 0; %no regularation for bias
Theta1_grad = Delta1 / m + londa/m * tmpTheta1; %25x401
Theta2_grad = Delta2 / m + londa/m * tmpTheta2; %25x401h


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
