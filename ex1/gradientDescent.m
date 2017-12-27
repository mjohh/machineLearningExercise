function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % derviation for theta1 and theta1
    %derv_theta1 = 0
		%derv_theta2 = 0
		
		%for i = 1:m
    %	derv_theta1 = derv_theta1 + theta(1) + theta(2)*x(i) - y(i);
		%	derv_theta2 = derv_theta2 + (theta(1) + theta(2)*x(i) - y(i))*x(i);
		%end
		%derv_theta1 = derv_theta1/m;
		%derv_theta2 = derv_theta2/m;
		
		%update thetas
		%theta(1) = theta(1) - alpha*derv_theta1;
		%theta(2) = theta(2) - alpha*derv_theta2;
		
		
		theta = theta - X' * (X * theta - y) * 1/m * alpha;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
