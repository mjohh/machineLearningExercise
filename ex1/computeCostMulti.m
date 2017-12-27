function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

x1 = X(:,2);
x2 = X(:,3);

for i = 1:m
	J = J + (theta(1) + theta(2)*x1(i) + theta(3)*x2(i) - y(i))^2;
end

J = J/(2*m);


% =========================================================================

end
