function plotDecisionBoundaryReg(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

ezplot (@(X1, X2) theta(1) + theta(1)*X1 + theta(3)*X2 +  theta(4)*X1.^2 + theta(5)*X2.^2 + theta(6)*X1.*X2 + theta(7)*X1.*X2.^2 + theta(8) * X2.*X1.^2);

hold off

end
