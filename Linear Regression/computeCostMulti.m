function J = computeCostMulti(X, y, theta)

% here we can benefit from computeCost function, which is independent
% from the length of theta
J = computeCost(X, y, theta);

end
