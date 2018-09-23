function J = computeCost(X, y, theta)

m = length(y); % number of training examples
constant_term = (1/(2*m));

main_term = sum(((X * theta) - y).^2);
J = constant_term * main_term;

end
