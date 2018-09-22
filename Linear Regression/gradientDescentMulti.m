function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples

% first we need to normalize 'y' values
mu_y = mean(y(:));
sigma_y = std(y(:));
for sample = 1:m
  y(sample) = (y(sample) - mu_y) / sigma_y;
end

% here we can benefit from gradientDescent function, which is independent
% from the number of features
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);


end
