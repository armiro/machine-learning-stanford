function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% here we can benefit from gradientDescent function, which is independent
% from the number of features
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

end
