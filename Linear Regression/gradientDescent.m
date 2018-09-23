function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

constant_term = alpha/m; % constant term in computing derivative term

for iter = 1:num_iters

    % defining initial terms which need to be cleared on every iteration
    temp = zeros(length(theta), 1);
    
    % assign values to all the elements of theta vector
    for theta_element = 1:length(theta)
      
      main_term = 0;
      % we need to perform on all data records (batch gradient descent)
      for i = 1:m
        sigma_term = ((X(i, :) * theta) - y(i)) * X(i, theta_element);
        main_term = main_term + sigma_term;
      end

      derivative_term = constant_term * main_term;
      % store each theta element in temporal value
      temp(theta_element) = theta(theta_element) - derivative_term;
    end
    
    % assign back the new theta values stored in temporal matrix 'temp'
    for theta_element = 1:length(theta)
      theta(theta_element) = temp(theta_element);
    end

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end