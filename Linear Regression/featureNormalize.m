function [X_norm, mu, sigma] = featureNormalize(X)

% Initialize some useful values
n = size(X, 2);
m = size(X, 1);
X_norm = X;
mu = zeros(1, n);
sigma = zeros(1, n);

for feature = 1:n
  
  % computing mu and sigma for each feature (each column of X variable)
  mu(feature) = mean(X(:, feature));
  sigma(feature) = std(X(:, feature));
  
  % substituting each sample's value with its normalized value
  for sample = 1:m
    X_norm(sample, feature) = (X(sample, feature) - mu(feature)) / sigma(feature);
  end
  
end

end
