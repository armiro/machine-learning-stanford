function [X_norm, mu, sigma] = featureNormalize(X)

% Initialize some useful values
n = size(X, 2);
%m = size(X, 1);
X_norm = X;
mu = zeros(1, n);
sigma = zeros(1, n);

mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ std(X);

end
