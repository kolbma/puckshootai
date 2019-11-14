function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  mu = mean(X);
  X_norm -= mu;
  sigma = std(X);
  X_norm = X_norm ./ sigma;
end
