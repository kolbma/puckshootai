function [theta, J_history] = puckshootai(alpha, num_iters)
  
  X = [ 0.1; 0.15; 0.2; 0.3; 0.4; 0.42; 0.43; 0.44; 0.45; 0.48; 0.5; 0.51; 0.52; 0.54; 0.58; 0.585; 0.59; 0.595; 0.6; 0.605; 0.61; 0.62; 0.63; 0.65; 0.68; 0.7; 0.8; 0.9; 0.95; 1.0 ];
  %X = [ -0.1; -0.15; -0.2; -0.3; -0.4; -0.5; -0.59; 0.6; 0.61; 0.7; 0.8; 0.9; 0.95; 1.0 ];
  y = [   0;    0;   0;   0;   0;    0;    0;    0;    0;    0;   0;    0;    0;    0;    0;     0;    0;     0;   1;     1;    1;    1;    1;    1;    1;   1;   1;   1;    1;   1 ];
  m = length(y);
  X -= 0.6;
  [X_norm, mu, sigma] = featureNormalize(X);
  X_ext = [ones(m, 1), X_norm];
  theta = zeros(2, 1);
  
  figure 1;
  clf;
  subplot(2,2,1);
  hold on;
  plot(X, y, "rx");

  sigmoidGraph(X, y, 0.0, 'Training');

  J_history = zeros(num_iters, 1);
  h_history = zeros(num_iters, m);
  
  
  for iter = 1:num_iters
    h = sigmoid(X_ext * theta);
    %h = 1 ./ (1 + exp(-((X)*theta)));
    %h = X*theta;
    %disp(h);
    
    theta = theta - (( alpha / m ) * X_ext' * (h - y));   
    %disp(theta);
   
    J_history(iter) = 1/m * (-(y') * log(h) - (1-y)' * log(1-h));
  endfor

  subplot(2,2,2);
  plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
  xlabel('Number of iterations');
  ylabel('Cost J');
  
  printf('Theta computed from gradient descent: \n');
  printf(' %f \n', theta);
  printf('mu: %f\nsigma: %f\n', mu, sigma);

  subplot(2,2,3);
  hold on;

  X_ext = [ones(m, 1), ((X - mu) ./ sigma)];
  Predict = sigmoid(X_ext * theta);
  disp([X, Predict, (Predict >= 0.5)]);
  plot(X, (Predict >= 0.5), 'rx');

  sigmoidGraph(X, (Predict >= 0.5), 0.0, 'Prediction');

  subplot(2,2,4);
  hold on;

  X_new = (0:0.01:1)';
  X_ext = [ones(length(X_new), 1), ((X_new - 0.6 - mu) ./ sigma)];
  Predict = X_ext * theta;
  Predict_g = sigmoid(Predict);
  disp([X_new, Predict, Predict_g, (Predict_g >= 0.5)]);
  idx = (Predict_g >= 0.5);
  plot(X_new(idx), Predict(idx), 'r*', X_new(~idx), Predict(~idx), 'bo');

  sigmoidGraph(X_new, Predict, 0.6, 'Prediction GenData with 0.6 diff');

endfunction
