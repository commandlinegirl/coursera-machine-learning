function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

steps = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
steps_num = size(steps, 1);
C_sigma_pairs = [];

best = zeros(3,1);
for i = 1:steps_num
  for j = 1:steps_num
    C_cur = steps(i);
    sigma_cur = steps(j);
    model = svmTrain(X, y, C_cur, @(x1, x2) gaussianKernel(x1, x2, sigma_cur));
    predictions = svmPredict(model, Xval); 
    prediction_error = mean(double(predictions ~= yval));
    v = [C_cur, sigma_cur, prediction_error];
    C_sigma_pairs = [C_sigma_pairs; v];
  endfor
endfor

[el, idx] = min(C_sigma_pairs(:,3),[],1)

C = C_sigma_pairs(idx, :)(1);
sigma = C_sigma_pairs(idx, :)(2);

C
sigma
% =========================================================================

end
