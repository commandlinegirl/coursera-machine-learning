function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%

size(y);

y_bool = zeros(size(y, 1), num_labels);
for i = 1:size(y, 1)
  j = y(i, 1);
  y_bool(i, j) = 1;
endfor

a_1 = [ones(m, 1), X];              % (5000x401)
z_2 = Theta1 * a_1';                % 
a_2 = sigmoid(z_2);                 % (25x5000)
a_2 = [ones(1, size(a_2, 2)); a_2]; % (26x5000)
z_3 = Theta2 * a_2;                 % (10*26) * (26*5000)
a_3 = sigmoid(z_3);                 % (10x5000)

size(y_bool);
posTerm = -y_bool .* log(a_3)';
negTerm = (1 - y_bool) .* log(1 - a_3)';
J = 1/m * sum((posTerm - negTerm)(:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regulatized cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%

Theta1_Filter = Theta1(:,2:end);
Theta2_Filter = Theta2(:,2:end);

theta_sum = sum(Theta1_Filter(:).^2) + sum(Theta2_Filter(:).^2);
regTerm = lambda / (2*m) * theta_sum;
J = J + regTerm;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% size(X) - 5000x400
% size(Theta1) - 25x401
% m = 5000
% y_bool = 5000x10

Delta_1 = 0;
Delta_2 = 0;

for t = 1:m
  % Compute activations for all layer (use forward propagation)
  a_1 = [1; X(t,:)'];                % result: 1x401
  size(Theta1);                      % 25x401
  z_2 = Theta1 * a_1;                % result: 25x401 * 401x1 -> 25x1
  a_2 = sigmoid(z_2);	             %
  a_2 = [1; a_2];                    % 26x1
  a_3 = sigmoid(Theta2 * a_2);       % result: 10x26 * 26x1 -> 10x1 
  
  % Compute the delta for the last layer in the network (backpropagation)
  current_y = y_bool(t, :)';         % result: 10x1
  delta_3 = a_3 - current_y;         % 10x1

  % Compute the delta for the earlier layers in the network
  delta_2 = Theta2_Filter' * delta_3 .* sigmoidGradient(z_2); 

  % STEP 4
  Delta_2 = Delta_2 + delta_3 * a_2';
  Delta_1 = Delta_1 + delta_2 * a_1';
 
endfor

Theta1_grad = 1/m * Delta_1;
Theta2_grad = 1/m * Delta_2;

size(Theta1_grad); %25x401;
size(Theta2_grad); %10x26;

% -------------------------------------------------------------

Theta1_grad(:, 2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1_Filter;
Theta2_grad(:, 2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2_Filter;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
