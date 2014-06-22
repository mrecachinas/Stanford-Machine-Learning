function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    summation = zeros(m, 1);
    for i=1:m
        hy = theta(1) + theta(2) * X(i,2) - y(i);
        summation(1) = summation(1) + hy * X(i,1);
        summation(2) = summation(2) + hy * X(i,2);
    end

    theta0 = theta(1) - alpha / m * summation(1);
    theta1 = theta(2) - alpha / m * summation(2);
    theta(1) = theta0;
    theta(2) = theta1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
