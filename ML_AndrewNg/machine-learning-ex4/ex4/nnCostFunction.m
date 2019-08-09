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
Theta1_grad = zeros(size(Theta1)); %25*401
Theta2_grad = zeros(size(Theta2)); %10*26

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

a1=[ones(m,1) X];           %m*401
z2=a1*Theta1';              %(m*401)*(401*25)=m*25
a2=[ones(m,1) sigmoid(z2)]; %m*26
z3=a2*Theta2';              %(m*26)*(26*k)=m*k k=10
h=sigmoid(z3);              %m*k
a3=h;

tmpy=h;  %tmp=FORMAT(y)  y=5000*1
[rnum,cnum]=size(tmpy);
%FORMAT(y)
for i=1:rnum
    for j=1:cnum
        if(y(i)==j)
            tmpy(i,j)=1;
        else
            tmpy(i,j)=0;
        end
    end
end


J=(sum(sum(((-tmpy).*log(h))-((1-tmpy).*log(1-h)))))/m; %y=m*k

% -------------------------------------------------------------

tmp1=Theta1(:,2:size(Theta1,2)); %Theta1(2:end)
tmp2=Theta2(:,2:size(Theta2,2)); %Theta2(2:end)

J=J+(lambda/(2*m))*(sum(sum(tmp1.^2))+sum(sum(tmp2.^2)));

% =========================================================================

theta3=a3-tmpy; %m*k 每一个样本的θ3 这里曾经因为写成tmpy-a3浪费了大量时间
theta2=theta3*Theta2.*[ones(m,1) sigmoidGradient(z2)];%m*26 每一个样本的θ2
theta2=theta2(1:m,2:hidden_layer_size+1); %删除θ(2)0 m*25


for p=1:m
   Theta1_grad=Theta1_grad+theta2(p,:)'*a1(p,:);
   Theta2_grad=Theta2_grad+theta3(p,:)'*a2(p,:);
end

Theta1_grad=Theta1_grad./m;
Theta2_grad=Theta2_grad./m;

%Regularized
tmp1=[zeros(size(tmp1,1),1) tmp1];
tmp2=[zeros(size(tmp2,1),1) tmp2];

Theta1_grad=Theta1_grad+(lambda/m)*tmp1;
Theta2_grad=Theta2_grad+(lambda/m)*tmp2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
