pkg load nnet;

load('ex4data1.mat')

% Invert data set to fit rows and not columns
%X = X';
%y = y';

% hidden neurons, output neurons
hidden_neurons = 16;
output_neurons = numel(unique(y));

epochs = 200;
train_size = 120;

% train, test, validation indices
ntrain = 0;
idx_train = ntrain+1:ntrain+train_size;
ntest = 4000;
idx_test = ntest+1:ntest+train_size;
nvali = 4800;
idx_vali = nvali+1:nvali+train_size;

y10 = zeros(size(y, 1), output_neurons);
y10( sub2ind(size(y10), [1:numel(y)]', y)) = 1;

X = [X y10 y];

% randomize orer of rows, to blend the training data
X = X( randperm(size(X,1)), : );
% get back re-ordered y
y = X(:,end);
% remove y
X(:, end) = [];
% row based
X = X';

% create train, test, validate data, randomized order
X_train = X(:, idx_train);
X_test = X(:, idx_test);
X_vali = X(:, idx_vali);
y_test = y(idx_test);

% indices of input & output data
idx_in = size(X_train, 1) - output_neurons;
idx_out = idx_in + 1;

% feed forward network
R = min_max(X_train(1:idx_in, :));
S = [hidden_neurons output_neurons];
net = newff(R, S, {'tansig', 'purelin'}, 'trainlm', '', 'mse');

% create randomized wights for simmetry breaking
epsilon_init = 0.12;
InW = rand(hidden_neurons, size(R, 1)) * 2 * epsilon_init - epsilon_init;
LaW = rand(output_neurons, hidden_neurons) * 2 * epsilon_init - epsilon_init;

net.IW{1, 1} = InW;
net.LW{2, 1} = LaW;
net.b{1, 1}(:) = 1;
net.b{2, 1}(:) = 1;

net.trainParam.epochs = epochs;

saveMLPStruct(net, "MLPstruct_digit.txt");

% define validation data new, for matlab compatibility
VV.P = X_vali(1 : idx_in , :);
VV.T = X_vali(idx_out : end, :);

% train
fprintf('\nTRAIN the network...\n');

net_train = train(net, X_train(1:idx_in, :), X_train(idx_out:end, :));

sim_out = sim(net_train, X_test(1:idx_in,:));

sim_out1 = round(sim_out);

% convert back 10 outputs to 1
[val, idx] = max(sim_out);

% scatter plot
xlim([0 max(y)]);
ylim([0 max(y)]);
scatter(idx', y_test);
title({'Handwriting Digit Scatter, ANN Simulation, Octave nnet';...
      sprintf('(epochs=%d, train-size=%d)', epochs, train_size)});
xlabel('Result Digit');
ylabel('Test Digit');

fprintf('\nTraining Set Accuracy: %f\n', mean(double(idx' == y_test)) * 100);


