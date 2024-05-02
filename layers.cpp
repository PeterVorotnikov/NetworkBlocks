#include "layers.h"



LinearLayer::LinearLayer(int inputs, int outputs, int batchSize, int threads) {
	nOfInputs = inputs;
	nOfOutputs = outputs;
	maxBatchSize = batchSize;
	nOfThreads = threads;

	init();
}

void LinearLayer::initStates() {
	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);

	for (int i = 0; i < maxBatchSize; i++) {
		output[i].resize(nOfOutputs);
		diff[i].resize(nOfInputs);
	}
}


void LinearLayer::initWeights() {
	biases.resize(nOfOutputs);
	biasesDiff.resize(nOfOutputs);

	weights.resize(nOfInputs);
	weightsDiff.resize(nOfInputs);
	for (int i = 0; i < nOfInputs; i++) {
		weights[i].resize(nOfOutputs);
		weightsDiff[i].resize(nOfOutputs);
		for (int j = 0; j < nOfOutputs; j++) {
			double r = (double)rand() / (double)RAND_MAX;
			r = r * (2.0 / (double)nOfInputs) - 1.0 / (double)nOfInputs;
			weights[i][j] = r;
		}
	}
}

void LinearLayer::init() {
	thread t1(&LinearLayer::initStates, this);
	thread t2(&LinearLayer::initWeights, this);
	t1.join();
	t2.join();
}

void LinearLayer::forwardParallel(vector<vector<double>>& input, int batchStart, int batchEnd) {
	for (int b = batchStart; b < batchEnd; b++) {
		for (int out = 0; out < nOfOutputs; out++) {
			double val = biases[out];
			for (int in = 0; in < nOfInputs; in++) {
				val += input[b][in] * weights[in][out];
			}
			output[b][out] = val;
		}
	}
}

void LinearLayer::backwardParallel(vector<vector<double>>& input, 
	vector<vector<double>>& nextDiff, int batchStart, int batchEnd) {
	for (int b = batchStart; b < batchEnd; b++) {
		for (int out = 0; out < nOfOutputs; out++) {
			biasesDiff[out] += nextDiff[b][out];
		}
		for (int in = 0; in < nOfInputs; in++) {
			double val = 0;
			for (int out = 0; out < nOfOutputs; out++) {
				val += nextDiff[b][out] * weights[in][out];
				weightsDiff[in][out] += nextDiff[b][out] * input[b][in];
			}
			diff[b][in] = val;
		}
	}
}








void LinearLayer::forward(vector<vector<double>>& input) {
	int batchSize = input.size();
	int size = batchSize / nOfThreads;
	int k = batchSize % nOfThreads;
	vector<thread> threads;
	int left = 0, right = 0;
	for (int i = 0; i < nOfThreads; i++) {
		right = left + size;
		if (i < k) {
			right++;
		}
		thread t(&LinearLayer::forwardParallel, this, ref(input), left, right);
		threads.push_back(move(t));
		left = right;
	}
	for (int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void LinearLayer::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	int size = batchSize / nOfThreads;
	int k = batchSize % nOfThreads;
	vector<thread> threads;
	int left = 0, right = 0;
	for (int i = 0; i < nOfThreads; i++) {
		right = left + size;
		if (i < k) {
			right++;
		}
		thread t(&LinearLayer::backwardParallel, this, ref(input), ref(nextDiff), left, right);
		threads.push_back(move(t));
		left = right;
	}
	for (int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}
