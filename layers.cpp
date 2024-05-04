#include "layers.h"



LinearLayer::LinearLayer(int inputs, int outputs, int batchSize) {
	nOfInputs = inputs;
	nOfOutputs = outputs;
	maxBatchSize = batchSize;

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
	initStates();
	initWeights();
}

void LinearLayer::forward(vector<vector<double>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int out = 0; out < nOfOutputs; out++) {
			double val = biases[out];
			for (int in = 0; in < nOfInputs; in++) {
				val += input[b][in] * weights[in][out];
			}
			output[b][out] = val;
		}
	}
}

void LinearLayer::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
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

void LinearLayer::zeroGradients() {
	for (int out = 0; out < nOfOutputs; out++) {
		biasesDiff[out] = 0;
		for (int in = 0; in < nOfInputs; in++) {
			weightsDiff[in][out] = 0;
		}
	}
}





ReLU2d::ReLU2d(int d, int batchSize, int l) {
	this->d = d;
	maxBatchSize = batchSize;
	this->l = l;

	output.resize(batchSize);
	diff.resize(batchSize);
	for (int i = 0; i < maxBatchSize; i++) {
		output[i].resize(d);
		diff[i].resize(d);
	}
}

void ReLU2d::forward(vector<vector<double>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			if (input[b][i] >= 0) {
				output[b][i] = input[b][i];
			}
			else {
				output[b][i] = l * input[b][i];
			}
		}
	}
}

void ReLU2d::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			if (input[b][i] >= 0) {
				diff[b][i] = nextDiff[b][i];
			}
			else {
				diff[b][i] = l * nextDiff[b][i];
			}
		}
	}
}



Sigmoid::Sigmoid(int d, int batchSize) {
	this->d = d;
	maxBatchSize = batchSize;

	output.resize(batchSize);
	diff.resize(batchSize);
	for (int i = 0; i < maxBatchSize; i++) {
		output[i].resize(d);
		diff[i].resize(d);
	}
}

void Sigmoid::forward(vector<vector<double>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			output[b][i] = f(input[b][i]);
		}
	}
}

void Sigmoid::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			diff[b][i] = nextDiff[b][i] * f(input[b][i]) * (1 - f(input[b][i]));
		}
	}
}