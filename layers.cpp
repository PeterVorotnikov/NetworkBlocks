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





ReLU1d::ReLU1d(int d, int batchSize, int l) {
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

void ReLU1d::forward(vector<vector<double>>& input) {
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

void ReLU1d::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
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



Sigmoid1d::Sigmoid1d(int d, int batchSize) {
	this->d = d;
	maxBatchSize = batchSize;

	output.resize(batchSize);
	diff.resize(batchSize);
	for (int i = 0; i < maxBatchSize; i++) {
		output[i].resize(d);
		diff[i].resize(d);
	}
}

void Sigmoid1d::forward(vector<vector<double>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			output[b][i] = f(input[b][i]);
		}
	}
}

void Sigmoid1d::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			diff[b][i] = nextDiff[b][i] * f(input[b][i]) * (1 - f(input[b][i]));
		}
	}
}



ReLU3d::ReLU3d(int d1, int d2, int d3, int batchSize, int l = 0) {
	this->d1 = d1;
	this->d2 = d2;
	this->d3 = d3;
	maxBatchSize = batchSize;
	this->l = l;

	output.resize(batchSize);
	diff.resize(batchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(d1);
		diff[b].resize(d1);
		for (int i1 = 0; i1 < d1; i1++) {
			output[b][i1].resize(d2);
			diff[b][i1].resize(d2);
			for (int i2 = 0; i2 < d2; i2++) {
				output[b][i1][i2].resize(d3);
				diff[b][i1][i2].resize(d3);
			}
		}
	}
}

void ReLU3d::forward(vector<vector<vector<vector<double>>>>& input) {
	for (int b = 0; b < maxBatchSize; b++) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					if (input[b][i1][i2][i3] >= 0) {
						output[b][i1][i2][i3] = input[b][i1][i2][i3];
					}
					else {
						output[b][i1][i2][i3] = l * input[b][i1][i2][i3];
					}
				}
			}
		}
	}
}

void ReLU3d::backward(vector<vector<vector<vector<double>>>>& input,
	vector<vector<vector<vector<double>>>>& nextDiff) {
	for (int b = 0; b < maxBatchSize; b++) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					if (input[b][i1][i2][i3] >= 0) {
						diff[b][i1][i2][i3] = nextDiff[b][i1][i2][i3];
					}
					else {
						diff[b][i1][i2][i3] = l * nextDiff[b][i1][i2][i3];
					}
				}
			}
		}
	}
}