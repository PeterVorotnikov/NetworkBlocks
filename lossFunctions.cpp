#include "lossFunctions.h"


MSELoss::MSELoss(int outputs, int batchSize) {
	nOfOutputs = outputs;
	maxBatchSize = batchSize;
	diff.resize(batchSize);
	for (int i = 0; i < batchSize; i++) {
		diff[i].resize(outputs);
	}
}

void MSELoss::calculate(vector<vector<double>>& output, vector<vector<double>>& target) {
	value = 0;
	int batchSize = target.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < nOfOutputs; i++) {
			value += pow(target[b][i] - output[b][i], 2);
			diff[b][i] = -2 * (target[b][i] - output[b][i]);
		}
	}
}



CategoricalCrossentropyLoss::CategoricalCrossentropyLoss(int n, int batchSize) {
	nOfClasses = n;
	maxBatchSize = batchSize;
	diff.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		diff[b].resize(nOfClasses);
	}
}

void CategoricalCrossentropyLoss::calculate(vector<vector<double>>& output, 
	vector<int>& target) {
	int batchSize = output.size();
	value = 0;
	for (int b = 0; b < batchSize; b++) {
		double sumOfExponent = 0;
		for (int i = 0; i < nOfClasses; i++) {
			sumOfExponent += exp(output[b][i]);
		}
		value += -output[b][target[b]] + log(sumOfExponent);
		for (int i = 0; i < nOfClasses; i++) {
			diff[b][i] = exp(output[b][i]) / sumOfExponent;
			if (i == target[b]) {
				diff[b][i] += (-1);
			}
		}
	}
}