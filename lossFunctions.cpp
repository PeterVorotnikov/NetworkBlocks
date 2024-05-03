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
	int batchSize = output.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < nOfOutputs; i++) {
			value += pow(target[b][i] - output[b][i], 2);
			diff[b][i] = -2 * (target[b][i] - output[b][i]);
		}
	}
}