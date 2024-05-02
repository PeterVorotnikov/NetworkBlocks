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

	weights.resize(nOfInputs);
	for (int i = 0; i < nOfInputs; i++) {
		weights[i].resize(nOfOutputs);
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
