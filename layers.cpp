#include "layers.h"



LinearLayer::LinearLayer(int inputs, int outputs, int batchSize, int threads) {
	nOfInputs = inputs;
	nOfOutputs = outputs;
	maxBatchSize = batchSize;
	nOfThreads = threads;

	init();
}

void LinearLayer::initStates(int batchStart, int batchEnd) {
	for (int i = batchStart; i < batchEnd; i++) {
		output[i].resize(nOfOutputs);
		diff[i].resize(nOfInputs);
	}
	cout << batchStart << " " << batchEnd << " " << this_thread::get_id() << endl;
}

void LinearLayer::initWeights(int inputStart, int inputEnd) {
	for (int i = inputStart; i < inputEnd; i++) {
		weights[i].resize(nOfOutputs);
		for (int j = 0; j < nOfOutputs; j++) {
			double r = (double)rand() / (double)RAND_MAX;
			r = r * (2.0 / (double)nOfInputs) - 1.0 / (double)nOfInputs;
			weights[i][j] = r;
		}
	}
}

void LinearLayer::init() {
	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	cout << "resized\n";

	//initStates(0, maxBatchSize);
	int size = maxBatchSize / nOfThreads;
	int k = maxBatchSize % nOfThreads;
	vector<thread> initStatesThreads;
	int left = 0, right = 0;
	for (int i = 0; i < nOfThreads; i++) {
		right = left + size;
		if (i < k) {
			right++;
		}
		thread t(&LinearLayer::initStates, this, left, right);
		initStatesThreads.push_back(move(t));
		left = right;
	}
	for (int i = 0; i < initStatesThreads.size(); i++) {
		initStatesThreads[i].join();
	}
	cout << "states done\n";

	weights.resize(nOfInputs);
	//initWeights(0, nOfInputs);
	size = nOfInputs / nOfThreads;
	k = nOfInputs % nOfThreads;
	vector<thread> initParametersThreads;
	left = 0, right = 0;
	for (int i = 0; i < nOfThreads; i++) {
		right = left + size;
		if (i < k) {
			right++;
		}
		thread t(&LinearLayer::initWeights, this, left, right);
		initParametersThreads.push_back(move(t));
		left = right;
	}
	for (int i = 0; i < initParametersThreads.size(); i++) {
		initParametersThreads[i].join();
	}

	biases.resize(nOfOutputs);
}
