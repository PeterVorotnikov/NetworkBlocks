#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <thread>

using namespace std;


class LinearLayer {
public:
	int nOfInputs = 0, nOfOutputs = 0, maxBatchSize = 0, nOfThreads = 0;

	vector<vector<double>> output;
	vector<vector<double>> diff;

	vector<vector<double>> weights;
	vector<double> biases;

private:
	void init();
	void initStates(int batchStart, int batchEnd);
	void initWeights(int inputStart, int inputEnd);

public:
	LinearLayer(int inputs, int outputs, int batchSize, int threads);
};