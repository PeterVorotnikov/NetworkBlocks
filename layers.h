#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>

using namespace std;


class LinearLayer {
public:
	int nOfInputs = 0, nOfOutputs = 0, maxBatchSize = 0, nOfThreads = 0;

	vector<vector<double>> output;
	vector<vector<double>> diff;

	vector<vector<double>> weights;
	vector<vector<double>> weightsDiff;
	vector<double> biases;
	vector<double> biasesDiff;

private:
	void init();
	void initStates();
	void initWeights();
	void forwardParallel(vector<vector<double>>& input, int batchStart, int batchEnd);
	void backwardParallel(vector<vector<double>>& input, vector<vector<double>>& nextDiff,
		int batchStart, int batchEnd);

public:
	LinearLayer(int inputs, int outputs, int batchSize, int threads);
	void forward(vector<vector<double>>& input);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff);
};