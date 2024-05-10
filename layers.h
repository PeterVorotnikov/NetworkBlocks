#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;


class LinearLayer {
public:
	int nOfInputs = 0, nOfOutputs = 0, maxBatchSize = 0;

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

public:
	LinearLayer(int inputs, int outputs, int batchSize);
	void forward(vector<vector<double>>& input, int batchSize);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff, int batchSize);
	void zeroGradients();
	void save(string fileName);
	void load(string fileName);
};



class ReLU1d {
public:
	int d = 0, l = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;

public:
	ReLU1d(int d, int batchSize, int l = 0);
	void forward(vector<vector<double>>& input, int batchSize);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff, int batchSize);
};

class Sigmoid1d {
public:
	int d = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;
	double f(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

public:
	Sigmoid1d(int d, int batchSize);
	void forward(vector<vector<double>>& input, int batchSize);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff, int batchSize);
};


class ReLU3d {
public:
	int d1 = 0, d2 = 0, d3 = 0, l = 0, maxBatchSize = 0;
	vector<vector<vector<vector<double>>>> output;
	vector<vector<vector<vector<double>>>> diff;

public:
	ReLU3d(int d1, int d2, int d3, int batchSize, int l = 0);
	void forward(vector<vector<vector<vector<double>>>>& input, int batchSize);
	void backward(vector<vector<vector<vector<double>>>>& input, 
		vector<vector<vector<vector<double>>>>& nextDiff, int batchSize);
};




class ConvolutionalLayer {
public:
	int rows = 0, cols = 0, inputChannels = 0, outputChannels = 0, maxBatchSize = 0, 
		kernelSize = 0;
	vector<vector<vector<vector<double>>>> output;
	vector<vector<vector<vector<double>>>> diff;
	vector<vector<vector<vector<double>>>> weights;
	vector<double> biases;
	vector<vector<vector<vector<double>>>> weightsDiff;
	vector<double> biasesDiff;

public:
	ConvolutionalLayer(int rows, int cols, int inputChannels, int outputChannels, 
		int batchSize, int kernelSize = 3);
	void forward(vector<vector<vector<vector<double>>>>& input, int batchSize);
	void backward(vector<vector<vector<vector<double>>>>& input,
		vector<vector<vector<vector<double>>>>& nextDiff, int batchSize);
	void zeroGradients();
	void save(string fileName);
	void load(string fileName);
};



class Flatten31 {
public:
	int d1 = 0, d2 = 0, d3 = 0, maxBatchSize = 0;
	int outputSize = 0;
	vector<vector<double>> output;
	vector<vector<vector<vector<double>>>> diff;

public:
	Flatten31(int d1, int d2, int d3, int batchSize);
	void forward(vector<vector<vector<vector<double>>>>& input, int batchSize);
	void backward(vector<vector<vector<vector<double>>>>& input, 
		vector<vector<double>>& nextDiff, int batchSize);
};


class MaxPooling {
public:
	int channels = 0, inputRows = 0, inputCols = 0, outputRows = 0, outputCols = 0,
		maxBatchSize, size = 0;
	vector<vector<vector<vector<double>>>> output;
	vector<vector<vector<vector<double>>>> diff;
	vector<vector<vector<vector<int>>>> memory;

public:
	MaxPooling(int channels, int inputRows, int inputCols, int batchSize, int size = 2);
	void forward(vector<vector<vector<vector<double>>>>& input, int batchSize);
	void backward(vector<vector<vector<vector<double>>>>& input,
		vector<vector<vector<vector<double>>>>& nextDiff, int batchSize);
};


class Softmax {
public:
	int d = 0, maxBatchSize = 0;
	vector<vector<double>> output;
	vector<vector<double>> diff;
	vector<double> exponents;
public:
	Softmax(int d, int batchSize);
	void forward(vector<vector<double>>& input, int batchSize);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff, int batchSize);
};



class Dropout1d {
public:
	int d = 0, maxBatchSize = 0;
	double p = 0;
	vector<vector<double>> output;
	vector<vector<double>> mask;
	vector<vector<double>> diff;
public:
	Dropout1d(int d, int batchSize, double p = 0.5);
	void forward(vector<vector<double>>& input, int batchSize, bool training = true);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff, int batchSize);
};

class Dropout3d {
public:
	int d1 = 0, d2 = 0, d3 = 0, maxBatchSize = 0;
	double p = 0;
	vector<vector<vector<vector<double>>>> output;
	vector<vector<vector<vector<double>>>> mask;
	vector<vector<vector<vector<double>>>> diff;
public:
	Dropout3d(int d1, int d2, int d3, int batchSize, double p = 0.5);
	void forward(vector<vector<vector<vector<double>>>>& input, int batchSize, bool training = true);
	void backward(vector<vector<vector<vector<double>>>>& input,
		vector<vector<vector<vector<double>>>>& nextDiff, int batchSize);
};


class BatchNormalization1d {
public:
	int d = 0, maxBatchSize = 0, count = 0;
	double epsilon = pow(10, -8);

	vector<vector<double>> output, diff;
	vector<double> gamma, beta, gammaDiff, betaDiff;

	vector<double> muLearn, varLearn, stdLearn, muLearnDiff, varLearnDiff;
	vector<double> mu, std;

public:
	BatchNormalization1d(int d, int batchSize);
	void forward(vector<vector<double>>& input, int batchSize, bool training = true);
	void backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff, int batchSize);
	void zeroGradients();
	void save(string fileName);
	void load(string fileName);
};


class BatchNormalization3d {
public:
	int d1 = 0, d2 = 0, d3 = 0, maxBatchSize = 0, count = 0;
	double epsilon = pow(10, -8);

	vector<vector<vector<vector<double>>>> output, diff;
	vector<vector<vector<double>>> gamma, beta, gammaDiff, betaDiff;

	vector<vector<vector<double>>> muLearn, varLearn, stdLearn, muLearnDiff, varLearnDiff;
	vector<vector<vector<double>>> mu, std;

public:
	BatchNormalization3d(int d1, int d2, int d3, int batchSize);
	void forward(vector<vector<vector<vector<double>>>>& input, int batchSize, 
		bool training = true);
	void backward(vector<vector<vector<vector<double>>>>& input,
		vector<vector<vector<vector<double>>>>& nextDiff, int batchSize);
	void zeroGradients();
	void save(string fileName);
	void load(string fileName);
};