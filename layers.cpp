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

void LinearLayer::save(string fileName) {
	ofstream file(fileName);
	for (int out = 0; out < nOfOutputs; out++) {
		file << biases[out] << " ";
		for (int in = 0; in < nOfInputs; in++) {
			file << weights[in][out] << " ";
		}
	}
	file.close();
}

void LinearLayer::load(string fileName) {
	ifstream file(fileName);
	for (int out = 0; out < nOfOutputs; out++) {
		file >> biases[out];
		for (int in = 0; in < nOfInputs; in++) {
			file >> weights[in][out];
		}
	}
	file.close();
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



ReLU3d::ReLU3d(int d1, int d2, int d3, int batchSize, int l) {
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
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
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
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
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



ConvolutionalLayer::ConvolutionalLayer(int rows, int cols, int inputChannels, int outputChannels,
	int batchSize, int kernelSize) {
	this->rows = rows;
	this->cols = cols;
	this->inputChannels = inputChannels;
	this->outputChannels = outputChannels;
	maxBatchSize = batchSize;
	this->kernelSize = kernelSize;

	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	for (int b = 0; b < batchSize; b++) {
		output[b].resize(outputChannels);
		for (int c = 0; c < outputChannels; c++) {
			output[b][c].resize(rows);
			for (int r = 0; r < rows; r++) {
				output[b][c][r].resize(cols);
			}
		}
		diff[b].resize(inputChannels);
		for (int c = 0; c < inputChannels; c++) {
			diff[b][c].resize(rows);
			for (int r = 0; r < rows; r++) {
				diff[b][c][r].resize(cols);
			}
		}
	}

	biases.resize(outputChannels);
	weights.resize(outputChannels);
	biasesDiff.resize(outputChannels);
	weightsDiff.resize(outputChannels);
	for (int outChannel = 0; outChannel < outputChannels; outChannel++) {
		weights[outChannel].resize(inputChannels);
		weightsDiff[outChannel].resize(inputChannels);
		for (int inChannel = 0; inChannel < inputChannels; inChannel++) {
			weights[outChannel][inChannel].resize(kernelSize);
			weightsDiff[outChannel][inChannel].resize(kernelSize);
			for (int i = 0; i < kernelSize; i++) {
				weights[outChannel][inChannel][i].resize(kernelSize);
				weightsDiff[outChannel][inChannel][i].resize(kernelSize);
				for (int j = 0; j < kernelSize; j++) {
					double r = (double)rand() / (double)RAND_MAX;
					r *= 2.0 / (double)(inputChannels * kernelSize * kernelSize);
					r -= 1.0 / (double)(inputChannels * kernelSize * kernelSize);
					weights[outChannel][inChannel][i][j] = r;
				}
			}
		}
	}
}

void ConvolutionalLayer::forward(vector<vector<vector<vector<double>>>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int out = 0; out < outputChannels; out++) {
			for (int r2 = 0; r2 < rows; r2++) {
				for (int c2 = 0; c2 < cols; c2++) {
					output[b][out][r2][c2] = biases[out];
					for (int in = 0; in < inputChannels; in++) {
						for (int r = 0; r < kernelSize; r++) {
							for (int c = 0; c < kernelSize; c++) {
								int r1 = r2 + r - kernelSize / 2;
								int c1 = c2 + c - kernelSize / 2;
								if (r1 < 0 || r1 >= rows || c1 < 0 || c1 >= cols) {
									continue;
								}
								output[b][out][r2][c2] += input[b][in][r1][c1] *
									weights[out][in][r][c];
							}
						}
					}
				}
			}
		}
	}
}

void ConvolutionalLayer::backward(vector<vector<vector<vector<double>>>>& input,
	vector<vector<vector<vector<double>>>>& nextDiff) {
	int batchSize = input.size();

	for (int b = 0; b < batchSize; b++) {
		for (int in = 0; in < inputChannels; in++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					diff[b][in][r][c] = 0;
				}
			}
		}
	}

	for (int b = 0; b < batchSize; b++) {
		for (int out = 0; out < outputChannels; out++) {
			for (int r2 = 0; r2 < rows; r2++) {
				for (int c2 = 0; c2 < cols; c2++) {
					
					biasesDiff[out] += nextDiff[b][out][r2][c2];
					for (int in = 0; in < inputChannels; in++) {
						for (int r = 0; r < kernelSize; r++) {
							for (int c = 0; c < kernelSize; c++) {
								int r1 = r2 + r - kernelSize / 2;
								int c1 = c2 + c - kernelSize / 2;
								if (r1 < 0 || r1 >= rows || c1 < 0 || c1 >= cols) {
									continue;
								}
								diff[b][in][r1][c1] += nextDiff[b][out][r2][c2] *
									weights[out][in][r][c];
								weightsDiff[out][in][r][c] += nextDiff[b][out][r2][c2] *
									input[b][in][r1][c1];
							}
						}
					}

				}
			}
		}
	}
}

void ConvolutionalLayer::zeroGradients() {
	for (int out = 0; out < outputChannels; out++) {
		biasesDiff[out] = 0;
		for (int in = 0; in < inputChannels; in++) {
			for (int r = 0; r < kernelSize; r++) {
				for (int c = 0; c < kernelSize; c++) {
					weightsDiff[out][in][r][c] = 0;
				}
			}
		}
	}
}

void ConvolutionalLayer::save(string fileName) {
	ofstream file(fileName);
	for (int out = 0; out < outputChannels; out++) {
		file << biases[out] << " ";
		for (int in = 0; in < inputChannels; in++) {
			for (int r = 0; r < kernelSize; r++) {
				for (int c = 0; c < kernelSize; c++) {
					file << weights[out][in][r][c] << " ";
				}
			}
		}
	}
	file.close();
}

void ConvolutionalLayer::load(string fileName) {
	ifstream file(fileName);
	for (int out = 0; out < outputChannels; out++) {
		file >> biases[out];
		for (int in = 0; in < inputChannels; in++) {
			for (int r = 0; r < kernelSize; r++) {
				for (int c = 0; c < kernelSize; c++) {
					file >> weights[out][in][r][c];
				}
			}
		}
	}
	file.close();
}



Flatten31::Flatten31(int d1, int d2, int d3, int batchSize) {
	this->d1 = d1;
	this->d2 = d2;
	this->d3 = d3;
	outputSize = d1 * d2 * d3;
	maxBatchSize = batchSize;

	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(outputSize);
		diff[b].resize(d1);
		for (int i1 = 0; i1 < d1; i1++) {
			diff[b][i1].resize(d2);
			for (int i2 = 0; i2 < d2; i2++) {
				diff[b][i1][i2].resize(d3);
			}
		}
	}
}

void Flatten31::forward(vector<vector<vector<vector<double>>>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		int i = 0;
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					output[b][i] = input[b][i1][i2][i3];
					i++;
				}
			}
		}
	}
}

void Flatten31::backward(vector<vector<vector<vector<double>>>>& input,
	vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
		int i = 0;
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					diff[b][i1][i2][i3] = nextDiff[b][i];
					i++;
				}
			}
		}
	}
}


MaxPooling::MaxPooling(int channels, int inputRows, int inputCols, int batchSize, int size) {
	this->channels = channels;
	this->inputRows = inputRows;
	this->inputCols = inputCols;
	maxBatchSize = batchSize;
	this->size = size;
	outputRows = inputRows / size;
	outputCols = inputCols / size;

	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	memory.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(channels);
		diff[b].resize(channels);
		memory[b].resize(channels);
		for (int c = 0; c < channels; c++) {
			output[b][c].resize(outputRows);
			memory[b][c].resize(outputRows);
			for (int r = 0; r < outputRows; r++) {
				output[b][c][r].resize(outputCols);
				memory[b][c][r].resize(outputCols);
			}
			diff[b][c].resize(inputRows);
			for (int r = 0; r < inputRows; r++) {
				diff[b][c][r].resize(inputCols);
			}
		}
	}
}

void MaxPooling::forward(vector<vector<vector<vector<double>>>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int channel = 0; channel < channels; channel++) {
			for (int r = 0; r < outputRows; r++) {
				for (int c = 0; c < outputCols; c++) {
					double maximum = input[b][channel][r * size][c * size];
					memory[b][channel][r][c] = 0;
					int i = 0;
					for (int dr = 0; dr < size; dr++) {
						for (int dc = 0; dc < size; dc++) {
							if (maximum < input[b][channel][r * size + dr][c * size + dc]) {
								maximum = input[b][channel][r * size + dr][c * size + dc];
								memory[b][channel][r][c] = i;
							}
							i++;
						}
					}
					output[b][channel][r][c] = maximum;
				}
			}
		}
	}
}

void MaxPooling::backward(vector<vector<vector<vector<double>>>>& input,
	vector<vector<vector<vector<double>>>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
		for (int channel = 0; channel < channels; channel++) {
			for (int r = 0; r < inputRows; r++) {
				for (int c = 0; c < inputCols; c++) {
					diff[b][channel][r][c] = 0;
				}
			}
		}
	}
	for (int b = 0; b < batchSize; b++) {
		for (int channel = 0; channel < channels; channel++) {
			for (int r = 0; r < outputRows; r++) {
				for (int c = 0; c < outputCols; c++) {
					int dr = memory[b][channel][r][c] / size;
					int dc = memory[b][channel][r][c] % size;
					diff[b][channel][r * size + dr][c * size + dc] = nextDiff[b][channel][r][c];
				}
			}
		}
	}
}



Softmax::Softmax(int d, int batchSize) {
	this->d = d;
	maxBatchSize = batchSize;
	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(d);
		diff[b].resize(d);
	}
	exponents.resize(d);
}

void Softmax::forward(vector<vector<double>>& input) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		double sumOfExponents = 0;
		for (int i = 0; i < d; i++) {
			sumOfExponents += exp(input[b][i]);
		}
		for (int i = 0; i < d; i++) {
			output[b][i] = exp(input[b][i]) / sumOfExponents;
		}
	}
}

void Softmax::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = nextDiff.size();
	for (int b = 0; b < batchSize; b++) {
		double sumOfExponents = 0;
		for (int i = 0; i < d; i++) {
			exponents[i] = exp(input[b][i]);
			sumOfExponents += exponents[i];
		}
		for (int in = 0; in < d; in++) {
			diff[b][in] = 0;
			for (int out = 0; out < d; out++) {
				double v = -exponents[out] / pow(sumOfExponents, 2);
				if (in == out) {
					v = (exponents[out] * sumOfExponents - pow(exponents[out], 2)) /
						pow(sumOfExponents, 2);
				}
				diff[b][in] += nextDiff[b][out] * v;
			}
		}
	}
}


Dropout1d::Dropout1d(int d, int batchSize, double p) {
	this->d = d;
	maxBatchSize = batchSize;
	this->p = p;

	output.resize(maxBatchSize);
	mask.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(d);
		mask[b].resize(d);
		diff[b].resize(d);
	}
}

void Dropout1d::forward(vector<vector<double>>& input, bool training) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			mask[b][i] = 1;
			if (training) {
				double r = (double)rand() / (double)RAND_MAX;
				if (r < p) {
					mask[b][i] = 0;
				}
			}
			double v = 1;
			if (training) {
				v /= (1.0 - p);
			}
			output[b][i] = input[b][i] * mask[b][i] * v;
		}
	}
}

void Dropout1d::backward(vector<vector<double>>& input, vector<vector<double>>& nextDiff) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i = 0; i < d; i++) {
			diff[b][i] = nextDiff[b][i] * mask[b][i] * (1.0 / (1.0 - p));
		}
	}
}


Dropout3d::Dropout3d(int d1, int d2, int d3, int batchSize, double p) {
	this->d1 = d1;
	this->d2 = d2;
	this->d3 = d3;
	maxBatchSize = batchSize;
	this->p = p;

	output.resize(maxBatchSize);
	mask.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(d1);
		mask[b].resize(d1);
		diff[b].resize(d1);
		for (int i1 = 0; i1 < d1; i1++) {
			output[b][i1].resize(d2);
			mask[b][i1].resize(d2);
			diff[b][i1].resize(d2);
			for (int i2 = 0; i2 < d2; i2++) {
				output[b][i1][i2].resize(d3);
				mask[b][i1][i2].resize(d3);
				diff[b][i1][i2].resize(d3);
			}
		}
	}
}

void Dropout3d::forward(vector<vector<vector<vector<double>>>>& input, bool training) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					mask[b][i1][i2][i3] = 1;
					if (training) {
						double r = (double)rand() / (double)RAND_MAX;
						if (r < p) {
							mask[b][i1][i2][i3] = 0;
						}
					}
					double v = 1;
					if (training) {
						v /= (1.0 - p);
					}
					output[b][i1][i2][i3] = input[b][i1][i2][i3] * mask[b][i1][i2][i3] * v;
				}
			}
		}
	}
}

void Dropout3d::backward(vector<vector<vector<vector<double>>>>& input,
	vector<vector<vector<vector<double>>>>& nextDiff) {
	int batchSize = input.size();
	for (int b = 0; b < batchSize; b++) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					diff[b][i1][i2][i3] = 
						nextDiff[b][i1][i2][i3] * mask[b][i1][i2][i3] * (1.0 / (1.0 - p));
				}
			}
		}
	}
}


BatchNormalization1d::BatchNormalization1d(int d, int batchSize) {
	this->d = d;
	maxBatchSize = batchSize;

	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
	for (int b = 0; b < maxBatchSize; b++) {
		output[b].resize(d);
		diff[b].resize(d);
	}

	muLearn.resize(d);
	varLearn.resize(d);
	stdLearn.resize(d);
	muLearnDiff.resize(d);
	varLearnDiff.resize(d);
	mu.resize(d);
	std.resize(d);

	gamma.resize(d);
	beta.resize(d);
	gammaDiff.resize(d);
	betaDiff.resize(d);

	for (int i = 0; i < d; i++) {
		gamma[i] = 1.0;
		std[i] = 1.0;
	}
}

void BatchNormalization1d::forward(vector<vector<double>>& input, bool training) {
	int batchSize = input.size();
	if (training) {
		for (int i = 0; i < d; i++) {
			double summ = 0, sumOfSquares = 0;
			for (int b = 0; b < batchSize; b++) {
				summ += input[b][i];
				sumOfSquares += pow(input[b][i], 2);
			}
			muLearn[i] = summ / (double)batchSize;
			varLearn[i] = sumOfSquares / (double)batchSize - pow(muLearn[i], 2) + epsilon;
			stdLearn[i] = sqrt(varLearn[i]);
			mu[i] = 0.9 * mu[i] + 0.1 * muLearn[i];
			std[i] = 0.9 * std[i] + 0.1 * stdLearn[i];
		}
	}

	for (int i = 0; i < d; i++) {
		for (int b = 0; b < batchSize; b++) {
			if (training) {
				output[b][i] = (input[b][i] - muLearn[i]) / stdLearn[i] * gamma[i] + beta[i];
			}
			else {
				output[b][i] = (input[b][i] - mu[i]) / std[i] * gamma[i] + beta[i];
			}
		}
	}
}

void BatchNormalization1d::backward(vector<vector<double>>& input,
	vector<vector<double>>& nextDiff) {
	int batchSize = input.size();

	for (int i = 0; i < d; i++) {
		muLearnDiff[i] = 0;
		varLearnDiff[i] = 0;
		for (int b = 0; b < batchSize; b++) {
			gammaDiff[i] += nextDiff[b][i] * (input[b][i] - muLearn[i]) / stdLearn[i];
			betaDiff[i] += nextDiff[b][i];
			muLearnDiff[i] += -gamma[i] / stdLearn[i] * nextDiff[b][i];
			varLearnDiff[i] += -gamma[i] * (input[b][i] - muLearn[i]) / 2.0 /
				pow(stdLearn[i], 3) * nextDiff[b][i];
		}
	}

	for (int i = 0; i < d; i++) {
		for (int b = 0; b < batchSize; b++) {
			diff[b][i] = nextDiff[b][i] * gamma[i] / stdLearn[i] +
				varLearnDiff[i] * 2.0 / (double)batchSize * (input[b][i] - muLearn[i]) +
				muLearnDiff[i] / (double)batchSize;
		}
	}
}

void BatchNormalization1d::zeroGradients() {
	for (int i = 0; i < d; i++) {
		gammaDiff[i] = 0;
		betaDiff[i] = 0;
	}
}

void BatchNormalization1d::save(string fileName) {
	ofstream file(fileName);
	for (int i = 0; i < d; i++) {
		file << gamma[i] << " " << beta[i] << " " << mu[i] << " " << std[i] << " ";
	}
	file.close();
}

void BatchNormalization1d::load(string fileName) {
	ifstream file(fileName);
	for (int i = 0; i < d; i++) {
		file >> gamma[i] >> beta[i] >> mu[i] >> std[i];
	}
	file.close();
}


BatchNormalization3d::BatchNormalization3d(int d1, int d2, int d3, int batchSize) {
	this->d1 = d1;
	this->d2 = d2;
	this->d3 = d3;
	maxBatchSize = batchSize;

	output.resize(maxBatchSize);
	diff.resize(maxBatchSize);
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

	muLearn.resize(d1);
	varLearn.resize(d1);
	stdLearn.resize(d1);
	muLearnDiff.resize(d1);
	varLearnDiff.resize(d1);
	mu.resize(d1);
	std.resize(d1);
	gamma.resize(d1);
	beta.resize(d1);
	gammaDiff.resize(d1);
	betaDiff.resize(d1);
	for (int i1 = 0; i1 < d1; i1++) {
		muLearn[i1].resize(d2);
		varLearn[i1].resize(d2);
		stdLearn[i1].resize(d2);
		muLearnDiff[i1].resize(d2);
		varLearnDiff[i1].resize(d2);
		mu[i1].resize(d2);
		std[i1].resize(d2);
		gamma[i1].resize(d2);
		beta[i1].resize(d2);
		gammaDiff[i1].resize(d2);
		betaDiff[i1].resize(d2);
		for (int i2 = 0; i2 < d2; i2++) {
			muLearn[i1][i2].resize(d3);
			varLearn[i1][i2].resize(d3);
			stdLearn[i1][i2].resize(d3);
			muLearnDiff[i1][i2].resize(d3);
			varLearnDiff[i1][i2].resize(d3);
			mu[i1][i2].resize(d3);
			std[i1][i2].resize(d3);
			gamma[i1][i2].resize(d3);
			beta[i1][i2].resize(d3);
			gammaDiff[i1][i2].resize(d3);
			betaDiff[i1][i2].resize(d3);
			for (int i3 = 0; i3 < d3; i3++) {
				gamma[i1][i2][i3] = 1;
				std[i1][i2][i3] = 1;
			}
		}
	}
}

void BatchNormalization3d::forward(vector<vector<vector<vector<double>>>>& input, 
	bool training) {
	int batchSize = input.size();
	if (training) {
		for (int i1 = 0; i1 < d1; i1++) {
			for (int i2 = 0; i2 < d2; i2++) {
				for (int i3 = 0; i3 < d3; i3++) {
					double summ = 0, sumOfSquares = 0;
					for (int b = 0; b < batchSize; b++) {
						summ += input[b][i1][i2][i3];
						sumOfSquares += pow(input[b][i1][i2][i3], 2);
					}
					muLearn[i1][i2][i3] = summ / (double)batchSize;
					varLearn[i1][i2][i3] = sumOfSquares / (double)batchSize -
						pow(muLearn[i1][i2][i3], 2) + epsilon;
					stdLearn[i1][i2][i3] = sqrt(varLearn[i1][i2][i3]);
					mu[i1][i2][i3] = 0.9 * mu[i1][i2][i3] + 0.1 * muLearn[i1][i2][i3];
					std[i1][i2][i3] = 0.9 * std[i1][i2][i3] + 0.1 * stdLearn[i1][i2][i3];
				}
			}
		}
	}

	for (int i1 = 0; i1 < d1; i1++) {
		for (int i2 = 0; i2 < d2; i2++) {
			for (int i3 = 0; i3 < d3; i3++) {
				for (int b = 0; b < batchSize; b++) {
					if (training) {
						output[b][i1][i2][i3] = (input[b][i1][i2][i3] - muLearn[i1][i2][i3]) / 
							stdLearn[i1][i2][i3] * gamma[i1][i2][i3] + beta[i1][i2][i3];
					}
					else {
						output[b][i1][i2][i3] = (input[b][i1][i2][i3] - mu[i1][i2][i3]) /
							std[i1][i2][i3] * gamma[i1][i2][i3] + beta[i1][i2][i3];
					}
				}
			}
		}
	}
}

void BatchNormalization3d::backward(vector<vector<vector<vector<double>>>>& input,
	vector<vector<vector<vector<double>>>>& nextDiff) {
	int batchSize = nextDiff.size();

	for (int i1 = 0; i1 < d1; i1++) {
		for (int i2 = 0; i2 < d2; i2++) {
			for (int i3 = 0; i3 < d3; i3++) {
				muLearnDiff[i1][i2][i3] = 0;
				varLearnDiff[i1][i2][i3] = 0;
				for (int b = 0; b < batchSize; b++) {
					gammaDiff[i1][i2][i3] += nextDiff[b][i1][i2][i3] * 
						(input[b][i1][i2][i3] - muLearn[i1][i2][i3]) / stdLearn[i1][i2][i3];
					betaDiff[i1][i2][i3] += nextDiff[b][i1][i2][i3];
					muLearnDiff[i1][i2][i3] += -gamma[i1][i2][i3] / stdLearn[i1][i2][i3] * 
						nextDiff[b][i1][i2][i3];
					varLearnDiff[i1][i2][i3] += -gamma[i1][i2][i3] * 
						(input[b][i1][i2][i3] - muLearn[i1][i2][i3]) / 2.0 /
						pow(stdLearn[i1][i2][i3], 3) * nextDiff[b][i1][i2][i3];
				}
			}
		}
	}

	for (int i1 = 0; i1 < d1; i1++) {
		for (int i2 = 0; i2 < d2; i2++) {
			for (int i3 = 0; i3 < d3; i3++) {
				for (int b = 0; b < batchSize; b++) {
					diff[b][i1][i2][i3] = nextDiff[b][i1][i2][i3] * gamma[i1][i2][i3] / 
						stdLearn[i1][i2][i3] + varLearnDiff[i1][i2][i3] * 2.0 / 
						(double)batchSize * (input[b][i1][i2][i3] - muLearn[i1][i2][i3]) +
						muLearnDiff[i1][i2][i3] / (double)batchSize;
				}
			}
		}
	}
}

void BatchNormalization3d::zeroGradients() {
	for (int i1 = 0; i1 < d1; i1++) {
		for (int i2 = 0; i2 < d2; i2++) {
			for (int i3 = 0; i3 < d3; i3++) {
				gammaDiff[i1][i2][i3] = 0;
				betaDiff[i1][i2][i3] = 0;
			}
		}
	}
}

void BatchNormalization3d::save(string fileName) {
	ofstream file(fileName);
	for (int i1 = 0; i1 < d1; i1++) {
		for (int i2 = 0; i2 < d2; i2++) {
			for (int i3 = 0; i3 < d3; i3++) {
				file << gamma[i1][i2][i3] << " " << beta[i1][i2][i3] << " " << 
					mu[i1][i2][i3] << " " << std[i1][i2][i3] << " ";
			}
		}
	}
	file.close();
}

void BatchNormalization3d::load(string fileName) {
	ifstream file(fileName);
	for (int i1 = 0; i1 < d1; i1++) {
		for (int i2 = 0; i2 < d2; i2++) {
			for (int i3 = 0; i3 < d3; i3++) {
				file >> gamma[i1][i2][i3] >> beta[i1][i2][i3] >> mu[i1][i2][i3] >> 
					std[i1][i2][i3];
			}
		}
	}
	file.close();
}