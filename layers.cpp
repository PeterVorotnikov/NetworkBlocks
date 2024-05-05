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