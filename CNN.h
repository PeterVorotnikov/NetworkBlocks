#pragma once

#include <vector>
#include <string>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

class CNN
{
private:
	int batchSize = 2;
	vector<int> imageSize = { 1, 28, 28 };
	vector<int> convChannels = { 6, 16 };
	int poolingSize = 2;
	vector<int> fullLayersSize = { 80, 60 };
	double reluParameter = 0.001;
	double dropoutParameter = 0.5;
	int kernelSize = 5;
	double alpha = 0.01, beta1 = 0.9, beta2 = 0.99;
	double l2 = 0;
	int flattenSize = convChannels[1] * (imageSize[1] / poolingSize / poolingSize) *
		(imageSize[2] / poolingSize / poolingSize);
	int nOfOutputs = 10;

	ConvolutionalLayer conv1 = ConvolutionalLayer(imageSize[1], imageSize[2], imageSize[0], 
		convChannels[0], batchSize, kernelSize);
	BatchNormalization3d norm1 = BatchNormalization3d(convChannels[0], imageSize[1],
		imageSize[2], batchSize);
	ReLU3d a1 = ReLU3d(convChannels[0], imageSize[1], imageSize[2], batchSize, reluParameter);
	MaxPooling pool1 = MaxPooling(convChannels[0], imageSize[1], imageSize[2], batchSize, 
		poolingSize);
	ConvolutionalLayer conv2 = ConvolutionalLayer(imageSize[1] / poolingSize, 
		imageSize[2] / poolingSize, convChannels[0], convChannels[1], batchSize, kernelSize);
	BatchNormalization3d norm2 = BatchNormalization3d(convChannels[1], 
		imageSize[1] / poolingSize, imageSize[2] / poolingSize, batchSize);
	ReLU3d a2 = ReLU3d(convChannels[1],
		imageSize[1] / poolingSize, imageSize[2] / poolingSize, batchSize, reluParameter);
	MaxPooling pool2 = MaxPooling(convChannels[1], imageSize[1] / poolingSize, 
		imageSize[2] / poolingSize, batchSize, poolingSize);
	Flatten31 flatten = Flatten31(convChannels[1], imageSize[1] / poolingSize / poolingSize, 
		imageSize[2] / poolingSize / poolingSize, batchSize);
	Dropout1d drop1 = Dropout1d(flattenSize, batchSize, dropoutParameter);
	LinearLayer linear1 = LinearLayer(flattenSize, fullLayersSize[0], batchSize);
	ReLU1d a3 = ReLU1d(fullLayersSize[0], batchSize, reluParameter);
	Dropout1d drop2 = Dropout1d(fullLayersSize[0], batchSize, dropoutParameter);
	LinearLayer linear2 = LinearLayer(fullLayersSize[0], fullLayersSize[1], batchSize);
	Sigmoid1d a4 = Sigmoid1d(fullLayersSize[1], batchSize);
	Dropout1d drop3 = Dropout1d(fullLayersSize[1], batchSize, dropoutParameter);
	LinearLayer linear3 = LinearLayer(fullLayersSize[1], nOfOutputs, batchSize);
	Softmax softmax = Softmax(nOfOutputs, batchSize);
	CategoricalCrossentropyLoss loss = CategoricalCrossentropyLoss(nOfOutputs, batchSize);

	Adam4d conv1w = Adam4d(convChannels[0], imageSize[0], kernelSize, kernelSize, 
		alpha, beta1, beta2, l2);
	Adam1d conv1b = Adam1d(convChannels[0], alpha, beta1, beta2, 0);
	Adam3d norm1g = Adam3d(convChannels[0], imageSize[1], imageSize[2], 
		alpha, beta1, beta2, 0);
	Adam3d norm1b = Adam3d(convChannels[0], imageSize[1], imageSize[2],
		alpha, beta1, beta2, 0);
	Adam4d conv2w = Adam4d(convChannels[1], convChannels[0], kernelSize, kernelSize,
		alpha, beta1, beta2, l2);
	Adam1d conv2b = Adam1d(convChannels[1], alpha, beta1, beta2, 0);
	Adam3d norm2g = Adam3d(convChannels[1], imageSize[1] / 2, imageSize[2] / 2,
		alpha, beta1, beta2, 0);
	Adam3d norm2b = Adam3d(convChannels[1], imageSize[1] / 2, imageSize[2] / 2,
		alpha, beta1, beta2, 0);;
	Adam2d linear1w = Adam2d(flattenSize, fullLayersSize[0], alpha, beta1, beta2, l2);
	Adam1d linear1b = Adam1d(fullLayersSize[0], alpha, beta1, beta2, 0);
	Adam2d linear2w = Adam2d(fullLayersSize[0], fullLayersSize[1], alpha, beta1, beta2, l2);
	Adam1d linear2b = Adam1d(fullLayersSize[1], alpha, beta1, beta2, 0);
	Adam2d linear3w = Adam2d(fullLayersSize[1], nOfOutputs, alpha, beta1, beta2, l2);
	Adam1d linear3b = Adam1d(nOfOutputs, alpha, beta1, beta2, 0);

	double lossValue = 0;
	vector<vector<double>> output;

public:
	void forward(vector<vector<vector<vector<double>>>>& images, bool training) {
		conv1.forward(images);
		norm1.forward(conv1.output, training);
		a1.forward(norm1.output);
		pool1.forward(a1.output);
		conv2.forward(pool1.output);
		norm2.forward(conv2.output, training);
		a2.forward(norm2.output);
		pool2.forward(a2.output);
		flatten.forward(pool2.output);
		drop1.forward(flatten.output, training);
		linear1.forward(drop1.output);
		a3.forward(linear1.output);
		drop2.forward(a3.output, training);
		linear2.forward(drop2.output);
		a4.forward(linear2.output);
		drop3.forward(a4.output);
		linear3.forward(drop3.output);

		softmax.forward(linear3.output);
		output = softmax.output;
	}
	void backward(vector<vector<vector<vector<double>>>>& images, vector<int>& targets) {
		loss.calculate(linear3.output, targets);
		lossValue = loss.value;

		linear3.backward(drop3.output, loss.diff);
		drop3.backward(a4.output, linear3.diff);
		a4.backward(linear2.output, drop3.diff);
		linear2.backward(drop2.output, a4.diff);
		drop2.backward(a3.output, linear1.diff);
		a3.backward(linear1.output, drop2.diff);
		linear1.backward(drop1.output, a3.diff);
		drop1.backward(flatten.output, linear1.diff);
		flatten.backward(pool2.output, drop1.diff);
		pool2.backward(a2.output, flatten.diff);
		a2.backward(norm2.output, pool2.diff);
		norm2.backward(conv2.output, a2.diff);
		conv2.backward(pool1.output, norm2.diff);
		pool1.backward(a1.output, conv2.diff);
		a1.backward(norm1.output, pool1.diff);
		norm1.backward(conv1.output, a1.diff);
		conv1.backward(images, norm1.diff);
	}
	void updateParameters() {
		conv1w.step(conv1.weights, conv1.weightsDiff);
		conv1b.step(conv1.biases, conv1.biasesDiff);
		norm1g.step(norm1.gamma, norm1.gammaDiff);
		norm1b.step(norm1.beta, norm1.betaDiff);
		conv2w.step(conv2.weights, conv2.weightsDiff);
		conv2b.step(conv2.biases, conv2.biasesDiff);
		norm2g.step(norm2.gamma, norm2.gammaDiff);
		norm2b.step(norm2.beta, norm2.betaDiff);
		linear1w.step(linear1.weights, linear1.weightsDiff);
		linear1b.step(linear1.biases, linear1.biasesDiff);
		linear2w.step(linear2.weights, linear2.weightsDiff);
		linear2b.step(linear2.biases, linear2.biasesDiff);
		linear3w.step(linear3.weights, linear3.weightsDiff);
		linear3b.step(linear3.biases, linear3.biasesDiff);

		conv1.zeroGradients();
		norm1.zeroGradients();
		conv2.zeroGradients();
		norm2.zeroGradients();
		linear1.zeroGradients();
		linear2.zeroGradients();
		linear3.zeroGradients();
	}
	double getLoss() {
		return lossValue;
	}
	vector<vector<double>> getOutputs() {
		return output;
	}

	void save(string fileName) {
		conv1.save(fileName + "conv1.txt");
		norm1.save(fileName + "norm1.txt");
		conv2.save(fileName + "conv2.txt");
		norm2.save(fileName + "norm2.txt");
		linear1.save(fileName + "linear1.txt");
		linear2.save(fileName + "linear2.txt");
		linear3.save(fileName + "linear3.txt");

		conv1w.save(fileName + "conv1w.txt");
		conv1b.save(fileName + "conv1b.txt");
		norm1g.save(fileName + "norm1g.txt");
		norm1b.save(fileName + "norm1b.txt");
		conv2w.save(fileName + "conv2w.txt");
		conv2b.save(fileName + "conv2b.txt");
		norm2g.save(fileName + "norm2g.txt");
		norm2b.save(fileName + "norm2b.txt");
		linear1w.save(fileName + "linear1w.txt");
		linear1b.save(fileName + "linear1b.txt");
		linear2w.save(fileName + "linear2w.txt");
		linear2b.save(fileName + "linear2b.txt");
		linear3w.save(fileName + "linear3w.txt");
		linear3b.save(fileName + "linear3b.txt");
	}
	void load(string fileName) {
		conv1.load(fileName + "conv1.txt");
		norm1.load(fileName + "norm1.txt");
		conv2.load(fileName + "conv2.txt");
		norm2.load(fileName + "norm2.txt");
		linear1.load(fileName + "linear1.txt");
		linear2.load(fileName + "linear2.txt");
		linear3.load(fileName + "linear3.txt");

		conv1w.load(fileName + "conv1w.txt");
		conv1b.load(fileName + "conv1b.txt");
		norm1g.load(fileName + "norm1g.txt");
		norm1b.load(fileName + "norm1b.txt");
		conv2w.load(fileName + "conv2w.txt");
		conv2b.load(fileName + "conv2b.txt");
		norm2g.load(fileName + "norm2g.txt");
		norm2b.load(fileName + "norm2b.txt");
		linear1w.load(fileName + "linear1w.txt");
		linear1b.load(fileName + "linear1b.txt");
		linear2w.load(fileName + "linear2w.txt");
		linear2b.load(fileName + "linear2b.txt");
		linear3w.load(fileName + "linear3w.txt");
		linear3b.load(fileName + "linear3b.txt");
	}
};

