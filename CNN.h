#pragma once

#include <vector>
#include <string>
#include "layers.h"
#include "lossFunctions.h"
#include "optimizers.h"

class CNN
{
private:
	int maxBatchSize = 100;
	vector<int> imageSize = { 1, 28, 28 };
	vector<int> convChannels = { 6, 16 };
	int poolingSize = 2;
	vector<int> fullLayersSize = { 80, 60 };
	double reluParameter = 0.001;
	double dropoutParameter = 0.3;
	bool useNormalization = true;
	int kernelSize = 5;
	double alpha = 0.003, beta1 = 0.9, beta2 = 0.99;
	double l2 = 0.1;
	int flattenSize = convChannels[1] * (imageSize[1] / poolingSize / poolingSize) *
		(imageSize[2] / poolingSize / poolingSize);
	int nOfOutputs = 10;

	ConvolutionalLayer conv1 = ConvolutionalLayer(imageSize[1], imageSize[2], imageSize[0], 
		convChannels[0], maxBatchSize, kernelSize);
	BatchNormalization3d norm1 = BatchNormalization3d(convChannels[0], imageSize[1],
		imageSize[2], maxBatchSize);
	ReLU3d a1 = ReLU3d(convChannels[0], imageSize[1], imageSize[2], maxBatchSize, reluParameter);
	MaxPooling pool1 = MaxPooling(convChannels[0], imageSize[1], imageSize[2], maxBatchSize, 
		poolingSize);
	ConvolutionalLayer conv2 = ConvolutionalLayer(imageSize[1] / poolingSize, 
		imageSize[2] / poolingSize, convChannels[0], convChannels[1], maxBatchSize, kernelSize);
	BatchNormalization3d norm2 = BatchNormalization3d(convChannels[1], 
		imageSize[1] / poolingSize, imageSize[2] / poolingSize, maxBatchSize);
	ReLU3d a2 = ReLU3d(convChannels[1],
		imageSize[1] / poolingSize, imageSize[2] / poolingSize, maxBatchSize, reluParameter);
	MaxPooling pool2 = MaxPooling(convChannels[1], imageSize[1] / poolingSize, 
		imageSize[2] / poolingSize, maxBatchSize, poolingSize);
	Flatten31 flatten = Flatten31(convChannels[1], imageSize[1] / poolingSize / poolingSize, 
		imageSize[2] / poolingSize / poolingSize, maxBatchSize);
	BatchNormalization1d norm3 = BatchNormalization1d(flattenSize, maxBatchSize);
	Dropout1d drop1 = Dropout1d(flattenSize, maxBatchSize, dropoutParameter);
	LinearLayer linear1 = LinearLayer(flattenSize, fullLayersSize[0], maxBatchSize);
	Sigmoid1d a3 = Sigmoid1d(fullLayersSize[0], maxBatchSize);
	Dropout1d drop2 = Dropout1d(fullLayersSize[0], maxBatchSize, dropoutParameter);
	LinearLayer linear2 = LinearLayer(fullLayersSize[0], fullLayersSize[1], maxBatchSize);
	ReLU1d a4 = ReLU1d(fullLayersSize[1], maxBatchSize, reluParameter);
	LinearLayer linear3 = LinearLayer(fullLayersSize[1], nOfOutputs, maxBatchSize);
	Softmax softmax = Softmax(nOfOutputs, maxBatchSize);
	CategoricalCrossentropyLoss loss = CategoricalCrossentropyLoss(nOfOutputs, maxBatchSize);

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
		alpha, beta1, beta2, 0);
	Adam1d norm3g = Adam1d(flattenSize, alpha, beta1, beta2, 0);
	Adam1d norm3b = Adam1d(flattenSize, alpha, beta1, beta2, 0);
	Adam2d linear1w = Adam2d(flattenSize, fullLayersSize[0], alpha, beta1, beta2, l2);
	Adam1d linear1b = Adam1d(fullLayersSize[0], alpha, beta1, beta2, 0);
	Adam2d linear2w = Adam2d(fullLayersSize[0], fullLayersSize[1], alpha, beta1, beta2, l2);
	Adam1d linear2b = Adam1d(fullLayersSize[1], alpha, beta1, beta2, 0);
	Adam2d linear3w = Adam2d(fullLayersSize[1], nOfOutputs, alpha, beta1, beta2, l2);
	Adam1d linear3b = Adam1d(nOfOutputs, alpha, beta1, beta2, 0);

	double lossValue = 0;
	vector<vector<double>> output;

public:
	void forward(vector<vector<vector<vector<double>>>>& images, int batchSize, bool training) {
		conv1.forward(images, batchSize);
		if (useNormalization) {
			norm1.forward(conv1.output, batchSize, training);
			a1.forward(norm1.output, batchSize);
		}
		else {
			a1.forward(conv1.output, batchSize);
		}
		pool1.forward(a1.output, batchSize);
		conv2.forward(pool1.output, batchSize);
		if (useNormalization) {
			norm2.forward(conv2.output, batchSize, training);
			a2.forward(norm2.output, batchSize);
		}
		else {
			a2.forward(conv2.output, batchSize);
		}
		pool2.forward(a2.output, batchSize);
		flatten.forward(pool2.output, batchSize);
		if (useNormalization) {
			norm3.forward(flatten.output, batchSize, training);
			drop1.forward(norm3.output, batchSize, training);
		}
		else {
			drop1.forward(flatten.output, batchSize, training);
		}
		linear1.forward(drop1.output, batchSize);
		a3.forward(linear1.output, batchSize);
		drop2.forward(a3.output, batchSize, training);
		linear2.forward(drop2.output, batchSize);
		a4.forward(linear2.output, batchSize);
		linear3.forward(a4.output, batchSize);

		softmax.forward(linear3.output, batchSize);
		output = softmax.output;
	}
	void backward(vector<vector<vector<vector<double>>>>& images,
		vector<int>& targets, int batchSize) {
		loss.calculate(linear3.output, targets);
		lossValue = loss.value;

		linear3.backward(a4.output, loss.diff, batchSize);
		a4.backward(linear2.output, linear3.diff, batchSize);
		linear2.backward(drop2.output, a4.diff, batchSize);
		drop2.backward(a3.output, linear2.diff, batchSize);
		a3.backward(linear1.output, drop2.diff, batchSize);
		linear1.backward(drop1.output, a3.diff, batchSize);
		if (useNormalization) {
			drop1.backward(norm3.output, linear1.diff, batchSize);
			norm3.backward(flatten.output, drop1.diff, batchSize);
			flatten.backward(pool2.output, norm3.diff, batchSize);
		}
		else {
			drop1.backward(flatten.output, linear1.diff, batchSize);
			flatten.backward(pool2.output, drop1.diff, batchSize);
		}
		pool2.backward(a2.output, flatten.diff, batchSize);
		if (useNormalization) {
			a2.backward(norm2.output, pool2.diff, batchSize);
			norm2.backward(conv2.output, a2.diff, batchSize);
			conv2.backward(pool1.output, norm2.diff, batchSize);
		}
		else {
			a2.backward(conv2.output, pool2.diff, batchSize);
			conv2.backward(pool1.output, a2.diff, batchSize);
		}
		pool1.backward(a1.output, conv2.diff, batchSize);
		if (useNormalization) {
			a1.backward(norm1.output, pool1.diff, batchSize);
			norm1.backward(conv1.output, a1.diff, batchSize);
			conv1.backward(images, norm1.diff, batchSize);
		}
		else {
			a1.backward(conv1.output, pool1.diff, batchSize);
			conv1.backward(images, a1.diff, batchSize);
		}
		
	}
	void updateParameters() {
		conv1w.step(conv1.weights, conv1.weightsDiff);
		conv1b.step(conv1.biases, conv1.biasesDiff);
		conv2w.step(conv2.weights, conv2.weightsDiff);
		conv2b.step(conv2.biases, conv2.biasesDiff);
		linear1w.step(linear1.weights, linear1.weightsDiff);
		linear1b.step(linear1.biases, linear1.biasesDiff);
		linear2w.step(linear2.weights, linear2.weightsDiff);
		linear2b.step(linear2.biases, linear2.biasesDiff);
		linear3w.step(linear3.weights, linear3.weightsDiff);
		linear3b.step(linear3.biases, linear3.biasesDiff);
		if (useNormalization) {
			norm1g.step(norm1.gamma, norm1.gammaDiff);
			norm1b.step(norm1.beta, norm1.betaDiff);
			norm2g.step(norm2.gamma, norm2.gammaDiff);
			norm2b.step(norm2.beta, norm2.betaDiff);
			norm3g.step(norm3.gamma, norm3.gammaDiff);
			norm3b.step(norm3.beta, norm3.betaDiff);
			norm1.zeroGradients();
			norm2.zeroGradients();
			norm3.zeroGradients();
		}

		conv1.zeroGradients();
		conv2.zeroGradients();
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

