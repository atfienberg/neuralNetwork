#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include <algorithm>
#include <memory>
#include <limits>
#include <ctime>

#include "TString.h"
#include "TRandom3.h"

#include "json11.hpp"

#include "NeuralNetwork.hh"

/*In general, remember weight matrix 0 connects layers 0 and 1, 1 layers 1 and
  2, etc.
  Similarly, bias vector 0 pertains to layer 1, bias vector 1 pertains to layer
  2.*/

// takes vector of ints, the number of neurons in each layer
NeuralNetwork::NeuralNetwork(std::vector<int> sizes, double eta)
    : eta_(eta), lambda_(0), mu_(0), weightMatrices_(0), biasVectors_(0) {
  // must have at least input and output layer
  assert(sizes.size() > 1);

  setSigmoid();

  initializeWeightsAndBiases(sizes);
  initializeNeurons(sizes);
}

// read neural network in from json file
NeuralNetwork::NeuralNetwork(const char* filename) {
  // start by reading in file and parsing json into a map
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    std::cerr << "Couldn't open " << filename << "!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::stringstream ss;
  ss << infile.rdbuf();
  infile.close();
  std::string err;
  auto netJson = json11::Json::parse(ss.str(), err);
  if (err.size() != 0) {
    std::cerr << "Parsing error for " << filename << " : " << err << std::endl;
    exit(EXIT_FAILURE);
  }
  auto netMap = netJson.object_items();

  // setup the network

  // learning parameters
  auto learningParametersMap = netMap.at("learning parameters").object_items();
  eta_ = learningParametersMap.at("eta").number_value();
  mu_ = learningParametersMap.at("mu").number_value();
  lambda_ = learningParametersMap.at("lambda").number_value();

  // neurons
  std::vector<int> sizes;
  for (const auto& entry : netMap.at("sizes").array_items()) {
    sizes.push_back(entry.int_value());
  }
  initializeWeightsAndBiases(sizes, false);
  initializeNeurons(sizes);

  // activation function
  if (netMap.at("activation function").string_value() == "sigmoid") {
    setSigmoid();
  } else if (netMap.at("activation function").string_value() ==
             "rectified linear") {
    setRectifiedLinear();
  } else {
    std::cerr << "Error: activation function "
              << netMap.at("activation function").string_value()
              << " not recognized" << std::endl;
    exit(EXIT_FAILURE);
  }

  // weights and biases
  for (index i = 0; i < sizes.size() - 1; ++i) {
    auto thisInterface =
        netMap.at("interface " + std::to_string(i)).object_items();

    // biases
    auto thisJsonBiasVector = thisInterface.at("biases").array_items();
    if (thisJsonBiasVector.size() != biasVectors_[i].size()) {
      inconsistentJsonError();
    }
    for (index j = 0; j < thisJsonBiasVector.size(); ++j) {
      biasVectors_[i][j] = thisJsonBiasVector[j].number_value();
    }

    // weights
    auto thisJsonWeightMatrix = thisInterface.at("weight matrix").array_items();
    if (weightMatrices_[i].size() != thisJsonWeightMatrix.size()) {
      inconsistentJsonError();
    }
    for (index j = 0; j < thisJsonWeightMatrix.size(); ++j) {
      if (weightMatrices_[i][j].size() !=
          thisJsonWeightMatrix[j].array_items().size()) {
        inconsistentJsonError();
      }
      for (index k = 0; k < thisJsonWeightMatrix[j].array_items().size(); ++k) {
        weightMatrices_[i][j][k] = thisJsonWeightMatrix[j][k].number_value();
      }
    }
  }
}

std::vector<double> NeuralNetwork::process(
    const std::vector<double>& inputValues) {
  feedForward(inputValues);
  return as_.back();
}

void NeuralNetwork::saveToFile(const char* filename) const {
  json11::Json::object outputMap;

  // basic info
  outputMap["filename"] = filename;
  outputMap["what"] = "saved neural network";
  // get time
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[80];
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 80, "%c", timeinfo);
  outputMap["when"] = buffer;

  // info about specific network
  std::vector<int> sizes;
  for (const auto& layer : as_) {
    sizes.push_back(layer.size());
  }
  outputMap["sizes"] = sizes;
  if (act_ == sigmoid) {
    outputMap["activation function"] = "sigmoid";
  } else if (act_ == rectifiedLinear) {
    outputMap["activation function"] = "rectified linear";
  }

  outputMap["learning parameters"] =
      json11::Json::object{{"eta", eta_}, {"mu", mu_}, {"lambda", lambda_}};

  for (index i = 0; i < weightMatrices_.size(); ++i) {
    std::string interfaceLabel = "interface " + std::to_string(i);
    outputMap[interfaceLabel.c_str()] = json11::Json::object{
        {"weight matrix", weightMatrices_[i]}, {"biases", biasVectors_[i]}};
  }

  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Error opening output file " << filename
              << ", network not saved. " << std::endl;
    return;
  }
  out << json11::Json(outputMap).dump();
  out.close();
}

void NeuralNetwork::dump() const {
  std::cout << "weight matrices" << std::endl;
  for (index i = 0; i < weightMatrices_.size(); ++i) {
    std::cout << "weight matrix " << i + 1 << ": " << std::endl;
    for (const auto& row : weightMatrices_[i]) {
      for (const auto& value : row) {
        std::cout << Form("%.5f", value) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "bias vector " << i + 1 << ": " << std::endl;
    for (const auto& value : biasVectors_[i]) {
      std::cout << Form("%.2f", value) << std::endl;
    }
  }

  std::cout << "Velocities : " << std::endl;
  for (index i = 0; i < velocityMatrices_.size(); ++i) {
    std::cout << "weight velocity matrix " << i + 1 << ": " << std::endl;
    for (const auto& row : velocityMatrices_[i]) {
      for (const auto& value : row) {
        std::cout << Form("%.5f", value) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "bias velocity vector " << i + 1 << ": " << std::endl;
    for (const auto& value : biasVelocityVectors_[i]) {
      std::cout << Form("%.5f", value) << std::endl;
    }
  }

  std::cout << "as then zs then deltas" << std::endl;
  const std::vector<col>* pCol[3] = {&as_, &zs_, &deltas_};
  for (auto ptr : pCol) {
    for (const auto& entry : *ptr) {
      for (const auto& v : entry) {
        std::cout << v << " ";
      }
      std::cout << std::endl
                << "--------" << std::endl;
    }
  }
}

void NeuralNetwork::learnOnline(const std::vector<double>& inputs,
                                const std::vector<double>& desiredOutput) {
  feedForward(inputs);
  updateDeltas(desiredOutput);
  updateVelocitiesFromDelta();
  updateWeightsAndBiases();
}

double NeuralNetwork::getCost(const std::vector<double>& inputs,
                              const std::vector<double>& desiredOutputs) {
  feedForward(inputs);
  double c0 = cost(desiredOutputs, as_.back());
  double regTerm = 0;
  if (lambda_ > 0) {
    regTerm = 0.5 * lambda_ * weightNormSquared();
  }
  return c0 + regTerm;
}

double NeuralNetwork::getCost(const std::vector<double>& inputs,
                              const std::vector<double>& desiredOutputs,
                              std::vector<double>& output) {
  double cost = getCost(inputs, desiredOutputs);
  output = as_.back();
  return cost;
}

void NeuralNetwork::setSigmoid() {
  act_ = sigmoid;
  actPrime_ = sigPrime;
}

void NeuralNetwork::setRectifiedLinear() {
  act_ = rectifiedLinear;
  actPrime_ = rectifiedLinearPrime;
}

// private helper functions

void NeuralNetwork::initializeWeightsAndBiases(const std::vector<int>& sizes,
                                               bool randomize) {
  std::unique_ptr<TRandom3> rand;
  if (randomize) {
    rand.reset(new TRandom3(0));
  }

  weightMatrices_.resize(sizes.size() - 1);
  velocityMatrices_.resize(sizes.size() - 1);
  biasVectors_.resize(sizes.size() - 1);
  biasVelocityVectors_.resize(sizes.size() - 1);

  for (index matrixNum = 1; matrixNum < sizes.size(); ++matrixNum) {
    // check if output layer, which never gets randomized
    bool outputLayer = (matrixNum == (sizes.size() - 1));
    bool randomizeLayer = (randomize && (!outputLayer));

    matrix& thisMatrix = weightMatrices_[matrixNum - 1];
    matrix& thisVMatrix = velocityMatrices_[matrixNum - 1];
    col& thisVector = biasVectors_[matrixNum - 1];
    col& thisVVector = biasVelocityVectors_[matrixNum - 1];

    // weight matrix nTo by nFrom
    thisMatrix.resize(sizes[matrixNum]);
    thisVMatrix.resize(sizes[matrixNum]);
    thisVector.resize(sizes[matrixNum]);
    thisVVector = col(sizes[matrixNum], 0);
    for (int rowNum = 0; rowNum < sizes[matrixNum]; ++rowNum) {
      thisMatrix[rowNum].resize(sizes[matrixNum - 1]);
      thisVMatrix[rowNum] = row(sizes[matrixNum - 1], 0);
      for (int colNum = 0; colNum < sizes[matrixNum - 1]; ++colNum) {
        double startWeightValue =
            randomizeLayer ? rand->Gaus(0, 1) / std::sqrt(sizes[matrixNum - 1])
                           : 0;
        thisMatrix[rowNum][colNum] = startWeightValue;
      }  // for colNum
      double startBiasValue = randomizeLayer ? rand->Gaus(0, 1) : 0;
      thisVector[rowNum] = startBiasValue;
    }  // for rowNum
  }    // for matrixnum
}

void NeuralNetwork::initializeNeurons(const std::vector<int>& sizes) {
  std::vector<col>* pCol[3] = {&as_, &zs_, &deltas_};
  for (auto ptr : pCol) {
    for (auto size : sizes) {
      ptr->emplace_back(size, 0);
    }
  }

  // zs_[0] won't be used, may as well make it empty
  zs_[0].resize(0);
  // same with deltas[0]
  deltas_[0].resize(0);
}

void NeuralNetwork::feedForward(const std::vector<double>& inputValues) {
  // make sure number of inputs equals number of input nodes
  assert(inputValues.size() == as_[0].size());

  // as are activations
  for (index i = 0; i < inputValues.size(); ++i) {
    as_[0][i] = inputValues[i];
  }

  // zs are inputs to activations
  for (index layer = 0; layer < weightMatrices_.size(); ++layer) {
    for (index i = 0; i < zs_[layer + 1].size(); ++i) {
      zs_[layer + 1][i] = biasVectors_[layer][i];
      for (index j = 0; j < as_[layer].size(); ++j) {
        zs_[layer + 1][i] += weightMatrices_[layer][i][j] * as_[layer][j];
      }
    }
    // activation depends on whether we're in the last layer
    if (layer != weightMatrices_.size() - 1) {
      // not last layer, use assigned activation function
      for (index i = 0; i < as_[layer + 1].size(); ++i) {
        as_[layer + 1][i] = act_(zs_[layer + 1][i]);
      }
    } else {
      // last layer, use softmax
      assert(layer == weightMatrices_.size() - 1);
      applySoftMax();
    }
  }
}

double NeuralNetwork::weightNormSquared() {
  double runningSum = 0;
  for (const auto& matrix : weightMatrices_) {
    for (const auto& r : matrix) {
      for (const auto& value : r) {
        runningSum += value * value;
      }
    }
  }
  return runningSum;
}

void NeuralNetwork::updateDeltas(const std::vector<double>& desiredOutput) {
  assert(desiredOutput.size() == as_.back().size());
  // start it off with last layer
  for (index i = 0; i < desiredOutput.size(); ++i) {
    deltas_.back()[i] = as_.back()[i] - desiredOutput[i];
  }

  // now rest of layers using back propagation
  for (index i = weightMatrices_.size() - 1; i > 0; --i) {  // loop over layers
    for (index j = 0; j < deltas_[i].size();
         ++j) {  // loop over neurons in layer
      deltas_[i][j] = 0;
      for (index k = 0; k < deltas_[i + 1].size();
           ++k) {  // sum over deltas in next layer
        deltas_[i][j] += deltas_[i + 1][k] * weightMatrices_[i][k][j];
      }  // end sum over next layer deltas
      deltas_[i][j] *= actPrime_(zs_[i][j]);
    }  // end loop over neurons
  }    // end loop over layers
}

// v_i is equal to
// mu * v_i - eta * grad(C)_i
void NeuralNetwork::updateVelocitiesFromDelta() {
  // loop over layers other than input layer
  for (index layer = 1; layer < deltas_.size(); ++layer) {
    // loop over neurons
    for (index i = 0; i < deltas_[layer].size(); ++i) {
      biasVelocityVectors_[layer - 1][i] =
          mu_ * biasVelocityVectors_[layer - 1][i] - eta_ * deltas_[layer][i];

      // update velocity for each weight connected to this neuron
      for (index j = 0; j < as_[layer - 1].size(); ++j) {
        velocityMatrices_[layer - 1][i][j] =
            mu_ * velocityMatrices_[layer - 1][i][j] -
            eta_ * as_[layer - 1][j] * deltas_[layer][i];
      }

    }  // end loop over neurons
  }    // end loop over layers
}

void NeuralNetwork::updateWeightsAndBiases() {
  for (index i = 0; i < weightMatrices_.size(); ++i) {
    for (index j = 0; j < weightMatrices_[i].size(); ++j) {
      biasVectors_[i][j] += biasVelocityVectors_[i][j];

      for (index k = 0; k < weightMatrices_[i][j].size(); ++k) {
        weightMatrices_[i][j][k] =
            (1 - eta_ * lambda_) * weightMatrices_[i][j][k] +
            velocityMatrices_[i][j][k];
      }
    }
  }
}

// the log likelihood. Assumes desired is always 1 in 1 position and 0's
// everywhere else
double NeuralNetwork::cost(const std::vector<double>& desired,
                           const std::vector<double>& found) {
  // desired same length as number of output neurons
  assert(desired.size() == found.size());

  for (index i = 0; i < desired.size(); ++i) {
    if (desired[i] == 1) {
      if (found[i] != 0) {
        return -1 * std::log(found[i]);
      } else {
        return std::numeric_limits<double>::max();
      }
    }
  }
  std::cout << "Test data had no classification!" << std::endl;
  assert(false);
  return -1;
}

void NeuralNetwork::inconsistentJsonError() {
  std::cerr << "Error: json file inconsistent." << std::endl;
  exit(EXIT_FAILURE);
}

double NeuralNetwork::sigmoid(double z) { return 1.0 / (1 + std::exp(-1 * z)); }
double NeuralNetwork::sigPrime(double z) {
  double s = sigmoid(z);
  return s * (1 - s);
}
double NeuralNetwork::rectifiedLinear(double z) { return z > 0 ? z : 0; }
double NeuralNetwork::rectifiedLinearPrime(double z) { return z > 0 ? 1 : 0; }

// old implementation with overflow checking
/*//as are activations, zs are weighted and biased inputs
  void NeuralNetwork::applySoftMax(){
  for(index i = 0; i < zs_.back().size(); ++i){
  double expValue = std::exp(zs_.back()[i]);
  if(std::isinf(expValue)){
  as_.back()[i] = std::numeric_limits<double>::max();
  }
  else{
  as_.back()[i] = expValue;
  }
  }
  double sum = std::accumulate(as_.back().begin(), as_.back().end(), 0.0);
  if(std::isinf(sum)){
  sum = std::numeric_limits<double>::max();
  }
  std::for_each(as_.back().begin(), as_.back().end(), [&](double& val){
  val = val / sum ;
  } );
  }*/

// this evaluation helps with numerical issues
void NeuralNetwork::applySoftMax() {
  double max = *std::max_element(zs_.back().begin(), zs_.back().end());
  double logOperand = 0;
  for (index i = 0; i < zs_.back().size(); ++i) {
    logOperand += std::exp(zs_.back()[i] - max);
  }
  double log = std::log(logOperand);
  for (index i = 0; i < as_.back().size(); ++i) {
    as_.back()[i] = exp(zs_.back()[i] - log - max);
  }
}
