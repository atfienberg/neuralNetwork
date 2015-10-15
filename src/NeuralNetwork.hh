/*
  Aaron Fienberg
  First attempt at coding up a neural network.
  uses momentum, L2 regularization,
  softmax ouput layer with log likelihood
  This has completely free weights with all neurons connected to all neurons in next layer
*/
#pragma once

#include <vector>

class NeuralNetwork{
public:
  //takes vector of ints, the number of neurons in each layer
  NeuralNetwork(std::vector<int> sizes, double eta = 0.01);

  //Read in neural network from a file created with saveToFile (json format)
  NeuralNetwork(const char* filename);

  //to get this going I'll start with just this function: 
  //does one pass through, updates weights, biases, velocity 
  void learnOnline(const std::vector<double>& inputs, const std::vector<double>& desiredOutputs);

  //returns cost given input and desired output
  //puts output in output vector
  double getCost(const std::vector<double>& inputs, 
		 const std::vector<double>& desiredOutputs,
		 std::vector<double>& output);


  //a version not returning the output
  double getCost(const std::vector<double>& inputs, const std::vector<double>& desiredOutputs);

  //process input values, get output values as return value
  std::vector<double> process(const std::vector<double>& inputValues);
  
  //write neural net to a file (json format)
  void saveToFile(const char* filename) const;

  //basic getters and setters
  void setEta(double e) { eta_ = e; }
  double getEta() const { return eta_; }

  void setLambda(double l) { lambda_ = l; }
  double getLambda() const { return lambda_; }

  void setMu(double m) { mu_ = m; }
  double getMu() const { return mu_; }
  
  //change activation function
  void setSigmoid();
  void setRectifiedLinear();

  void dump() const;

  //typedefs
  typedef std::vector<double> row;
  typedef std::vector<double> col;
  typedef std::vector<row> matrix;
  typedef unsigned int index;

  //to later allow for other activation functions, like linear rect
  typedef double (*activationFunction)(double); //activation function
  typedef double (*activationFunctionPrime)(double); //activation function prime

private:
  
  void initializeWeightsAndBiases(const std::vector<int>& sizes, bool randomize = true);
  //initialize as_, zs_, deltas_
  void initializeNeurons(const std::vector<int>& sizes);
 
  void feedForward(const std::vector<double>& inputValues);

  double weightNormSquared();

  //updates deltas with back propagation
  //only to be called after feed forward has been run in inputs to go with desired output
  void updateDeltas(const std::vector<double>& desiredOutput);

  //updates velocities from current deltas
  //there should also be way to do it from given matrix of gradients (for minibatches)
  //that's a to do for later if I decide I nede it 
  void updateVelocitiesFromDelta();

  //updates weights and biases using current velocity matrix. 
  void updateWeightsAndBiases();

  //use the log likelihood
  static double cost(const std::vector<double>& desired, const std::vector<double>& found);

  //softmax applies softmax in place to final layer
  void applySoftMax();
  
  static void inconsistentJsonError();

  //sigmoid 
  static double sigmoid(double z);
  //derivative of sigmoid
  static double sigPrime(double z);

  //recified linear
  static double rectifiedLinear(double z);
  static double rectifiedLinearPrime(double z);

  double eta_; //learning parameter
  double lambda_; //L2 regularization parameter
  double mu_; //momentum coefficient
  std::vector<matrix> weightMatrices_;
  std::vector<matrix> velocityMatrices_;
  std::vector<col> biasVectors_;
  std::vector<col> biasVelocityVectors_;
  std::vector<col> as_; //activations. Member variable so I don't have to reallocate
  std::vector<col> zs_; //input to activations. Member variable so I don't have to reallocate
  std::vector<col> deltas_; //errors at neurons (dC/dz). same as above

  activationFunction act_;
  activationFunctionPrime actPrime_;
};
