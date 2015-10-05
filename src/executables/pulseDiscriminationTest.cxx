#include "NeuralNetwork.hh"
#include <iostream>
#include <vector>
#include <algorithm>
#include "TRandom3.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TAxis.h"

using namespace std;

double trianglePulse(double t);
double gaussianPulse(double t);
double sipmLike(double t);

typedef vector<double> pulse;
typedef vector<double> outcome;

const double NOISE = 0.01;
const double RNDMOFFSET = 0.05;
const double AMPPARAMETER = 1;//amplitude parameter

const int NPTS = 20;

void addPulse(vector<pulse>& pulses, vector<outcome>& outcomes, TRandom3& rand);

int main(){  
  //try to distinguish triangle, gaussian, and sipm-like function (with noise)
  NeuralNetwork net(vector<int>({NPTS,3}));
  net.setEta(0.1);
  //  net.setMu(0.3);
  //  net.setLambda(0.00001);
  TRandom3 rand(0);
  //build training set
  vector<pulse> trainingPulses;
  vector<outcome> desiredTrainingOutcomes;
  for(int i = 0; i < 10000; ++i){
    addPulse(trainingPulses, desiredTrainingOutcomes, rand);
  }
  
  //build testing set
  vector< pulse > testPulses;
  vector< outcome > testOutcomes;
  for(int i = 0; i < 100; ++i){
    addPulse(testPulses, testOutcomes, rand);
  }
  //  cout << "Num epochs? " << endl;
  int nEpochs = 5;
  
  TGraph* fractionCorrect = new TGraph(0);
  fractionCorrect->SetName("fractionCorrect");
  TGraph* costGraph = new TGraph(0);
  fractionCorrect->SetName("costGraph");

  for (int epoch = 0; epoch < nEpochs; ++epoch){
    cout << "Begin group " << epoch << endl;
    cout << "Testing..." << endl;
    double cost = 0;
    int nRight = 0;
    vector<double> netOutput;
    for(unsigned int i = 0; i < testPulses.size(); ++i){
      cost += net.getCost(testPulses[i], testOutcomes[i], netOutput);
      
      //did we classify correctly?
      if( (max_element(testOutcomes[i].begin(), testOutcomes[i].end()) - testOutcomes[i].begin()) 
	  ==
	  (max_element(netOutput.begin(), netOutput.end()) - netOutput.begin())){
	nRight++;
      }      
    }
    
    //output diagnostics
    
    cout << "Cost : " << cost / testPulses.size() << endl;
    cout << "Fraction correct: " << static_cast<double>(nRight) / testPulses.size() << endl;
    fractionCorrect->SetPoint(epoch, epoch, static_cast<double>(nRight) / testPulses.size());
    costGraph->SetPoint(epoch, epoch, cost / testPulses.size() );

    cout << "training ... " << endl;
    for(unsigned int i = 0; i < trainingPulses.size(); ++i){
      net.learnOnline(trainingPulses[i], desiredTrainingOutcomes[i]);
    }
    cout << endl;
  }
  TCanvas* c1 = new TCanvas("c1","c1");
  fractionCorrect->SetMarkerStyle(20);
  fractionCorrect->Draw("ap");
  fractionCorrect->GetXaxis()->SetTitle("group #");
  fractionCorrect->GetYaxis()->SetTitle("fraction correct");
  c1->Print("learningDiscrimination.pdf");

  costGraph->Draw("ap");
  costGraph->SetMarkerStyle(20);
  c1->Print("costGraphDiscrimination.pdf");

  cout << "final test" << endl;
  
  cout << "build triangle" << endl;
  pulse triangle(20);
  double offset = rand.Gaus(0, RNDMOFFSET);
  for(unsigned int i = 0; i < NPTS; ++i){
    triangle[i] = 0.5*(trianglePulse(i*10.0/NPTS+offset) + rand.Gaus(0, NOISE));
  }
  vector<double> output = net.process(triangle);
  cout << "triangle outputs: " << endl;
  for(unsigned int i = 0; i < output.size(); ++i){
    cout << output[i] << " ";
  }
  cout << endl;

  cout << "build gaussian" << endl;
  pulse gaussian(20);
  offset = rand.Gaus(0, RNDMOFFSET);
  for(unsigned int i = 0; i < NPTS; ++i){
    gaussian[i] = 0.5*(gaussianPulse(i*10.0/NPTS+offset) + rand.Gaus(0, NOISE));
  }
  output = net.process(gaussian);
  cout << "gaussian outputs: " << endl;
  for(unsigned int i = 0; i < output.size(); ++i){
    cout << output[i] << " ";
  }
  cout << endl;

  cout << "build sipm" << endl;
  pulse sipm(20);
  offset = rand.Gaus(0, RNDMOFFSET);
  for(unsigned int i = 0; i < NPTS; ++i){
    sipm[i] = 0.5*(sipmLike(i*10.0/NPTS+offset) + rand.Gaus(0, NOISE));
  }
  output = net.process(sipm);
  cout << "sipm outputs: " << endl;
  for(unsigned int i = 0; i < output.size(); ++i){
    cout << output[i] << " ";
  }
  cout << endl;
  

}
 
void addPulse(vector<pulse>& pulses, vector<outcome>& outcomes, TRandom3& rand){
  double (*pulseShape)(double);

  outcome thisOutcome(3, 0);
  pulse thisPulse(NPTS);

  double randNum = rand.Rndm();
  if(randNum < 1.0/3.0){
    pulseShape = trianglePulse;
    thisOutcome[0] = 1;
  }
  else if(randNum < 2.0/3.0){
    pulseShape = gaussianPulse;
    thisOutcome[1] = 1;
  }
  else{
    pulseShape = sipmLike;
    thisOutcome[2] = 1;
  }
  double offset = rand.Gaus(0, RNDMOFFSET);
  double amp;
  do{
    amp = rand.Exp(AMPPARAMETER);
  } while (amp < 0.25);
    
  for(int i = 0; i < NPTS; ++i){
    thisPulse[i] = amp*pulseShape(i*10.0/NPTS+offset) + rand.Gaus(0, NOISE);
  }
  
  pulses.push_back(thisPulse);
  outcomes.push_back(thisOutcome);
}
    
double trianglePulse(double t){
  double absTerm = -abs((t-5.0)/3.2) + 1;
  return absTerm > 0 ? absTerm : 0;
}

double gaussianPulse(double t){
  return exp(-(t-5)*(t-5)/4);
}

double sipmLike(double t){
  if( t < 3.5){
    return 0;
  }
  return 12.5*(exp(-(t-3.5)/1.25)*(1-exp(-(t-3.5)/5.0)));
}
