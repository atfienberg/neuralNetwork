#include "NeuralNetwork.hh"
#include <iostream>
#include <vector>
#include <algorithm>
#include "TRandom3.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TFile.h"
#include "TSpline.h"
#include "TStyle.h"
#include "TAxis.h"

#include <ctime>


using namespace std;

typedef vector<double> pulse;
typedef vector<double> outcome;

const double NOISE = 0.01;
const double RNDMOFFSET = 0.05;
const double AMPPARAMETER = 1;//amplitude parameter

const int NPTS = 20;

const int NPTSPERFORMANCEGRAPH = 30;

//return delta t
double addPulse(vector<pulse>& pulses, vector<outcome>& outcomes, TRandom3& rand,
		TSpline3* s, bool toZero);

int main(){  
  gStyle->SetOptStat(0);
  //try to distinguish triangle, gaussian, and sipm-like function (with noise)
  NeuralNetwork net(vector<int>({NPTS,3,2}));
  net.setEta(0.001);
  net.setMu(0.5);
  net.setLambda(0.000001);
  TRandom3 rand(0);
  //build training set
  TFile* f = new TFile("../sipmTemplates/ledTemplateForDoublePulse.root");
  TSpline3* spline = (TSpline3*) f->Get("masterSpline");
  TFile* outf = new TFile("pileupSeparationOut.root", "recreate");

  vector<pulse> trainingPulses;
  vector<outcome> desiredTrainingOutcomes;
  for(int i = 0; i < 50000; ++i){
    addPulse(trainingPulses, desiredTrainingOutcomes, rand, spline, false);
  }
  
  //build testing set
  vector< pulse > testPulses;
  vector< outcome > testOutcomes;
  for(int i = 0; i < 2000; ++i){
    addPulse(testPulses, testOutcomes, rand, spline, false);
  }
  //  cout << "Num epochs? " << endl;
  int nEpochs = 15;
  
  TGraph* fractionCorrect = new TGraph(0);
  fractionCorrect->SetName("fractionCorrect");
  TGraph* costGraph = new TGraph(0);
  costGraph->SetName("costGraph");

  double bestFraction = 0;

  for (int epoch = 0; epoch < nEpochs; ++epoch){
    cout << "Begin epoch " << epoch << endl;
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
    double fracCorrect = static_cast<double>(nRight) / testPulses.size();
    if(fracCorrect > bestFraction){
      bestFraction = fracCorrect;
    }
    cout << "Cost : " << cost / testPulses.size() << endl;
    cout << "Fraction correct: " << fracCorrect << endl;
    fractionCorrect->SetPoint(epoch, epoch, fracCorrect);
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
  fractionCorrect->GetXaxis()->SetTitle("epoch #");
  fractionCorrect->GetYaxis()->SetTitle("fraction correct");
  fractionCorrect->Write();
  c1->Print("learningPileup.pdf");

  costGraph->Draw("ap");
  costGraph->SetMarkerStyle(20);
  costGraph->Write();
  c1->Print("costGraphPileup.pdf"); 

  cout << "Test final training group, new" << endl;
  testPulses.resize(0);
  testOutcomes.resize(0);
  vector<double> deltaTs;
  for(int i = 0; i < 1000000; ++i){
    deltaTs.push_back(addPulse(testPulses, testOutcomes, rand, spline, true));
  }
  int nRight = 0;
  int nMissedSingles = 0;

  vector<double> netOutput;
  
  TH1D* singleHist = new TH1D("singleHist","singleHist",1000,0,1);
  TH1D* doubleHist = new TH1D("doubleHist","doubleHist",1000,0,1);
  TH1D* doubleHistOver1 = new TH1D("doubleHisto1","doubleHisto1",1000,0,1);
  
  doubleHist->SetLineColor(kRed);
  doubleHistOver1->SetLineColor(kRed);

  vector<double> successes(NPTSPERFORMANCEGRAPH, 0);
  vector<double> totals(NPTSPERFORMANCEGRAPH, 0);
  clock_t startTime = clock();
  for(unsigned int i = 0; i < testPulses.size(); ++i){
    netOutput = net.process(testPulses[i]);
    bool right = false;
    bool foundSingle = netOutput[0] > 0.5;
    bool wasSingle = testOutcomes[i][0] == 1;

    if( foundSingle == wasSingle){
      nRight++;
      right = true;
    }
    if(testOutcomes[i][0] == 1){
      singleHist->Fill(netOutput[0]);
    }
    else{
      doubleHist->Fill(netOutput[0]);
      if(deltaTs[i] > 1){
	doubleHistOver1->Fill(netOutput[0]);
      }
      totals[deltaTs[i]/6.0*NPTSPERFORMANCEGRAPH]++;
      if(right){
	successes[deltaTs[i]/6.0*NPTSPERFORMANCEGRAPH]++;
      }
    }
    if((!right) && (testOutcomes[i][0] == 1)){
      nMissedSingles++;
    }      
  }
  cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << endl;
  //create time performance graph
  TGraph* performanceGraph = new TGraph(0);
  performanceGraph->SetName("performanceGraph");
  for(unsigned int i = 0; i < successes.size(); ++i){
    performanceGraph->SetPoint(i, i*6.0/NPTSPERFORMANCEGRAPH, 
			       static_cast<double>(successes[i])/totals[i]);
  }
  performanceGraph->GetXaxis()->SetTitle("delta t [ns]");
  performanceGraph->GetYaxis()->SetTitle("success fraction");
  performanceGraph->SetMarkerStyle(20);
  performanceGraph->Draw("ap");
  performanceGraph->Write();
  c1->Print("performanceGraphPileup.pdf");

  doubleHist->Draw();
  singleHist->Draw("same");
  c1->SetLogy(1);
  c1->Print("pileupSingleHist.pdf");
  doubleHistOver1->Draw();
  singleHist->Draw("same");
  c1->Print("pileupSingleHistOver1ns.pdf");

  cout << "final performance: " << static_cast<double>(nRight)/testPulses.size() << endl;
  cout << "n missed singles: " << nMissedSingles << endl;
  outf->Write();
}
 
double addPulse(vector<pulse>& pulses, vector<outcome>& outcomes, TRandom3& rand, TSpline3* s, bool toZero){
  outcome thisOutcome(2, 0.0);
  pulse thisPulse(NPTS);
  double amp;
  do{
    amp = rand.Exp(AMPPARAMETER);
  } while (amp < 0.25);
  double deltaT = 0;
  double offset = rand.Gaus(0, RNDMOFFSET)-5;
  for(int i = 0; i < NPTS; ++i){
    thisPulse[i] = amp*s->Eval(i*1.25+offset) + rand.Gaus(0, NOISE);
  }
  if(rand.Rndm() > 0.5){
    double offset1 = offset;
    if(toZero){
      offset = offset - 6*gRandom->Rndm();
    }
    else{
      offset = offset - 4.75*gRandom->Rndm()-1.25;
    }
    deltaT = offset1 - offset;
    do{
      amp = rand.Exp(AMPPARAMETER);
    } while (amp < 0.25);
    for(int i = 0; i < NPTS; ++i){
      thisPulse[i] += amp*s->Eval(i*1.25+offset);
    }
    thisOutcome[1] = 1.0;
  }
  else{
    thisOutcome[0] = 1.0;
  }      
  pulses.push_back(thisPulse);
  outcomes.push_back(thisOutcome);
  return deltaT;
}


