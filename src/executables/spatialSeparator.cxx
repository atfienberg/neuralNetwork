#include "NeuralNetwork.hh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include "TRandom3.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"
#include "TLine.h"
#include "TSpline.h"
#include "TStyle.h"
#include "TTree.h"
#include "TAxis.h"

#include "TApplication.h"
#include "TSystem.h"

#include <ctime>


using namespace std;

typedef vector<double> dep;
typedef vector<double> outcome;

void displayCluster(const dep& caloDep);

const int NPTSPERFORMANCEGRAPH = 30;

//return whether it succeeded (fails after running out of entries)
bool addDeps(vector<dep>& deps, vector<outcome>& outcomes, TTree* t, int& entry, bool trainingCuts = true);

int main(){  
  //new TApplication("app", 0, nullptr);
  gStyle->SetOptStat(0);
  //try to distinguish triangle, gaussian, and sipm-like function (with noise)
  NeuralNetwork net(vector<int>({54,54,2}));
  net.setEta(0.0001);
  net.setMu(0.2);
  net.setLambda(0.0000001);

  //prepare tree to be read
  TFile* f = new TFile("../datafiles/clusterData.root");
  TTree* t = (TTree*) f->Get("t");
  TFile* outf = new TFile("spatialSeparatorOut.root", "recreate");
  int entryCounter = 0;
  vector<dep> trainingDeps;
  vector<outcome> desiredTrainingOutcomes;
  for(int i = 0; i < 100000; ++i){
    if(!addDeps(trainingDeps, desiredTrainingOutcomes, t, entryCounter)){
      cout << "Error: not enough entries to build training set" << endl;
      exit(EXIT_SUCCESS);
    }
  }
  cout << trainingDeps.size() << endl;
  cout << desiredTrainingOutcomes.size() << endl;
  cout << entryCounter << endl;
  
  cout << "Training deps made." << endl;

  //build testing set
  vector< dep > testDeps;
  vector< outcome > testOutcomes;
  for(int i = 0; i < 10000; ++i){
    if(!addDeps(testDeps, testOutcomes, t, entryCounter)){
      cout << "Error: not enough entries to build testing set" << endl;
      exit(EXIT_SUCCESS);
    }
  }
  //  cout << "Num epochs? " << endl;
  int nEpochs = 1000;
  
  TGraph* fractionCorrect = new TGraph(0);
  fractionCorrect->SetName("fractionCorrect");
  TGraph* costGraph = new TGraph(0);
  costGraph->SetName("costGraph");

  vector<TLine*> newLearningRateLines;

  double bestFraction = 0;
  int nSinceBest = 0;
  int nSinceBestOrUpdate = 0; //whichever is shorter
  NeuralNetwork bestNet = net;
  
  for (int epoch = 0; epoch < nEpochs; ++epoch){
    cout << "Begin epoch " << epoch << endl;
    cout << "Testing..." << endl;
    double cost = 0;
    int nRight = 0;
    vector<double> netOutput;
    for(unsigned int i = 0; i < testDeps.size(); ++i){
      cost += net.getCost(testDeps[i], testOutcomes[i], netOutput);
      
      //did we classify correctly?
      if( (max_element(testOutcomes[i].begin(), testOutcomes[i].end()) - testOutcomes[i].begin()) 
	  ==
	  (max_element(netOutput.begin(), netOutput.end()) - netOutput.begin())){
	nRight++;
      }      
    }
    
    //output diagnostics
    double fracCorrect = static_cast<double>(nRight) / testDeps.size();
    if(fracCorrect > bestFraction){
      bestFraction = fracCorrect;
      bestNet = net;
      nSinceBest = 0;
      nSinceBestOrUpdate = 0;
    }
    else{
      nSinceBest++;
      nSinceBestOrUpdate++;
    }
    if(nSinceBestOrUpdate > 10){
      nSinceBestOrUpdate = 0;
      cout << "updating eta ... " << endl;
      cout << "old eta: " << net.getEta() << endl;;
      newLearningRateLines.push_back(new TLine(epoch, 0.0, epoch, 1.0));
      newLearningRateLines.back()->SetLineColor(kRed);
      newLearningRateLines.back()->SetLineWidth(2);
      net.setEta(net.getEta() / 2.0);
      cout << "new eta: " << net.getEta() << endl;
    }
    if(nSinceBest > 30){
      cout << "stopping." << endl;
      break;
    }
      
    cout << "Cost : " << cost / testDeps.size() << endl;
    cout << "Fraction correct: " << fracCorrect << endl;
    fractionCorrect->SetPoint(epoch, epoch, fracCorrect);
    costGraph->SetPoint(epoch, epoch, cost / testDeps.size() );  
    cout << "training ... " << endl;
    for(unsigned int i = 0; i < trainingDeps.size(); ++i){
      net.learnOnline(trainingDeps[i], desiredTrainingOutcomes[i]);
    }
    cout << endl;
  }
  net = std::move(bestNet);

  TCanvas* c1 = new TCanvas("c1","c1");
  fractionCorrect->SetMarkerStyle(20);
  fractionCorrect->Draw("ap");
  fractionCorrect->GetXaxis()->SetTitle("epoch #");
  fractionCorrect->GetYaxis()->SetTitle("fraction correct");
  double yMin = 0;
  double yMax = 1;
  fractionCorrect->GetYaxis()->SetRangeUser(yMin, yMax);
  fractionCorrect->Write(); 
  
  for(auto& entry : newLearningRateLines){
    entry->SetY1(yMin);
    entry->SetY2(yMax);
    entry->Draw("same");
  }
  c1->Print("learningSpatial.pdf");

  costGraph->Draw("ap");
  costGraph->SetMarkerStyle(20);
  costGraph->Write();
  c1->Print("costGraphSpatial.pdf"); 
  
  cout << "Test final training group, new" << endl;
  testDeps.resize(0);
  testOutcomes.resize(0);
  vector<double> dists;
  vector<double> ratios;

  for(int i = 0; i < 200000; ++i){
    if(!addDeps(testDeps, testOutcomes, t, entryCounter, false)){
      cout << "Error: not enough entries to build testing set" << endl;
      exit(EXIT_SUCCESS);
    }
  }
  cout << "filled final test group" << endl;

  int nRight = 0;
  int nDoubles = 0;
  int nSingles = 0;
  int nWrongSingles = 0;
  int nWrongDoubles = 0;

  vector<double> netOutput;
  TH1D* singleHist = new TH1D("singleHist","singleHist",100,0,1);
  TH1D* doubleHist = new TH1D("doubleHist","doubleHist",100,0,1);
  doubleHist->SetLineColor(kRed); 

  clock_t startTime = clock();
  for(unsigned int i = 0; i < testDeps.size(); ++i){
    netOutput = net.process(testDeps[i]);
    bool foundSingle = netOutput[0] > netOutput[1];
    bool wasSingle = testOutcomes[i][0] == 1;
    if(wasSingle){
      nSingles++;
      singleHist->Fill(netOutput[0]);
    }
    else{
      nDoubles++;
      doubleHist->Fill(netOutput[0]);
    }
    if( foundSingle == wasSingle){
      nRight++;
    }
    else if(wasSingle){      
      nWrongSingles++;
    }
    else{
      /*      cout << "FAILED DOUBLE!" << endl;
      cout << "OUTPUT: " << netOutput[0] << endl;
      displayCluster(testDeps[i]);*/
      nWrongDoubles++;
    }
  }
  cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << endl;
  cout << "n singles : " << nSingles << endl;
  cout << "n doubles : " << nDoubles << endl;
  cout << "fraction right: " << static_cast<double>(nRight)/(nSingles + nDoubles) << endl;
  cout << "f wrong singles: " << static_cast<double>(nWrongSingles)/(nSingles) << endl;
  cout << "f wrong doubles: " << static_cast<double>(nWrongDoubles)/(nDoubles) << endl;

  doubleHist->Draw();
  singleHist->Draw("same");
  c1->SetLogy(1);
  c1->Print("spatialOutputHist.pdf");

  outf->Write();
}
 
bool addDeps(vector<dep>& deps, vector<outcome>& outcomes, TTree* t, int& entry, bool trainingCuts){ 
  outcome thisOutcome(2, 0.0);
  dep thisDep(54);
    
  double energies[54];
  double totalDep;
  int nPositrons;
  double pEnergies[3];
  double pPositions[6];
  t->SetBranchAddress("positronEnergies", pEnergies);
  t->SetBranchAddress("crystalDeps", energies);
  t->SetBranchAddress("totalDep", &totalDep);
  t->SetBranchAddress("nPositrons", &nPositrons);
  t->SetBranchAddress("positronPositions", pPositions);
  double dist = 100;
  bool tooClose = false;
  bool tooLowEnergy = false;
  bool totalDepTooLow = false;
  do{
    if(entry >= t->GetEntries()){
      return false;
    }
    t->GetEntry(entry++);
    if(nPositrons == 2){
      dist = sqrt((pPositions[0]-pPositions[2])*(pPositions[0]-pPositions[2]) + (pPositions[1]-pPositions[3])*(pPositions[1]-pPositions[3]));
    }
    if(trainingCuts){
      tooClose = dist < 50;
      tooLowEnergy = *min_element(pEnergies, pEnergies + nPositrons) < 150;
      totalDepTooLow = totalDep < 1000;
    }
    else{
      totalDepTooLow = totalDep < 1680;
    }
  } while( (totalDepTooLow) 
	   || (tooLowEnergy)
	   || (tooClose));

  for(int i = 0; i < 54; ++i){
    thisDep[i] = energies[i];
  }

  if(nPositrons == 1){
    thisOutcome[0] = 1.0;
  }
  else{
    thisOutcome[1] = 1.0;
  }
  deps.push_back(thisDep);
  outcomes.push_back(thisOutcome);
  return true;
}

void displayCluster(const dep& caloDep){
  
  
  TCanvas* c2 = new TCanvas("c2", "c2", 1600,1000);
  
  gStyle->SetOptStat(0);
  c2->Divide(1, 2);
  c2->cd(1);
  
  int maxdex = max_element(caloDep.begin(), caloDep.end()) - caloDep.begin();
  int maxRow = maxdex / 9;
  int maxCol = maxdex % 9;

  TGraph* g = new TGraph(0);
  TH2D* depQuilt = new TH2D("depQuilt", "depQuilt", 9, 0, 9, 6, 0, 6);
  for(int row = 0; row < 6; ++row){
    for(int col = 0; col < 9; ++col){
      g->SetPoint(g->GetN(), sqrt((row-maxRow)*(row-maxRow) + (col-maxCol)*(col-maxCol)), caloDep[row*9 + col]);	 
      depQuilt->Fill(col + 0.5, 5.5 - row, caloDep[9*row + col]);
    }
  }
  
  g->SetMarkerStyle(20);
  g->GetXaxis()->SetTitle("Distance From Max");
  g->GetYaxis()->SetTitle("E / E max");
  g->SetTitle("");
  g->Draw("ap");
  c2->cd(2);
  depQuilt->Draw("colz");
  c2->Draw();
  c2->Modified();
  c2->Update();
  gSystem->ProcessEvents();
  cin.ignore();
  delete depQuilt;
  delete g;
  delete c2;
  gStyle->SetOptStat(1);
}
  
