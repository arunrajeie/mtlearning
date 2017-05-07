/*
 * MLearning.cpp
 * Copyright (c) 2016 Yasuo Tabei All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE and * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "MLearning.hpp"

using namespace std;

void MLearning::readfile1(ifstream &ifs) {
  datNames_.clear();
  
  string line;
  getline(ifs, line);
  stringstream ss(line);
  string fname;
#ifdef SKIPSIM
  uint64_t pt = 0;
#endif
  fecNames_.clear();
  while (ss >> fname) {
#ifdef SKIPSIM
    if (pt++ == 0)
      continue;
#endif // SKIPSIM
    fecNames_.push_back(fname);
  }
   
  numDat_ = 0;
  dim_ = 0;
  while (getline(ifs, line)) {
    numDat_++;
    fvs_.resize(fvs_.size() + 1);
    vector<pair<uint32_t, double> > &fv = fvs_[fvs_.size() - 1];
    stringstream ss(line);
    string name1, name2;
    ss >> name1;
    ss >> name2;
    datNames_.push_back(make_pair(name1, name2));
    uint32_t fid = 0;
    double val;
#ifdef SKIPSIM
    ss >> val; // for similarity
#endif // SKIPSIM
    while (ss >> val) {
      if (val != 0.0) 
	fv.push_back(make_pair(fid, val));
      fid++;
    }
    dim_ = fid;
#ifdef ERRORCHECK
    if (fid != fecNames_.size()) {
      cerr << "error 1 : " << fid << " " << fecNames_.size() << endl;
      exit(1);
    }
#endif
  }
  cerr << "num : " << numDat_ << " dim : " << dim_ << endl;
}

void MLearning::readfile2(ifstream &ifs) {
  string line;
  {
    getline(ifs, line);
    stringstream ss(line);
    string name;
    while (ss >> name) 
      taskNames_.push_back(name);
  }

#ifdef ERRORCHECK
  uint64_t pt = 0;
#endif // ERRORCHECK
  uint64_t labeldim = 0;
  while (getline(ifs, line)) {
    stringstream ss(line);
    labelMat_.resize(labelMat_.size() + 1);
    vector<int> &labelVec = labelMat_[labelMat_.size() - 1];
    string name1, name2;
    ss >> name1;
    ss >> name2;
   
#ifdef ERRORCHECK 
    if (datNames_[pt].first != name1 || datNames_[pt].second != name2) {
      cerr << "error 2 : " << datNames_[pt].first << " " << datNames_[pt].second << " " << name1 << " " << name2 << endl;
      exit(1);
    }
    pt++;
#endif // ERRORCHECK
      
    int val;
    while (ss >> val) {
      if (val != 0)  
	labelVec.push_back(1);
      else
	labelVec.push_back(0);
    }
    if (labeldim != 0 && labelVec.size() != labeldim) {
      cerr << "error : " << labeldim << " " << labelVec.size() << endl;
      exit(1);
    }
    labeldim = labelVec.size();
  }
  cerr << "labelMat_ : " << labelMat_.size() << " labeldim : " << labeldim << endl;
  numTask_ = labeldim;
}

void MLearning::readfile3(ifstream &ifs) {
  string line;
  getline(ifs, line);
#ifdef ERRORCHECK
  string name;
  uint64_t pt = 0;
  stringstream ss(line);
  while (ss >> name) {
    if (taskNames_[pt++] != name) {
      cerr << "error 3 : " << taskNames_[pt-1] << " " << name << endl;
      exit(1);
    }
  }
#endif // ERRORCHECK

  pt = 0;
  while (getline(ifs, line)) {
    stringstream ss(line);
    simMatY_.resize(simMatY_.size() + 1);
    vector<double> &simVecY = simMatY_[simMatY_.size() - 1];
    string name;
    ss >> name;
#ifdef ERRORCHECK
    if (taskNames_[pt++] != name) {
      cerr << "error 4 : " << taskNames_[pt-1] << " " << name << endl;
      exit(1);
    }
#endif // ERRORCHECK
    double val;
    while (ss >> val) 
      simVecY.push_back(val);
  }
#ifdef ERRORCHECK
  if (simMatY_.size() != simMatY_[0].size()) {
    cerr << "error 5 : " << simMatY_.size() << " " << simMatY_[0].size() << endl;
    exit(1);
  }
#endif
  
  cerr << "simmaty size : " << simMatY_.size() << " " << simMatY_[0].size() << endl;
}

void MLearning::readfile4(ifstream &ifs) {
  string line;
  getline(ifs, line);
#ifdef ERRORCHECK
  string name;
  uint64_t pt = 0;
  stringstream ss(line);
  while (ss >> name) {
    if (fecNames_[pt++] != name) {
      cerr << "error 5 : " << fecNames_[pt-1] << " " << name << endl;
      exit(1);
    }
  }
#endif // ERRORCHECK

  pt = 0;
  while (getline(ifs, line)) {
    stringstream ss(line);
    simMatX_.resize(simMatX_.size() + 1);
    vector<double> &simVecX = simMatX_[simMatX_.size() - 1];
    string name;
    ss >> name;
#ifdef ERRORCHECK
    if (fecNames_[pt++] != name) {
      cerr << "error 5 : " << fecNames_[pt-1] << " " << name << endl;
      exit(1);
    }
#endif
    double val;
    while (ss >> val)
      simVecX.push_back(val);
  }
  cerr << "simmatx size : " << simMatX_.size() << endl;
}

double MLearning::sigmoid(double v) {
  return 1.0/(1.0 + exp(-v));
}

void MLearning::convertSimToLaplacian(vector<vector<double> > &simMat, vector<vector<double> > &matLap) {
  vector<double> vecD(simMat.size());
  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < simMat.size(); ++i) {
    vector<double> &simVec = simMat[i];
    double sum = 0.0;
    for (size_t j = 0; j < simVec.size(); ++j) 
      sum += simVec[j];
    vecD[i] = sum;
  }

  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < simMat.size(); ++i) 
    simMat[i][i] = vecD[i] - simMat[i][i];

  vector<vector<double> > tmpMat(vecD.size());
  for (size_t i = 0; i < vecD.size(); ++i)
    tmpMat[i].resize(vecD.size());
  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < vecD.size(); ++i) {
    for (size_t j = 0; j < vecD.size(); ++j) {
      if (vecD[i] != 0.0)
	tmpMat[i][j] += simMat[i][j]/sqrt(vecD[i]);
    }
  }
  
  matLap.resize(vecD.size());
  for (size_t i = 0; i < matLap.size(); ++i)
    matLap[i].resize(vecD.size());
  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < vecD.size(); ++i) {
    for (size_t j = 0; j < vecD.size(); ++j) {
      if (vecD[j] != 0.0)
	matLap[i][j] += tmpMat[i][j]/sqrt(vecD[j]);
    }
  }
}

double MLearning::compObjectiveFun(vector<vector<double> > &weightMat) {
  vector<double> objs(numDat_);
  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < numDat_; ++i) {
    vector<int> &labelVec = labelMat_[i];
    vector<pair<uint32_t, double> > &fv = fvs_[i];
    for (size_t j = 0; j < numTask_; ++j) {
      vector<double> &weightVec = weightMat[j];
      double prod = 0.0;
      for (size_t k = 0; k < fv.size(); ++k) 
	prod += (weightVec[fv[k].first] * fv[k].second);
      double label = 1;
      if (labelVec[j] != 1)
	label = -1;
      /*
      #ifdef _PARALLEL_
        #pragma omp atomic
      #endif
      */
      double wLabel = 1.0;
      if (label != 1.0)
	wLabel = wLabelMinus_[j];
      else
	wLabel = wLabelPlus_[j];
      objs[i] += (wLabel*log(1.0 + exp(-label * prod)));
    }
  }

  double obj = 0.0;
  for (size_t i = 0; i < objs.size(); ++i)
    obj += objs[i]; 
  
  return obj/double(numDat_* numTask_);
}

double MLearning::compTrace1(vector<vector<double> > &weightMat, vector<vector<double> > &lapMat) {
  vector<vector<double> > tmpMat(dim_);
  for (size_t i = 0; i < dim_; ++i) 
    tmpMat[i].resize(numTask_);

  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < dim_; ++i) {
    for (size_t j = 0; j < numTask_; ++j) {
      for (size_t k = 0; k < numTask_; ++k)
	tmpMat[i][j] += weightMat[k][i] * lapMat[k][j];
    }
  }

  vector<vector<double> > tmpMat2(numTask_);
  for (size_t i = 0; i < numTask_; ++i)
    tmpMat2[i].resize(numTask_);
  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < numTask_; ++i) {
    for (size_t j = 0; j < numTask_; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < dim_; ++k)  
	sum += tmpMat[k][i] * weightMat[j][k];
      tmpMat2[i][j] = sum;
    }
  }

  double trace1 = 0.0;
  for (size_t i = 0; i < numTask_; ++i)
    trace1 += tmpMat2[i][i];
  trace1 *= 0.5;
  
  cerr << "trace1 : " << trace1 << endl;
  
  return trace1;
}

double MLearning::compTrace2(vector<vector<double> > &weightMat) {
  vector<double> tmpVec(numTask_);
  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t k = 0; k < numTask_; ++k) {
    double sum = 0.0;
    for (size_t i = 0; i < dim_; ++i) 
      sum += weightMat[k][i] * weightMat[k][i];
    tmpVec[k] = sum;
  }
    
  double trace2 = 0.0;
  for (size_t i = 0; i < numTask_; ++i)
    trace2 += tmpVec[i];
  trace2 *= 0.5;

  cerr << "trace2 : " << trace2 << endl;
  
  return trace2;
}

double MLearning::compTrace3(vector<vector<double> > &weightMat, vector<vector<double> > &lapMat) {
  vector<vector<double> > tmpMat(numTask_);
  for (size_t i = 0; i < numTask_; ++i)
    tmpMat[i].resize(dim_);

  for (size_t i = 0; i < numTask_; ++i) {
    for (size_t j = 0; j < dim_; ++j) {
      for (size_t k = 0; k < dim_; ++k) 
	tmpMat[i][j] += weightMat[i][k] * lapMat[k][j];
    }
  }

  vector<vector<double> > tmpMat2(numTask_);
  for (size_t i = 0; i < numTask_; ++i) {
    tmpMat2[i].resize(numTask_);
    for (size_t j = 0; j < numTask_; ++j) {
      for (size_t k = 0; k < dim_; ++k) {
	
      }
    }
  }
  
}

void MLearning::compGradients(vector<vector<double> > &weightMat, vector<vector<double> > &lapMatY, double lam, double lamR, vector<vector<double> > &gradMat) {
  gradMat.resize(numTask_);

  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < numTask_; ++i) {
    vector<double> &weightVec = weightMat[i];
    vector<double> &grad = gradMat[i];
    grad.resize(dim_);
    for (size_t j = 0; j < numDat_; ++j) {
      vector<pair<uint32_t, double> > &fv = fvs_[j];
      double sum = 0.0;
      for (size_t k = 0; k < fv.size(); ++k)
	sum += (weightVec[fv[k].first] * fv[k].second);

      double label = 1.0;
      if (labelMat_[j][i] != 1)
	label = -1.0;

      double wLabel = 1.0;
      if (label != 1.0)
	wLabel = wLabelMinus_[i];
      else
	wLabel = wLabelPlus_[i];
      
      double sumExp = exp(-label * sum);
      double coff = label/(1.0/sumExp + 1.0);
      for (size_t k = 0; k < fv.size(); ++k) 
	grad[fv[k].first] -= wLabel * coff * fv[k].second;
    }
  }

  for (size_t i = 0; i < numTask_; ++i) {
    for (size_t j = 0; j < dim_; ++j) 
      gradMat[i][j] /= ((double)(numDat_ * numTask_));
  }

  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < dim_; ++i) {
    for (size_t j = 0; j < numTask_; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < numTask_; ++k) 
	sum +=  weightMat[k][i] * lapMatY[k][j];
      gradMat[j][i] += lambda1_ * sum;
    }
  }

  #ifdef _PARALLEL_
    #pragma omp parallel for schedule(static)
  #endif
  for (size_t i = 0; i < numTask_; ++i) {
    vector<double> &grad = gradMat[i];
    for (size_t j = 0; j < dim_; ++j)
      grad[j] += lambda2_ * weightMat[i][j];
  }
}

void MLearning::save(ostream &os) {
  {
    size_t size = fecNames_.size();
    os.write((const char*)(&size), sizeof(size));
    for (size_t i = 0; i < size; ++i) {
      string &name = fecNames_[i];
      size_t s = name.size();
      os.write((const char*)(&s), sizeof(s));
      os.write((const char*)(&name[0]), sizeof(char) * s);
    }
  }
  {
    size_t size = datNames_.size();
    os.write((const char*)(&size), sizeof(size));
    for (size_t i = 0; i < size; ++i) {
      string &name1 = datNames_[i].first;
      string &name2 = datNames_[i].second;
      size_t s = name1.size();
      os.write((const char*)(&s), sizeof(s));
      os.write((const char*)(&name1[0]), sizeof(char) * s);
      s = name2.size();
      os.write((const char*)(&s), sizeof(s));
      os.write((const char*)(&name2[0]), sizeof(char) * s);
    }
  }
  {
    size_t size = taskNames_.size();
    os.write((const char*)(&size), sizeof(size));
    for (size_t i = 0; i < size; ++i) {
      string &name = taskNames_[i];
      size_t s = name.size();
      os.write((const char*)(&s), sizeof(s));
      os.write((const char*)(&name[0]), sizeof(char) * s);
    }
  }
  {
    size_t size = weightMat_.size();
    os.write((const char*)(&size), sizeof(size));
    for (size_t i = 0; i < size; ++i) {
      vector<double> &weightVec = weightMat_[i];
      size_t s = weightVec.size();
      os.write((const char*)(&s), sizeof(s));
      os.write((const char*)(&weightVec[0]), sizeof(double) * s);
    }
  }
  {
    os.write((const char*)(&numTask_), sizeof(numTask_));
    os.write((const char*)(&numDat_), sizeof(numDat_));
    os.write((const char*)(&dim_), sizeof(dim_));
  }
}

void MLearning::load(istream &is) {
  {
    size_t size;
    is.read((char*)(&size), sizeof(size));
    fecNames_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      string &name = fecNames_[i];
      size_t s;
      is.read((char*)(&s), sizeof(s));
      name.resize(s);
      is.read((char*)(&name[0]), sizeof(char) * s);
    }
  }
  {
    size_t size;
    is.read((char*)(&size), sizeof(size));
    datNames_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      string &name1 = datNames_[i].first;
      string &name2 = datNames_[i].second;
      size_t s;
      is.read((char*)(&s), sizeof(s));
      name1.resize(s);
      is.read((char*)(&name1[0]), sizeof(char) * s);
      is.read((char*)(&s), sizeof(s));
      name2.resize(s);
      is.read((char*)(&name2[0]), sizeof(char) * s);
    }
  }
  {
    size_t size;
    is.read((char*)(&size), sizeof(size));
    taskNames_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      string &name = taskNames_[i];
      size_t s;
      is.read((char*)(&s), sizeof(s));
      name.resize(s);
      is.read((char*)(&name[0]), sizeof(char) * s);
    }
  }
  {
    size_t size;
    is.read((char*)(&size), sizeof(size));
    weightMat_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      vector<double> &weightVec = weightMat_[i];
      size_t s;
      is.read((char*)(&s), sizeof(s));
      weightVec.resize(s);
      is.read((char*)(&weightVec[0]), sizeof(double) * s);
    }
  }
  {
    is.read((char*)(&numTask_), sizeof(numTask_));
    is.read((char*)(&numDat_), sizeof(numDat_));
    is.read((char*)(&dim_), sizeof(dim_));
  }
}

void MLearning::printFeatures(ostream &os) {
  if (numTask_ != taskNames_.size()) {
    cerr << "error 0 : " << numTask_ << " " << taskNames_.size() << endl;
    exit(1);
  }
  /*
  if (dim_ != fecNames_.size()) {
    cerr << "error 1 : " << dim_ << " "<< fecNames_.size() << endl;
    exit(1);
  }
  */
  if (numTask_ != weightMat_.size()) {
    cerr << "error 2 : " << numTask_ << " " << weightMat_.size() << endl;
    exit(1);
  }
  for (size_t i = 0; i < numTask_; ++i) {
    if (weightMat_[i].size() != dim_) {
      cerr << "error 3 : " << dim_ << " " << weightMat_[i].size() << endl;
      exit(1);
    }
  }

  for (size_t i = 0; i < taskNames_.size(); ++i)
    os << taskNames_[i] << "\t";
  os << endl;

  for (size_t i = 0; i < dim_; ++i) {
    os << fecNames_[i];
    for (size_t j = 0; j < numTask_; ++j) 
      os << "\t" << weightMat_[j][i];
    os << endl;
  }
}

void MLearning::train(double lambda1, double lambda2, uint64_t num_threads) {
  lambda1_ = lambda1;
  lambda2_ = lambda2;

#ifdef _PARALLEL_
   omp_set_num_threads(num_threads);
#endif

  cerr << "numTask : " << numTask_ << endl;
  cerr << "numDat : " << numDat_ << endl;
  cerr << "dim : " << dim_ << endl;
  cerr << "lambda1 : " << lambda1 << endl;
  cerr << "lambda2 : " << lambda2 << endl;

  wLabelPlus_.resize(numTask_);
  wLabelMinus_.resize(numTask_);
  for (size_t i = 0; i < labelMat_[0].size(); ++i) {
    uint64_t cPlus = 0, cMinus = 0;
    for (size_t j = 0; j < labelMat_.size(); ++j) {
      if (labelMat_[j][i] != 1) 
	cMinus++;
      else
	cPlus++;
    }
    //    wLabelPlus_[i]  = 1.0/(double)cPlus;
    //    wLabelMinus_[i] = 1.0/(double)cMinus;
    wLabelPlus_[i]  = double(cPlus + cMinus)/(double)cPlus;
    wLabelMinus_[i] = double(cPlus + cMinus)/(double)cMinus;
  }
  
  uint64_t maxitr = 1000000;
  //  double eta = 1e-5;
  //  double eta = 0.1;
  double eta = 0.0001;
  double prevObj = DBL_MAX;
  vector<vector<double> > lapMatY;
  
  convertSimToLaplacian(simMatY_, lapMatY); // numTask_*

  //  vector<vector<double> > lapMatX;
  //  convertSimToLaplacian(simMatX_, lapMatX);

  weightMat_.resize(numTask_);
  for (size_t i = 0; i < numTask_; ++i) 
    weightMat_[i].resize(dim_);

  LBFGS lbfgs;
  int converge = 0;
  
  for (size_t itr = 0; itr < maxitr; ++itr) {
    double obj = compObjectiveFun(weightMat_);

    obj += lambda1_ * compTrace1(weightMat_, lapMatY);
    obj += lambda2_ * compTrace2(weightMat_);

    double diff = (itr == 0 ? 1.0 : fabs(prevObj - obj)/prevObj);
    //    double diff = std::abs(prevObj - obj);
    cerr << "iteration : " << itr << " obj : " << obj << " " << " prevObj : " << prevObj << " diff : " << (double)diff << " eta : " << eta << endl;
    prevObj = obj;
    if (diff < eta)
      converge++;
    else
      converge = 0;

    if (itr > maxitr || converge == 3)
      break;
    
    vector<vector<double> > gradMat;
    compGradients(weightMat_, lapMatY, lambda1, lambda2, gradMat);

    vector<double> grad;
    for (size_t i = 0; i < numTask_; ++i) {
      for (size_t j = 0; j < dim_; ++j) 
	grad.push_back(gradMat[i][j]);
    }
    
    vector<double> weights;
    for (size_t i = 0; i < numTask_; ++i) {
      for (size_t j = 0; j < dim_; ++j) 
	weights.push_back(weightMat_[i][j]);
    }
    
    if(lbfgs.optimize(grad.size(), &weights[0], obj, &grad[0], false, 1.0) <= 0) {
      cerr << "faiture in optimization" << endl;
      srand((unsigned int)time(0));
      for (size_t i = 0; i < weights.size(); ++i)
	weights[i] = rand();
    }

    uint64_t pt = 0;
    for (size_t i = 0; i < numTask_; ++i) {
      for (size_t j = 0; j < dim_; ++j) 
	weightMat_[i][j] = weights[pt++];
    }
  }
}

void MLearning::predict(ostream &os) {
  for (size_t i = 0; i < taskNames_.size(); ++i) {
    os << taskNames_[i];
    os << "\t";
  }
  os << endl;

  for (size_t i = 0; i < fvs_.size(); ++i) {
    os << datNames_[i].first << "_" << datNames_[i].second;
    
    vector<pair<uint32_t, double> > &fv = fvs_[i];
    for (size_t j = 0; j < weightMat_.size(); ++j) {
      vector<double> &weightVec = weightMat_[j];
      double sum = 0.0;
      for (size_t k = 0; k < fv.size(); ++k) {
	if (weightVec.size() > fv[k].first)
	  sum += weightVec[fv[k].first] * fv[k].second;
      }
      os << "\t" << sum;
    }
    os << endl;
  }
}
