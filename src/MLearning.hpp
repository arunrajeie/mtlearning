/*
 * MLearning.hpp
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

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <map>
#include <stdlib.h>

#include "Util.hpp"

#include "lbfgs.h"

#ifdef _PARALLEL_
#include <omp.h>
#endif 

class MLearning {
private:
  double sigmoid(double v);
  double compObjectiveFun(std::vector<std::vector<double> > &weightMat);
  void convertSimToLaplacian(std::vector<std::vector<double> > &simMat, std::vector<std::vector<double> > &matLap);
  double compTrace1(std::vector<std::vector<double> > &weightMat, std::vector<std::vector<double> > &lapMat);
  double compTrace2(std::vector<std::vector<double> > &weightMat);
  double compTrace3(std::vector<std::vector<double> > &weightMat, std::vector<std::vector<double> > &lapMat);
  void compGradients(std::vector<std::vector<double> > &weightMat, std::vector<std::vector<double> > &lapMatY, double lam, double lamR, std::vector<std::vector<double> > &gradMat);
public:
  void readfile1(std::ifstream &ifs);
  void readfile2(std::ifstream &ifs);
  void readfile3(std::ifstream &ifs);
  void readfile4(std::ifstream &ifs);
  void save(std::ostream &os);
  void load(std::istream &is);
  void printFeatures(std::ostream &os);
  void train(double lam, double lamR, uint64_t num_threads);
  void predict(std::ostream &os);
private:
  std::vector<std::vector<std::pair<uint32_t, double> > > fvs_;
  std::vector<std::vector<int> > labelMat_;
  std::vector<std::vector<double> > simMatY_;
  std::vector<std::vector<double> > simMatX_;

  std::vector<std::string> fecNames_;
  std::vector<std::pair<std::string, std::string> > datNames_;
  std::vector<std::string> taskNames_;

  std::vector<std::vector<double> > weightMat_;
  std::vector<double> wLabelPlus_;
  std::vector<double> wLabelMinus_;
  
  uint32_t numTask_;
  uint32_t numDat_;
  uint32_t dim_;
  double   lambda1_;
  double   lambda2_;
};
