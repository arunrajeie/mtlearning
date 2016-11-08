/*
 * Train.cpp
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

#include <iostream>
#include <string>
#include "MLearning.hpp"
#include "cmdline.h"
#include <stdint.h>
using namespace std;

int main(int argc, char **argv) {
  cmdline::parser p;
  p.add<string>("featurevec_file", 'f', "feature vector file, ex: patch5.txt", true);
  p.add<string>("label_file", 'l', "label file, ex: sep14-6.txt", true);
  p.add<string>("sim_file", 's', "similarity file, ex: sep24-6.txt", true);
  p.add<string>("output_file", 'o', "output file name", true);
  p.add<double>("lambda1", 'a', "hyper parameter 1", false, 1);
  p.add<double>("lambda2", 'b', "hyper parameter 2", false, 1);
  p.add<uint64_t>("num_threads", 'n', "#threads", false, 1);
  p.parse_check(argc, argv);

  const string featurevec_file = p.get<string>("featurevec_file");
  const string label_file      = p.get<string>("label_file");
  const string sim_file        = p.get<string>("sim_file");
  const string output_file     = p.get<string>("output_file");
  double lambda1               = p.get<double>("lambda1");
  double lambda2               = p.get<double>("lambda2");
  uint64_t num_threads         = p.get<uint64_t>("num_threads");

  MLearning ml;
  {
    cerr << "readfile : " << featurevec_file << endl;
    ifstream ifs(featurevec_file.c_str()); //patch5.txt
    ml.readfile1(ifs);
    ifs.close();
  }
  {
    cerr << "readfile : " << label_file << endl;
    ifstream ifs(label_file.c_str()); //sep24-6.txt
    ml.readfile2(ifs);
    ifs.close();
  }
  {
    cerr << "readfile : " << sim_file << endl;
    ifstream ifs(sim_file.c_str()); //sep24-6.txt
    ml.readfile3(ifs);
    ifs.close();
  }
    
  cerr << "start optimization" << endl;
  ml.train(lambda1, lambda2, num_threads);
  
  ofstream ofs(output_file.c_str());
  if (!ofs) {
    cerr << "cannot open : " << output_file << endl;
    exit(1);
  }
  ml.save(ofs);
  ofs.close();
  
  return 0;
}
