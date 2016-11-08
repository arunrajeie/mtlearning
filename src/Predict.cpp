/*
 * Predict.cpp
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

using namespace std;

int main(int argc, char **argv) {
  cmdline::parser p;

  p.add<string>("featurevec_file", 'f', "feature vector file, ex: patch5.txt", true);
  p.add<string>("model_file", 'm', "model file", true);
  p.add<string>("output_file", 'o', "output file", true);

  p.parse_check(argc, argv);

  const string featurevec_file = p.get<string>("featurevec_file");
  const string model_file      = p.get<string>("model_file");
  const string output_file     = p.get<string>("output_file");

  MLearning ml;

  {
    cerr << "readfile : " << model_file << endl;
    ifstream ifs(model_file.c_str());
    ml.load(ifs);
  }
  {
    cerr << "readfile : " << featurevec_file << endl;
    ifstream ifs(featurevec_file.c_str()); //patch5.txt
    ml.readfile1(ifs);
    ifs.close();
  }
  {
    ofstream ofs(output_file.c_str());
    ml.predict(ofs);
    ofs.close();
  }
}
