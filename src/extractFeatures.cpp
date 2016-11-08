#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdint.h>
#include <algorithm>

using namespace std;

class Cmp {
public:
  bool operator()(pair<uint32_t, double> &left, pair<uint32_t, double> &right) {
    return (left.second > right.second);
  }
};

int main(int argc, char **argv) {
  vector<string> konames;
  vector<string> pnames;

  ifstream ifs(argv[1]);
  string line;
  while (getline(ifs, line)) {
    stringstream ss(line);
    string name;
    while (ss >> name){
      konames.push_back(name);
    }
    break;
  }

  uint64_t id = 0;
  vector<vector<pair<uint32_t, double> > > invertedindex;
  while (getline(ifs, line)) {
    stringstream ss(line);
    uint32_t fid = 0;
    string name;
    double val;
    ss >> name;
    pnames.push_back(name);
    while (ss >> val) {
      if (invertedindex.size() <= fid)
	invertedindex.resize(fid + 1);
      invertedindex[fid].push_back(make_pair(id, val));
      fid++;
    }
    id++;
  }

  for (size_t i = 0; i < invertedindex.size(); ++i) {
    vector<pair<uint32_t, double> > &lists = invertedindex[i];
    sort(lists.begin(), lists.end(), Cmp());
  }

  for (size_t i = 0; i < invertedindex.size(); ++i) {
    vector<pair<uint32_t, double> > lists = invertedindex[i];
    cout << konames[i];
    for (size_t j = 0; j < 1000; ++j) {
      cout << "\t" << pnames[lists[j].first] << ":" << lists[j].second;
    }
    cout << endl;
  }
  
}
