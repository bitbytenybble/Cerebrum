#include "test_utils.h"
#include <iostream>
#include <fstream>

using namespace std;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


void writeToCSVfile(string name, Eigen::MatrixXd matrix) {
    ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
 }
