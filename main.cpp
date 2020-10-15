#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <Cost.h>
#include <Layer.h>
#include <Network.h>
#include "test/test_utils.h"
#include <Activations.h>

using namespace std;

Eigen::MatrixXd genX(){
    Eigen::ArrayXXd array(1,100);
    for(int32_t i=0; i < 100; i++) array(0,i) = i/10.0;
    return array;
}



Eigen::MatrixXd genY(const Eigen::ArrayXXd & x){
    Eigen::ArrayXXd array(1,100);
    for(int32_t i=0; i < 100; i++) array(0,i) = x(0,i)*x(0,i);
    return array;
}





// --- TODOs to complete before ConvNets ---

// TODO work out virtual functionality


// 0) TODO Conditionally add -pthread flag to outter makefile

// 1) TODO CAN MULTI THREAD AT NETWORK.UPDATEALL

// 1.5) TODO CAN MULTI THREAD AT NETWORK.GETCOSTS

// 2) TODO GPU computations for matrix arithmetic

// 3) TODO implement normalization

// 4) TODO implement regulairzation and dropouts

// 5) TODO ADD MORE ACTIVATION FUNCTIONS

// 6) TODO ADD MORE COST FUNCTIONS

// 7) TODO Add testing capabilities by storing weights


// -----------------------------------------







void timeFunc(std::function<void()> func){

    auto start = std::chrono::high_resolution_clock::now();

	func();

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

    cout << "Time taken: " << microseconds << " microseconds" << endl;
}

void timedMain() {

    Eigen::MatrixXd x(2,2), y(2,2);
    double learningRate = 0.0007;
    uint32_t epochs = 1;

    x = genX();
    y = genY(x);

    Network n(1, Cost::mean_squared_error);
    n.setSeed(0); // Seed needs to be set before appending layers

    n.appendLayer(LayerType::Vanilla, 5, Activation::elu);
    n.appendLayer(LayerType::Vanilla, 5, Activation::elu);
    n.appendLayer(LayerType::Vanilla, 5, Activation::elu);
    n.appendLayer(LayerType::Vanilla, 1, Activation::elu);

    n.train(x, y, epochs, learningRate);

    writeToCSVfile("prediction.csv",n.predict(x));
    writeToCSVfile("target.csv", y);
    writeToCSVfile("input.csv", x);
    writeToCSVfile("costs.csv", n.getCosts().transpose());

}
int main() {
	//23 202 696
	timeFunc(timedMain);
	return 0;
}
