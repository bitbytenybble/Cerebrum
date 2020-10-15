#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H
#include <Layer.h>
#include <vector>
#include <functional>
class FullyConnected: public Layer {
    
    public:
        FullyConnected(uint32_t numNodes, uint32_t previousNodes,
                       std::function<void(Eigen::MatrixXd &, float, bool)> func, float alpha);

};


#endif