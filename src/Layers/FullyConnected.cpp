#include <FullyConnected.h>
#include <Activations.h>

                                                                                
FullyConnected::FullyConnected(uint32_t numNodes, uint32_t previousNodes,
                std::function<void(Eigen::MatrixXd &, float, bool)> func, float alpha):
                Layer(numNodes, previousNodes, func, alpha) {}

