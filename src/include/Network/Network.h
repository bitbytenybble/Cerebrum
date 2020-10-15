#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include <Layer.h>
#include <Cost.h>
#include <memory>
#include <LayerType.h>
#include <functional>
// Forward declare class due to cyclical dependency of friend member in Layer
class Layer;


class Network {
    private:

        // Network hyperparameters
        uint32_t numLayers;
        uint32_t numFeatures; // Features into the input layer

        // vector of LayerPairs
        std::vector<std::unique_ptr<Layer>> layerVec;
        std::vector<double> costs;

        CostFunction costFunction;

        void createLayer(uint32_t previousNodes, LayerType type, uint32_t numNodes,
                         std::function<void(Eigen::MatrixXd &, float, bool)> func, float alpha);

        void updateAll(const float learningRate);


    public:

        Network(uint32_t numFeatures,
            std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &, bool)> function);

        void appendLayer(LayerType type, uint32_t numNodes,
                         std::function<void(Eigen::MatrixXd &, float, bool)>, float alpha = 1.0);

        Eigen::MatrixXd getCosts();

        Eigen::MatrixXd predict(const Eigen::MatrixXd &x);

        void train(const Eigen::MatrixXd &xin, const Eigen::MatrixXd &y_target,
                    uint32_t epochs, float learningRate);

        void updateCosts(Eigen::MatrixXd &prediction, Eigen::MatrixXd y_target);

        void backpropagate(const Eigen::MatrixXd &prediction, const Eigen::MatrixXd &y_target,
                           const float learningRate, const Eigen::MatrixXd &networkInput);

        void setSeed(uint32_t seed);


};



#endif
