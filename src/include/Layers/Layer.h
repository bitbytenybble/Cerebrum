#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>
#include <Activations.h>
#include <Network.h>
#include <functional>
#define RETURN_GRADIENT true
#define RETURN_ACTIVATION false



class Layer {

    protected:

        friend class Network;

        // Layer Hyperparameters
        uint32_t numWeights;
        uint32_t numNodes;

        // Layer activation function
        ActivationFunction function;

        // Matrices //
        Eigen::MatrixXd weightMat; // Store weights
        Eigen::RowVectorXd biasMat;   // Store bias values

        // Stores gradients for weights and biases
        Eigen::MatrixXd weightGradient;
        Eigen::RowVectorXd biasGradient;

        // Stores computed sum
        Eigen::MatrixXd cache;


    public:
        // Constructor that takes in the number of nodes to be initialized
        // in the current layer.
        Layer(uint32_t nodes, uint32_t previousNodes,
             std::function<void(Eigen::MatrixXd &, float, bool)> func, float alpha);


        // Adds gradients to be cached in the layer and updated after backpropagation has completed
        // void addGradients(Eigen::MatrixXd weightGradient, Eigen::RowVectorXd biasGradient);
        void addGradients(Eigen::MatrixXd weightGradient, Eigen::MatrixXd biasGradient);

        // Update the layer's weights
        bool update(const float learningRate);

        // Virtual function to be overriden in the child class
        virtual void feedForward(Eigen::MatrixXd &x);

};







#endif
