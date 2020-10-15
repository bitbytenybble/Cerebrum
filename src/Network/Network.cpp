#include <Network.h>
#include <Activations.h>
#include <FullyConnected.h>
#include <iostream>
#include <Eigen/Dense>
#include <LayerType.h>
#include <vector>
#include <math.h>
#include <ctime>
#include <thread>


Network::Network(uint32_t numFeatures, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &, bool)> function):
    numLayers(0), numFeatures(numFeatures), costFunction(function){}



void Network::createLayer(uint32_t previousNodes, LayerType type, uint32_t numNodes,
                          std::function<void(Eigen::MatrixXd &, float, bool)> function, float alpha){

    Layer * layerPtr = nullptr;

    switch(type) {
        case LayerType::Vanilla:
            layerPtr = new FullyConnected(numNodes, previousNodes, function, alpha);
            //layerPtr->initWeights(previousNodes);
            break;

        case LayerType::Convolutional2D:
            break;

        default:
            std::cerr << "Error: Invalid type in Network.createLayer()\n" <<
             "Couldn't create model with layer of type: " << type << std::endl;
            exit(EXIT_FAILURE);
    }

    // Create unique_ptr<Layer> object to store layer and the derived class type to
    // cast it to in later processing
    layerVec.push_back(std::unique_ptr<Layer>(layerPtr));

}


void Network::appendLayer(LayerType type, uint32_t numNodes,
                          std::function<void(Eigen::MatrixXd &, float, bool)> function, float alpha){
    // Nodes from the previous layer, provide the input dimensions
    // to the current layer to be added.
    uint32_t previousNodes;

    if (numNodes < 1){
        std::cerr << "Error: Layer must have at least 1 node" << std::endl;
        exit(EXIT_FAILURE);
    }

    // if this layer is the first layer in the list,
    // the number of weights to each node is the number of input features
    if (numLayers == 0) previousNodes = numFeatures;
    //otherwise, use the number of nodes from the last layer added
    else previousNodes = layerVec.back()->numNodes;

    this->createLayer(previousNodes,type, numNodes, function, alpha);
    numLayers++;
}



Eigen::MatrixXd Network::predict(const Eigen::MatrixXd &x){

    Eigen::MatrixXd feedMat = x;
    std::vector<std::unique_ptr<Layer>>::iterator iter;

    try{
        for(iter = layerVec.begin(); iter < layerVec.end(); iter++){
            (*iter)->feedForward(feedMat);
        }
    }

    catch (const std::exception & error){
        std::cerr << "Error: Unhandled error Network.predict()\n" << error.what() << std::endl;
    }

    return feedMat;
}





void Network::train(const Eigen::MatrixXd & xin, const Eigen::MatrixXd & y_target,
                    uint32_t epochs, float learningRate) {

    Eigen::MatrixXd prediction;

    if (!numLayers) {
        std::cerr << "Error: No layers have been added.\nAdd layers to train" << std::endl;
        return;
    }

    if (epochs < 1){
        std::cerr << "Error: Network must have at least 1 training epoch" << std::endl;
        return;
    }

    // Checkpoint is approximately 5% of the iterations, or 1 epoch if 0.05*epochs < 1
    const uint32_t checkPoint = ceil(0.05 * epochs);

    // For each layer in the network get a prediction/output value
    // for the given training input, xin. Update the cost vector and
    // backpropagate the error and update the weights and biases
    for(uint32_t i = 0; i < epochs; i++){
        if (i % checkPoint == 0){
            std::cout << "\n" << ((float)i*100)/epochs << "% complete" << std::endl;
        }

        prediction = predict(xin); // get prediction given
        costs.push_back(costFunction.cost(prediction, y_target)); // Get costs and push into cost vector
        backpropagate(prediction, y_target, learningRate, xin);

    }

}


Eigen::MatrixXd Network::getCosts() {
    uint32_t size = costs.size();
    Eigen::MatrixXd costMat(size,1);

    for (uint32_t i = 0; i < size; i++) costMat(i,0) = costs[i];

    return costMat;
}





void Network::backpropagate(const Eigen::MatrixXd &prediction, const Eigen::MatrixXd &y_target,
                    const float learningRate, const Eigen::MatrixXd &networkInput){

    std::unique_ptr<Layer> & outterMost = layerVec[layerVec.size() - 1];
    Eigen::MatrixXd outterMostInput, gradientChain, cacheFromLayer, weightGradient;
    Eigen::MatrixXd biasGradient;

    //  z= wx+b, this is stored during forward propagation
    cacheFromLayer = outterMost->cache;

    if (layerVec.size() > 1){
        // Get second last layer and activate a copy of its cache
        std::unique_ptr<Layer> & secondOutterMost = layerVec[layerVec.size() - 2];
        outterMostInput = secondOutterMost->cache;
        secondOutterMost->function.activate(outterMostInput);
    }
    // if single layer network the "cache" value is the input to the network
    else outterMostInput = networkInput;

    // Activate cache from outtermost layer.
    // Note: Activation functions and their gradients always take z as an input: f(z) and f'(z)
    outterMost->function.gradient(cacheFromLayer);

    // Hadamard product | dC/dz[L] =  dC/da[L] * da[L]/dz[L] | where cacheFromLayer = da[L]/dz[L]
    gradientChain = costFunction.gradient(prediction, y_target).cwiseProduct(cacheFromLayer);

    // dC/dW = dC/da[L] * da[L]/dW
    weightGradient = gradientChain * outterMostInput.transpose();

    // dC/db = dC/da[L]
    biasGradient = gradientChain;

    // Store gradients in layer to update after backprop is complete
    outterMost->addGradients(weightGradient, biasGradient);

    // Propagate backwards through the remaining layers. I.e. towards the input layer
    for(int32_t i = numLayers - 2; i > -1; i--){

        // Note: Previous, current, and next refer to the order of the layers starting
        // from the output layer towards the input layer. Therefore, previous refers to
        // the layer one layer closer to the output layer than the current layer and vice verse.
        std::unique_ptr<Layer> & previousLayer = layerVec[i+1];
        std::unique_ptr<Layer> & currentLayer  = layerVec[i];


        // Feed forward input to current layer
        Eigen::MatrixXd ffInputToCurrentLayer;

        if (i) {
            std::unique_ptr<Layer> & nextLayer = layerVec[i-1];
            ffInputToCurrentLayer = nextLayer->cache;
            nextLayer->function.activate(ffInputToCurrentLayer);
        }
        else ffInputToCurrentLayer = networkInput;


        // Gradient of the activation function: da/dz
        Eigen::MatrixXd activationGradient = currentLayer->cache;
        currentLayer->function.gradient(activationGradient);


        // Weight matrix from the previous layer
        Eigen::MatrixXd previousWeights = previousLayer->weightMat;

        // ((W[L+1]).T * dC/dz[L+1]) elementwise multiplication with activationGradient
        gradientChain = (previousWeights.transpose() * gradientChain).array() * \
                        activationGradient.array();


        weightGradient = gradientChain * ffInputToCurrentLayer.transpose();
        biasGradient = gradientChain;


        currentLayer->addGradients(weightGradient, biasGradient);

    }

   updateAll(learningRate);
}







void Network::setSeed(uint32_t seed){
    std::srand(seed);
}





// Update all layers with their cached gradient matrices
// which were calculated during backprop
#if defined(LINUX)




void Network::updateAll(const float learningRate){

	for(uint32_t i=0; i < numLayers; i++) layerVec[i]->update(learningRate);
}


#elif defined(WINDOWS_NT)
void Network::updateAll(const float learningRate){
    for(uint32_t i=0; i < numLayers; i++) layerVec[i]->update(learningRate);
}


#endif
