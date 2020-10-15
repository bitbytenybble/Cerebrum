#include <Layer.h>
#include <Eigen/Dense>
#include <iostream>
#include <stdlib.h>

// TODO each function needs more descriptive error messages


// TODO change weight initialization to GLOROT/XAVIER



Layer::Layer(uint32_t nodes, uint32_t previousNodes, 
             std::function<void(Eigen::MatrixXd &, float, bool)> func, float alpha):
            numWeights(previousNodes), numNodes(nodes), function(func, alpha), 
            weightMat(Eigen::MatrixXd::Random(this->numNodes, this->numWeights) * 0.1),
            biasMat(Eigen::RowVectorXd::Zero(numNodes)){}




// Cache the computed gradients for weights and biases
void Layer::addGradients(Eigen::MatrixXd weightGradient, Eigen::MatrixXd biasGradient) {
    
    this->weightGradient = weightGradient;
    
    // biasGradient is passed in as matrix but needs to be transformed
    // into a column vector. Eigen stores the biasMat as a row vector
    // so it needs to be transposed and reduced from matrix to vector form
    this->biasGradient = biasGradient.transpose().rowwise().mean();

    
}








// Update weights after computing all the matrix gradients
// in the network. Updating layers DURING backprop propagates
// the incorrect error/gradient through the rest of the network.
bool Layer::update(const float learningRate){
    try {
        // W = W - (learningRate * dw/dC) where dw/dc is the partial derivitive of the current weights
        // with respect to the networks cost function.
        weightMat -= learningRate * weightGradient;        

        // biasGradient is not a vector, so it needs to be summed across each row to create
        // a (numNodes,1) sized row vector to be subtracted from.
        biasMat = biasMat.array() - learningRate * (biasGradient.sum());

    }
    
    catch (const std::exception & error){
        std::cout << "Unhandled error in Layer.update()\n" << error.what() << std::endl;
        return false;
    }
    
    return true;
}









void Layer::feedForward(Eigen::MatrixXd &x){

    try {
        // Assign the reference to the input, x, equal to Wx + b
        x = (weightMat*x).colwise() + biasMat.transpose();
        
        // Set cache to the output of Wx + b, to be used in backpropagation
        cache = x;
    
        // "Activates" the values in the matrix 'x'
        this->function.activate(x);
    }
    catch (const std::exception & error){
        std::cout << "Unhandled error in Layer.feedForward()\n" << error.what() << std::endl;
        std::cout << "Exiting program" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    

}















