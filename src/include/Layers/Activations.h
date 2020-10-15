#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <Eigen/Dense>



class ActivationFunction {
    private:
        //Allows custom activation functions to be used, if needed
        std::function<void(Eigen::MatrixXd &, float, bool)> function;
        float alpha;
    public:
        // Activation class functions
        ActivationFunction(std::function<void(Eigen::MatrixXd&, float, bool)> func, float alpha);
        void activate(Eigen::MatrixXd &z);
        void gradient(Eigen::MatrixXd &z);
};



namespace Activation {
    // Activation functions to pass to the Activation instance
    void elu(Eigen::MatrixXd &x, float alpha, bool returnGradient);
    void relu(Eigen::MatrixXd &x, float alpha, bool returnGradient);
    void sigmoid(Eigen::MatrixXd &x, float alpha, bool returnGradient);
    void tanh(Eigen::MatrixXd &x, float alpha, bool returnGradient);
}


#endif