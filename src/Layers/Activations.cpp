#include <Activations.h>
#include <Eigen/Dense>



ActivationFunction::ActivationFunction(std::function<void(Eigen::MatrixXd &, float, bool)> func, float alpha): 
                                      function(func), alpha(alpha){}


void ActivationFunction::activate(Eigen::MatrixXd &z) { 
    function(z, alpha, false); 
}
void ActivationFunction::gradient(Eigen::MatrixXd &z) { 
    function(z, alpha, true); 
}



// Functions wrapped in namespace to avoid 
namespace Activation{

    void elu(Eigen::MatrixXd &x, float alpha, bool returnGradient) {
        
        Eigen::MatrixXd greater = ((x.array() > 0).cast<double>()).matrix();
        Eigen::MatrixXd lessOrEq = ((x.array() <= 0).cast<double>()).matrix();

        Eigen::MatrixXd expPiece = alpha * (x.array().exp() - 1);
        Eigen::MatrixXd activation = greater.cwiseProduct(x) + lessOrEq.cwiseProduct(expPiece);
        

        //Return gradient here
        if (returnGradient) {
            x = greater + lessOrEq.cwiseProduct((activation.array() + alpha).matrix()) ;
        }
        // return activation here
        else x = activation;
    }


    void relu(Eigen::MatrixXd &x, float alpha, bool returnGradient) {

        Eigen::MatrixXd temp = ((x.array() > 0).cast<double>()).matrix();
        //Return gradient here
        if (returnGradient) x = temp; 
            
        // return activation here
        else x = temp.cwiseProduct(x);
        
    }












    void sigmoid(Eigen::MatrixXd &x, float alpha, bool returnGradient){
        Eigen::MatrixXd sigmoid = (1 + (-x).array().exp()).matrix().cwiseInverse();

        // f'(x) = S(x)*(1-S(x))
        if (returnGradient) x = sigmoid.cwiseProduct((1-sigmoid.array()).matrix());
        else x = sigmoid; // f(x) = S(x)
    }










    void tanh(Eigen::MatrixXd &x, float alpha, bool returnGradient){
        Eigen::ArrayXXd tanh = x.array().tanh();
        
        if (returnGradient) x = 1 - tanh.pow(2); // f'(x) = 1 - f(x)^2 = 1- tanh^2(x) 
        else x = tanh; // f(x) = tanh(x)
    }


}