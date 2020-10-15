#ifndef COST_H
#define COST_H
#include <Eigen/Dense>



// Nested-class object that provides the network with a cost
// function to minimize
class CostFunction {
    
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &, bool)> function;

    public:

        CostFunction(std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &, bool)> function);
        double cost(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target);    
        Eigen::MatrixXd gradient(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target);
};


namespace Cost {
    Eigen::MatrixXd mean_squared_error(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target, bool returnGradient);
    Eigen::MatrixXd cross_entropy(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target, bool returnGradient);
}




#endif