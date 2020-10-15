#include <Cost.h>
#include <Eigen/Dense>



CostFunction::CostFunction(std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &, bool)> function): 
    function(function){}

double CostFunction::cost(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target){
    return function(y_out, y_target, false).mean();
}

Eigen::MatrixXd CostFunction::gradient(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target) {
    return function(y_out, y_target, true);
}


Eigen::MatrixXd Cost::mean_squared_error(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target, bool returnGradient) {
    Eigen::ArrayXXd error = y_out.array() - y_target.array();

    // Sum the squared error across each row, i.e. from left to right
    if (returnGradient) {
        // find the mean of the squared error
        return error / y_out.cols();
    }

    return 0.5 * error.pow(2);

}



Eigen::MatrixXd Cost::cross_entropy(const Eigen::MatrixXd &y_out, const Eigen::MatrixXd &y_target, bool returnGradient){
    const double EPSILON = std::pow(10,-15); // Minimum value in matrices, because log(0)==inf

    // cwiseMax replaces each coefficient with the maximum of the two coefficients being compared [y_out(i,j) vs EPSILON] 
    // cwiseMin replaces each coefficient with the minimum of the two coefficients being compared [y_out(i,j) vs EPSILON] 
    Eigen::ArrayXXd y_out_array = y_out.cwiseMax(EPSILON).cwiseMin(1);
    Eigen::ArrayXXd y_target_array = y_target; // Cast from matrix to array

    if (returnGradient){
        return (1 - y_target_array).cwiseQuotient(1 - y_out_array) \
                            -y_target_array.cwiseQuotient(y_out_array);
    }

    return -((y_target_array * y_out_array.log()) + ((1-y_target_array) * (1-y_out_array).log()));
}
