#include "Node.hpp"

Node::Node(int Nr, int PD, const Eigen::VectorXd& X_input,
           const Eigen::VectorXd& C, const Eigen::VectorXd& V,
           const Eigen::MatrixXd& EIL1, const Eigen::MatrixXd& EIL2,
           const Eigen::VectorXd& field_input, const Eigen::Vector2i& field_dim)
{
    this->Nr = Nr;
    this->PD = PD;

    X = X_input;
    x = X_input;

    int uvSize = C.size() + V.size();
    U = Eigen::VectorXd(uvSize);
    u = Eigen::VectorXd(uvSize);
    un = Eigen::VectorXd(uvSize);

    U << C, V;
    u << C, V;
    un << C, V;

    this -> EIL_1 = EIL1;
    this -> EIL_2 = EIL2;

    BC = Eigen::VectorXd(field_dim[0] + field_dim[1]);
    BC.head(field_dim[0]) = Eigen::VectorXd::Ones(field_dim[0]) * field_input[0];
    BC.tail(field_dim[1]) = Eigen::VectorXd::Ones(field_dim[1]) * field_input[1];

    DOF = Eigen::VectorXd::Zero(BC.size());

    field = field_input;

    int gp_size = PD * PD;
    GP_BC   = Eigen::VectorXd::Ones(gp_size) * field_input[0];
    GP_DOF  = Eigen::VectorXd::Zero(gp_size);
    GP_vals = Eigen::VectorXd::Zero(gp_size);

    BCset = false;
}


