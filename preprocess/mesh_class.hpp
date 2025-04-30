

#ifndef COUPLED_FIELD_PROBLEM_2_MESH_CLASS_HPP
#define COUPLED_FIELD_PROBLEM_2_MESH_CLASS_HPP
#include "../Eigen/Dense"
#include <tuple>


//To do: this is just the constructor and what i need the class to have the element_list the fields for the fields,
//it is a tuple so we can expand it without the need to change the others functions definitions. change the name of the class to just mesh,
//we dont need the other mesh.cpp then.

class mesh_class {
public:
    mesh_class(int problem_domain,
               int domain_size,
               int partition,
               Eigen::VectorXd element_order,
               std::string plot_mesh
               );

    //This is supposed to be EIL1 and EIL2
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> EIL_tuple;
};


#endif //COUPLED_FIELD_PROBLEM_2_MESH_CLASS_HPP
