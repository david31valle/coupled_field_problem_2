#include "mesh.hpp"
#include "mesh_1D.hpp"
#include "mesh_2D.hpp"
#include "mesh_3D.hpp"
#include <iostream>
#include <stdexcept>

std::tuple<Eigen::MatrixXd,  std::vector<Eigen::MatrixXd> >
generate_mesh(int /*PD*/,
              double domain_size,
              int partition,
              const std::vector<int>& element_orders,
              int problem_dimension)
{
    Eigen::MatrixXd nl, el1, el2;
    try {
        switch (problem_dimension) {
        case 1: {
                Mesh_1D m;
                auto res = m.generate(domain_size, partition, {element_orders[0]});
                printMesh1D(res.NL, res.EL[0]);
                nl  = res.NL;
                el1 = res.EL[0].cast<double>();
                el2 = res.EL[0].cast<double>();  // empty
                break;
        }
        case 2: {
                Mesh_2D m;
                std::vector<int> ord = { element_orders[0], element_orders[1] };
                auto res = m.generate(domain_size, partition, ord);
                printMesh2D(res.NL, res.EL[0]);
                nl  = res.NL;
                el1 = res.EL[0].cast<double>();
                el2 = res.EL[1].cast<double>();
                break;
        }
        case 3: {
                Mesh_3D m;
                std::vector<int> ord = { element_orders[0], element_orders[1], element_orders[2] };
                auto res = m.generate(domain_size, partition, ord);
                printMesh3D(res.NL, res.EL[0]);
                nl  = res.NL;
                el1 = res.EL[0].cast<double>();
                el2 = res.EL[1].cast<double>();
                break;
        }
        default:
            throw std::invalid_argument("problem_dimension must be 1,2, or 3");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Mesh generation error: " << e.what() << "\n";
    }
    std::vector<Eigen::MatrixXd> Element_lists;
    Element_lists.push_back(el1);
    Element_lists.push_back(el2);
    std::cout<<"testin"<<std::endl;
    std::cout<<el1<<std::endl;
    std::cout<<"Testing el2"<<std::endl;
    std::cout<<el2<<std::endl;


    return {nl, Element_lists};
}