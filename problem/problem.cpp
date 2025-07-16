
#include "problem.hpp"


#include "problem.hpp"
#include <fstream>
#include <iomanip>

#include <set>


problem_coupled::problem_coupled(
        int PD,
        std::vector<Node>& NL,
        std::vector<element>& EL,
        int domain_size,
        const std::string& BC_type,
        const std::string Corners,
        const std::string& Initial_density,
        const std::vector<double>& parameters,
        const std::vector<int>& element_order,
        const Eigen::Vector2i& field_dim,
        const std::string& GP_vals,
        const std::string& time_incr_method,
        double T,
        double dt,
        double time_factor,
        int max_iter,
        double tol
) :
        PD(PD), Node_List(NL), Element_List(EL), domain_size(domain_size),
        BC_type(BC_type), Initial_density(Initial_density), parameters(parameters),
        element_order(element_order), field_dim(field_dim),
        GP_vals(GP_vals), time_incr_method(time_incr_method),
        T(T), dt(dt), time_factor(time_factor), max_iter(max_iter), tol(tol)
{
    // === Construct output filename ===
    std::ostringstream oss;
    oss << PD << "D_Normal_" << EL.size()
        << "_EL=[";
    for (int eo : element_order) oss << eo;
    oss << "]_" << Initial_density << "_" << time_incr_method << "_" << BC_type << "_[";
    for (size_t i = 0; i < parameters.size(); ++i) {
        oss << parameters[i];
        if (i != parameters.size() - 1) oss << ",";
    }
    oss << "].txt";

    filename = oss.str();

    // === Assign boundary conditions ===
    Assign_BC(Corners);  // fills BC and DOFs
    Assign_DOF_DBC();    // computes global DOF numbering

    // === Assign Gauss Point DOFs if requested ===
    if (GP_vals == "On") {
        Assign_GP_DOFs();
    }

    // === Output problem setup info ===
    problem_info();  // displays PD, mesh, DOFs, etc.
}



void problem_coupled::Assign_BC(const std::string Corners) {
    int NoNs = Node_List.size();
    double tol = 1e-6;

    double W = static_cast<double>(domain_size);
    double H = W;
    double D = W;

    if (BC_type == "DBC") {
        // (DBC branch from before...)
        Assign_DOF_DBC();
        return;
    }

    // === PBC ===
    std::vector<int> NLC;  // Corner nodes
    std::vector<int> NLS;  // Star nodes
    std::vector<int> NLM;  // Minus edge/face
    std::vector<int> NLP;  // Plus edge/face

    if (PD == 1) {
        for (int i = 0; i < NoNs; ++i) {
            auto& node = Node_List[i];
            double x = node.X(0);

            if (std::abs(x - 0.0) < tol || std::abs(x - W) < tol)
                NLC.push_back(i);
            else
                NLS.push_back(i);
        }
    }

    else if (PD == 2) {
        std::vector<int> MEX, PEX, MEY, PEY;

        for (int i = 0; i < NoNs; ++i) {
            const auto& X = Node_List[i].X;
            double x = X(0), y = X(1);

            bool onLeft = std::abs(x - 0.0) < tol;
            bool onRight = std::abs(x - W) < tol;
            bool onBottom = std::abs(y - 0.0) < tol;
            bool onTop = std::abs(y - H) < tol;

            bool isCorner =
                    (onLeft && onBottom) || (onRight && onBottom) ||
                    (onRight && onTop) || (onLeft && onTop);

            if (isCorner) {
                NLC.push_back(i);
            } else if (onLeft) {
                MEY.push_back(i);
            } else if (onRight) {
                PEY.push_back(i);
            } else if (onBottom) {
                MEX.push_back(i);
            } else if (onTop) {
                PEX.push_back(i);
            } else {
                NLS.push_back(i);
            }
        }

        // Match edges to construct NLM/NLP
        for (int pi : PEX) {
            for (int mi : MEX) {
                if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }

        for (int pi : PEY) {
            for (int mi : MEY) {
                if (std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }
    }

    else if (PD == 3) {
        std::vector<int> MEX, PEX, MEY, PEY, MEZ, PEZ;
        std::vector<int> MFX, PFX, MFY, PFY, MFZ, PFZ;

        for (int i = 0; i < NoNs; ++i) {
            const auto& X = Node_List[i].X;
            double x = X(0), y = X(1), z = X(2);

            bool onX0 = std::abs(x - 0.0) < tol;
            bool onXW = std::abs(x - W) < tol;
            bool onY0 = std::abs(y - 0.0) < tol;
            bool onYH = std::abs(y - H) < tol;
            bool onZ0 = std::abs(z - 0.0) < tol;
            bool onZD = std::abs(z - D) < tol;

            if ((onX0 && onY0 && onZ0) || (onX0 && onY0 && onZD) ||
                (onX0 && onYH && onZ0) || (onX0 && onYH && onZD) ||
                (onXW && onY0 && onZ0) || (onXW && onY0 && onZD) ||
                (onXW && onYH && onZ0) || (onXW && onYH && onZD)) {
                NLC.push_back(i);
            } else if (onX0 && onY0) {
                MEZ.push_back(i);
            } else if (onX0 && onYH) {
                PEZ.push_back(i);
            } else if (onX0 && onZ0) {
                MEY.push_back(i);
            } else if (onX0 && onZD) {
                PEY.push_back(i);
            } else if (onY0 && onXW) {
                PEZ.push_back(i);
            } else if (onY0 && onZ0) {
                MEX.push_back(i);
            } else if (onY0 && onZD) {
                PEX.push_back(i);
            } else if (onX0) {
                MFX.push_back(i);
            } else if (onXW) {
                PFX.push_back(i);
            } else if (onY0) {
                MFY.push_back(i);
            } else if (onYH) {
                PFY.push_back(i);
            } else if (onZ0) {
                MFZ.push_back(i);
            } else if (onZD) {
                PFZ.push_back(i);
            } else {
                NLS.push_back(i);
            }
        }

        // Match each pair of periodic faces
        auto match_nodes = [&](const std::vector<int>& plus, const std::vector<int>& minus, int comp) {
            for (int pi : plus) {
                for (int mi : minus) {
                    if (std::abs(Node_List[pi].X(comp) - Node_List[mi].X(comp)) < tol) {
                        NLP.push_back(pi);
                        NLM.push_back(mi);
                    }
                }
            }
        };

        match_nodes(PEX, MEX, 0);
        match_nodes(PEY, MEY, 1);
        match_nodes(PEZ, MEZ, 2);

        for (int pi : PFX) {
            for (int mi : MFX) {
                if (std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol &&
                    std::abs(Node_List[pi].X(2) - Node_List[mi].X(2)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }

        for (int pi : PFY) {
            for (int mi : MFY) {
                if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol &&
                    std::abs(Node_List[pi].X(2) - Node_List[mi].X(2)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }

        for (int pi : PFZ) {
            for (int mi : MFZ) {
                if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol &&
                    std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol) {
                    NLP.push_back(pi);
                    NLM.push_back(mi);
                }
            }
        }
    }

    // === Assign values for corner and periodic nodes ===

    for (int i : NLC) {
        Node_List[i].U = Node_List[NLC[0]].U;
        Node_List[i].u = Node_List[NLC[0]].u;
        Node_List[i].un = Node_List[NLC[0]].un;
    }

    for (size_t i = 0; i < NLP.size(); ++i) {
        Node_List[NLP[i]].U = Node_List[NLM[i]].U;
        Node_List[NLP[i]].u = Node_List[NLM[i]].u;
        Node_List[NLP[i]].un = Node_List[NLM[i]].un;
    }

    // === Corner constraint to avoid rigid body motion ===
    if (Corners == "Fixed") {
        for (int i : NLC) {
            Node_List[i].U.segment(1, PD).setZero();
            Node_List[i].u.segment(1, PD).setZero();
            Node_List[i].un.segment(1, PD).setZero();
            Node_List[i].BC.segment(1, PD).setZero();
        }
    }

    // === Assign DOFs for PBC ===
    Assign_DOF_PBC(NLC, NLS, NLM, NLP);
}

void problem_coupled::Assign_DOF_DBC() {
    int dofs = 0;
    int NoNs = Node_List.size();

    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC = Node_List[i].BC;
        Eigen::VectorXd& DOF = Node_List[i].DOF;

        if (DOF.size() != BC.size()) {
            DOF = Eigen::VectorXd::Zero(BC.size());
        }

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                ++dofs;
                DOF(p) = dofs;
            }
        }

        Node_List[i].DOF = DOF;
    }

    DOFs = dofs;  // store the global number of DOFs
}


void problem_coupled::Assign_DOF_PBC(const std::vector<int>& NLC,
                                     const std::vector<int>& NLS,
                                     const std::vector<int>& NLM,
                                     const std::vector<int>& NLP) {
    int dofs = 0;

    int NoCNs = static_cast<int>(NLC.size());
    int NoSNs = static_cast<int>(NLS.size());
    int NoMNs = static_cast<int>(NLM.size());
    int NoPNs = static_cast<int>(NLP.size());

    // === Corner nodes ===
    if (NoCNs > 0) {
        for (int i = 0; i < NoCNs; ++i) {
            int idx = NLC[i];
            Eigen::VectorXd& BC = Node_List[idx].BC;
            Eigen::VectorXd& DOF = Node_List[idx].DOF;

            if (i == 0) {
                for (int p = 0; p < BC.size(); ++p) {
                    if (BC(p) == 1) {
                        ++dofs;
                        DOF(p) = dofs;
                    }
                }
                Node_List[idx].DOF = DOF;
            }

            // Copy reference DOFs to all corners
            Node_List[idx].DOF = Node_List[NLC[0]].DOF;
        }
    }

    // === Star nodes ===
    for (int idx : NLS) {
        Eigen::VectorXd& BC = Node_List[idx].BC;
        Eigen::VectorXd& DOF = Node_List[idx].DOF;

        if (DOF.size() != BC.size())
            DOF = Eigen::VectorXd::Zero(BC.size());

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                ++dofs;
                DOF(p) = dofs;
            }
        }

        Node_List[idx].DOF = DOF;
    }

    // === Minus nodes ===
    for (int idx : NLM) {
        Eigen::VectorXd& BC = Node_List[idx].BC;
        Eigen::VectorXd& DOF = Node_List[idx].DOF;

        if (DOF.size() != BC.size())
            DOF = Eigen::VectorXd::Zero(BC.size());

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                ++dofs;
                DOF(p) = dofs;
            }
        }

        Node_List[idx].DOF = DOF;
    }

    // === Plus nodes ===
    for (size_t i = 0; i < NLP.size(); ++i) {
        Node_List[NLP[i]].BC  = Node_List[NLM[i]].BC;
        Node_List[NLP[i]].DOF = Node_List[NLM[i]].DOF;
    }

    // Save total count
    DOFs = dofs;
}

void problem_coupled::Assign_GP_DOFs() {
    int GP_dofs = 0;
    int NoNs = Node_List.size();

    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC = Node_List[i].GP_BC;
        Eigen::VectorXd& DOF = Node_List[i].GP_DOF;

        // If any entry in GP_BC is 1, assign a new DOF index to all GP_DOF entries
        if ((BC.array() == 1).any()) {
            ++GP_dofs;
            DOF = Eigen::VectorXd::Constant(BC.size(), GP_dofs);
            Node_List[i].GP_DOF = DOF;
        }
    }

    GP_DOFs = GP_dofs;  // store total number of Gauss point DOFs
}



void problem_coupled::update(const Eigen::VectorXd& dx) {
    int NoNs = Node_List.size();
    int NoEs = Element_List.size();
    int PD   = PD;

    // === Update node unknowns u based on DOFs and dx ===
    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC  = Node_List[i].BC;
        Eigen::VectorXd& DOF = Node_List[i].DOF;
        Eigen::VectorXd& u   = Node_List[i].u;

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                int dofIdx = static_cast<int>(DOF(p)) - 1;  // convert to 0-based index
                if (dofIdx >= 0 && dofIdx < dx.size()) {
                    u(p) += dx(dofIdx);
                }
            }
        }

        Node_List[i].u = u;
    }

    // === Update element field values from updated nodal u ===
    for (int e = 0; e < NoEs; ++e) {
        auto& elem = Element_List[e];

        int NPE1 = elem.NPE1;
        int NPE2 = elem.NPE2;

        // === Scalar field c (dimension 1) ===
        Eigen::MatrixXd c(1, NPE1);
        for (int i = 0; i < NPE1; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL1(i)) - 1;  // 1-based to 0-based
            c(0, i) = Node_List[nodeIdx].u(0);  // scalar field is first entry
        }
        elem.c = c;

        // === Vector field v (dimension PD) ===
        Eigen::MatrixXd v(PD, NPE2);
        for (int i = 0; i < NPE2; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL2(i)) - 1;
            v.col(i) = Node_List[nodeIdx].u.segment(1, PD);  // vector field is offset by 1
        }
        elem.v = v;
    }
}

void problem_coupled::update_GP(const Eigen::VectorXd& dx_gp) {
    int NoNs = Node_List.size();

    for (int i = 0; i < NoNs; ++i) {
        const Eigen::VectorXd& BC  = Node_List[i].GP_BC;
        const Eigen::VectorXd& DOF = Node_List[i].GP_DOF;

        if ((BC.array() == 1).any()) {
            int dof_idx = static_cast<int>(DOF(0)) - 1;  // 0-based index
            if (dof_idx >= 0 && dof_idx < dx_gp.size()) {
                Node_List[i].GP_vals = Eigen::VectorXd::Constant(BC.size(), dx_gp(dof_idx));
            }
        }
    }
}

void problem_coupled::update_time() {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    for (int e = 0; e < NoEs; ++e) {
        Element_List[e].cn = Element_List[e].c;
        Element_List[e].vn = Element_List[e].v;
    }

    for (int i = 0; i < NoNs; ++i) {
        Node_List[i].un = Node_List[i].u;
    }
}


void problem_coupled::downdate_time() {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    for (int e = 0; e < NoEs; ++e) {
        Element_List[e].c = Element_List[e].cn;
        Element_List[e].v = Element_List[e].vn;
    }

    for (int i = 0; i < NoNs; ++i) {
        Node_List[i].u = Node_List[i].un;
    }
}


Eigen::VectorXd problem_coupled::Residual(double dt) {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    // Assuming first element is representative for all
    int NPE1 = Element_List[0].NPE1;
    int NPE2 = Element_List[0].NPE2;

    Eigen::VectorXd Rtot = Eigen::VectorXd::Zero(DOFs);

    for (int e = 0; e < NoEs; ++e) {
        auto [R1, R2] = Element_List[e].Residual(dt);

        const Eigen::VectorXd& NdL1 = Element_List[e].NdL1;
        const Eigen::VectorXd& NdL2 = Element_List[e].NdL2;

        // === R1: Scalar field residual ===
        int dim = 1;
        int first = 0;
        int last  = dim;

        for (int i = 0; i < NPE1; ++i) {
            int nodeIdx = static_cast<int>(NdL1(i)) - 1;
            const Eigen::VectorXd& BC  = Node_List[nodeIdx].BC;
            const Eigen::VectorXd& DOF = Node_List[nodeIdx].DOF;

            for (int p = first; p < last; ++p) {
                if (BC(p) == 1) {
                    int dof_idx = static_cast<int>(DOF(p)) - 1;
                    if (dof_idx >= 0 && dof_idx < Rtot.size()) {
                        int r_idx = i * dim + (p - first);
                        Rtot(dof_idx) += R1(r_idx);
                    }
                }
            }
        }

        // === R2: Vector field residual ===
        dim = PD;
        first = last;
        last = first + dim;

        for (int i = 0; i < NPE2; ++i) {
            int nodeIdx = static_cast<int>(NdL2(i)) - 1;
            const Eigen::VectorXd& BC  = Node_List[nodeIdx].BC;
            const Eigen::VectorXd& DOF = Node_List[nodeIdx].DOF;

            for (int p = first; p < last; ++p) {
                if (BC(p) == 1) {
                    int dof_idx = static_cast<int>(DOF(p)) - 1;
                    if (dof_idx >= 0 && dof_idx < Rtot.size()) {
                        int r_idx = i * dim + (p - first);
                        Rtot(dof_idx) += R2(r_idx);
                    }
                }
            }
        }
    }

    return Rtot;
}


void problem_coupled::assemble(double dt) {
    int NoEs = Element_List.size();
    int NPE1 = Element_List[0].NPE1;
    int NPE2 = Element_List[0].NPE2;

    std::vector<Eigen::Triplet<double>> triplets;
    Rtot = Eigen::VectorXd::Zero(DOFs);

    for (int e = 0; e < NoEs; ++e) {
        auto [R1, R2, K11, K12, K21, K22] = Element_List[e].RK(dt);
        const Eigen::VectorXd& NdL1 = Element_List[e].NdL1;
        const Eigen::VectorXd& NdL2 = Element_List[e].NdL2;

        // === Assemble R1 ===
        int dim = 1;
        int first = 0;
        int last = dim;

        for (int i = 0; i < NPE1; ++i) {
            int node_i = static_cast<int>(NdL1(i)) - 1;
            const auto& BC_i = Node_List[node_i].BC;
            const auto& DOF_i = Node_List[node_i].DOF;

            for (int p = first; p < last; ++p) {
                if (BC_i(p) == 1) {
                    int dof_idx = static_cast<int>(DOF_i(p)) - 1;
                    int r_idx = i * dim + (p - first);
                    if (dof_idx >= 0) {
                        Rtot(dof_idx) += R1(r_idx);
                    }
                }
            }
        }

        // === Assemble R2 ===
        dim = PD;
        first = last;
        last = first + dim;

        for (int i = 0; i < NPE2; ++i) {
            int node_i = static_cast<int>(NdL2(i)) - 1;
            const auto& BC_i = Node_List[node_i].BC;
            const auto& DOF_i = Node_List[node_i].DOF;

            for (int p = first; p < last; ++p) {
                if (BC_i(p) == 1) {
                    int dof_idx = static_cast<int>(DOF_i(p)) - 1;
                    int r_idx = i * dim + (p - first);
                    if (dof_idx >= 0) {
                        Rtot(dof_idx) += R2(r_idx);
                    }
                }
            }
        }

        // === Assemble K11 & K12 ===
        int dim_i = 1;
        int first_i = 0;
        int last_i = dim_i;

        for (int i = 0; i < NPE1; ++i) {
            int node_i = static_cast<int>(NdL1(i)) - 1;
            const auto& BC_i = Node_List[node_i].BC;
            const auto& DOF_i = Node_List[node_i].DOF;

            for (int p = first_i; p < last_i; ++p) {
                if (BC_i(p) == 1) {
                    int row = static_cast<int>(DOF_i(p)) - 1;

                    // K11
                    int dim_j = 1;
                    int first_j = 0;
                    int last_j = dim_j;

                    for (int j = 0; j < NPE1; ++j) {
                        int node_j = static_cast<int>(NdL1(j)) - 1;
                        const auto& BC_j = Node_List[node_j].BC;
                        const auto& DOF_j = Node_List[node_j].DOF;

                        for (int q = first_j; q < last_j; ++q) {
                            if (BC_j(q) == 1) {
                                int col = static_cast<int>(DOF_j(q)) - 1;
                                int k_idx_row = i * dim_i + (p - first_i);
                                int k_idx_col = j * dim_j + (q - first_j);
                                triplets.emplace_back(row, col, K11(k_idx_row, k_idx_col));
                            }
                        }
                    }

                    // K12
                    dim_j = PD;
                    first_j = last_j;
                    last_j = first_j + dim_j;

                    for (int j = 0; j < NPE2; ++j) {
                        int node_j = static_cast<int>(NdL2(j)) - 1;
                        const auto& BC_j = Node_List[node_j].BC;
                        const auto& DOF_j = Node_List[node_j].DOF;

                        for (int q = first_j; q < last_j; ++q) {
                            if (BC_j(q) == 1) {
                                int col = static_cast<int>(DOF_j(q)) - 1;
                                int k_idx_row = i * dim_i + (p - first_i);
                                int k_idx_col = j * dim_j + (q - first_j);
                                triplets.emplace_back(row, col, K12(k_idx_row, k_idx_col));
                            }
                        }
                    }
                }
            }
        }

        // === Assemble K21 & K22 ===
        dim_i = PD;
        first_i = 1;
        last_i = first_i + dim_i;

        for (int i = 0; i < NPE2; ++i) {
            int node_i = static_cast<int>(NdL2(i)) - 1;
            const auto& BC_i = Node_List[node_i].BC;
            const auto& DOF_i = Node_List[node_i].DOF;

            for (int p = first_i; p < last_i; ++p) {
                if (BC_i(p) == 1) {
                    int row = static_cast<int>(DOF_i(p)) - 1;

                    // K21
                    int dim_j = 1;
                    int first_j = 0;
                    int last_j = dim_j;

                    for (int j = 0; j < NPE1; ++j) {
                        int node_j = static_cast<int>(NdL1(j)) - 1;
                        const auto& BC_j = Node_List[node_j].BC;
                        const auto& DOF_j = Node_List[node_j].DOF;

                        for (int q = first_j; q < last_j; ++q) {
                            if (BC_j(q) == 1) {
                                int col = static_cast<int>(DOF_j(q)) - 1;
                                int k_idx_row = i * dim_i + (p - first_i);
                                int k_idx_col = j * dim_j + (q - first_j);
                                triplets.emplace_back(row, col, K21(k_idx_row, k_idx_col));
                            }
                        }
                    }

                    // K22
                    dim_j = PD;
                    first_j = 1;
                    last_j = first_j + dim_j;

                    for (int j = 0; j < NPE2; ++j) {
                        int node_j = static_cast<int>(NdL2(j)) - 1;
                        const auto& BC_j = Node_List[node_j].BC;
                        const auto& DOF_j = Node_List[node_j].DOF;

                        for (int q = first_j; q < last_j; ++q) {
                            if (BC_j(q) == 1) {
                                int col = static_cast<int>(DOF_j(q)) - 1;
                                int k_idx_row = i * dim_i + (p - first_i);
                                int k_idx_col = j * dim_j + (q - first_j);
                                triplets.emplace_back(row, col, K22(k_idx_row, k_idx_col));
                            }
                        }
                    }
                }
            }
        }
    }

    // Finalize global stiffness matrix
    Ktot = Eigen::SparseMatrix<double>(DOFs, DOFs);
    Ktot.setFromTriplets(triplets.begin(), triplets.end());
}

void problem_coupled::assemble_GP(double dt) {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();
    int NPE1 = Element_List[0].NPE1;
    int NGP_val = Node_List[0].GP_BC.size();  // Number of Gauss point values

    Rtot = Eigen::VectorXd::Zero(GP_DOFs * NGP_val);
    std::vector<Eigen::Triplet<double>> triplets;

    for (int e = 0; e < NoEs; ++e) {
        auto [R, K] = Element_List[e].RK_GP(dt, NGP_val);
        const Eigen::VectorXd& NdL1 = Element_List[e].NdL1;

        // === Assemble Rtot_GP ===
        for (int s = 0; s < NGP_val; ++s) {
            for (int i = 0; i < NPE1; ++i) {
                int node_i = static_cast<int>(NdL1(i)) - 1;
                const auto& BC = Node_List[node_i].GP_BC;
                const auto& DOF = Node_List[node_i].GP_DOF;

                for (int p = 0; p < 1; ++p) {  // scalar field only
                    if (BC(p) == 1) {
                        int dof_idx = static_cast<int>(DOF(p)) - 1;
                        if (dof_idx >= 0) {
                            int global_row = dof_idx * NGP_val + s;
                            int local_row = i * 1 + (p - 0);  // equivalent to i in this case
                            Rtot(global_row) += R(local_row, s);
                        }
                    }
                }
            }
        }

        // === Assemble Ktot_GP ===
        for (int i = 0; i < NPE1; ++i) {
            int node_i = static_cast<int>(NdL1(i)) - 1;
            const auto& BC_i = Node_List[node_i].GP_BC;
            const auto& DOF_i = Node_List[node_i].GP_DOF;

            for (int p = 0; p < 1; ++p) {
                if (BC_i(p) == 1) {
                    int row = static_cast<int>(DOF_i(p)) - 1;

                    for (int j = 0; j < NPE1; ++j) {
                        int node_j = static_cast<int>(NdL1(j)) - 1;
                        const auto& BC_j = Node_List[node_j].GP_BC;
                        const auto& DOF_j = Node_List[node_j].GP_DOF;

                        for (int q = 0; q < 1; ++q) {
                            if (BC_j(q) == 1) {
                                int col = static_cast<int>(DOF_j(q)) - 1;
                                int k_idx_row = i * 1 + (p - 0);
                                int k_idx_col = j * 1 + (q - 0);
                                triplets.emplace_back(row, col, K(k_idx_row, k_idx_col));
                            }
                        }
                    }
                }
            }
        }
    }

    // Finalize Ktot_GP sparse matrix
    Ktot = Eigen::SparseMatrix<double>(GP_DOFs, GP_DOFs);
    Ktot.setFromTriplets(triplets.begin(), triplets.end());
}

Eigen::MatrixXd problem_coupled::Get_all_velocity() {
    int NoNs = Node_List.size();
    int PD = this->PD;

    Eigen::MatrixXd v(PD, NoNs);

    for (int i = 0; i < NoNs; ++i) {
        v.col(i) = Node_List[i].u.segment(1, PD);  // Skip scalar field at index 0
    }

    return v;
}

#include <fstream>
#include <iomanip>
#include <iostream>

void problem_coupled::problem_info() {
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();

    std::cout << "======================================================" << std::endl;
    std::cout << "================  Problem information  ===============" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Problem dimension                   : " << PD << std::endl;
    std::cout << "Time increment                      : " << dt << std::endl;
    std::cout << "Final time                          : " << T << std::endl;
    std::cout << "Time increment method               : " << time_incr_method << std::endl;
    std::cout << "Number of nodes                     : " << NoNs << std::endl;
    std::cout << "Number of bulk elements             : " << NoEs << std::endl;
    std::cout << "Number of DOFs                      : " << DOFs << std::endl;

    std::cout << "Element order                       : [";
    for (size_t i = 0; i < element_order.size(); ++i) {
        std::cout << element_order[i];
        if (i != element_order.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "E  R  xi                            : [";
    for (size_t i = 0; i < parameters.size(); ++i) {
        std::cout << parameters[i];
        if (i != parameters.size() - 1) std::cout << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "======================================================" << std::endl;

    std::ofstream file(filename);
    file << "======================================================\n";
    file << "================  Problem information  ===============\n";
    file << "======================================================\n";
    file << "Problem dimension                   : " << PD << "\n";
    file << "Time increment                      : " << dt << "\n";
    file << "Final time                          : " << T << "\n";
    file << "Time increment method               : " << time_incr_method << "\n";
    file << "Number of nodes                     : " << NoNs << "\n";
    file << "Number of bulk elements             : " << NoEs << "\n";
    file << "Number of DOFs                      : " << DOFs << "\n";

    file << "Element order                       : ";
    for (int val : element_order)
        file << val << " ";
    file << "\n";

    file << "E  R  xi                            : ";
    for (double val : parameters)
        file << val << " ";
    file << "\n";

    file << "======================================================\n\n\n";
    file.close();
}

std::pair<double, double> problem_coupled::calculate_max_min_difference() {
    int NoNs = Node_List.size();
    int PD = this->PD;

    std::vector<double> c_values, p_values;
    c_values.reserve(NoNs);
    p_values.reserve(NoNs);

    for (int i = 0; i < NoNs; ++i) {
        const auto& u = Node_List[i].u;
        c_values.push_back(u(0));              // c is the first entry
        p_values.push_back(u(1 + PD));         // p is after c and velocity
    }

    auto [c_min_it, c_max_it] = std::minmax_element(c_values.begin(), c_values.end());
    auto [p_min_it, p_max_it] = std::minmax_element(p_values.begin(), p_values.end());

    double c_range = *c_max_it - *c_min_it;
    double p_range = *p_max_it - *p_min_it;

    return {c_range, p_range};
}

//
//double problem_coupled::calculate_overall_density() {
//    double out = 0.0;
//    int NoE = Element_List.size();
//
//    for (int i = 0; i < NoE; ++i) {
//        out += Element_List[i].density;
//    }
//
//    return out;
//}
