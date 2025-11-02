
#include "problem.hpp"


#include "../postprocess/postprocess.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>


void writeTripletsToFile(const Eigen::SparseMatrix<double>& A,
                    const std::string& path,
                    bool one_based = true);
class Timer {
public:
    Timer(const std::string& name = "Timer")
            : name(name), start(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << " took " << duration << " ms\n";
    }
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
};

template <class Derived>
void writeVectorToFile(const Eigen::MatrixBase<Derived>& v_expr,
                       const std::string& path,
                       bool withIndex = false,
                       bool oneBased  = true,
                       int precision  = 17)
{
    const auto& v = v_expr.derived();
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    f << std::setprecision(precision);
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        if (withIndex) {
            long long idx = static_cast<long long>(i) + (oneBased ? 1 : 0);
            f << idx << " " << v(i) << "\n";
        } else {
            f << v(i) << "\n";
        }
    }
    f.close();
}

struct MatrixHealth {
    int rows = 0, cols = 0;
    Eigen::Index nnz = 0;
    int zero_rows = 0, zero_cols = 0;
    Eigen::Index nonfinite = 0;
    int zero_diagonal = 0;
    int inconsistent_zero_rows = 0;    // zero row but R(i) != 0
    double rel_asym = std::numeric_limits<double>::quiet_NaN(); // only if expectSym
    double ones_resid = std::numeric_limits<double>::quiet_NaN(); // ||K*1||
    bool ok = false;
};

inline MatrixHealth check_matrix_health(const Eigen::SparseMatrix<double>& Ktot,
                                        const Eigen::VectorXd* Rtot = nullptr,
                                        bool expectSym = false,
                                        double tol = 1e-12,
                                        int print_limit = 10,
                                        bool verbose=false)
{

    MatrixHealth H;
    Eigen::SparseMatrix<double> A = Ktot;
    if (!A.isCompressed()) A.makeCompressed();

    H.rows = A.rows();
    H.cols = A.cols();
    H.nnz  = A.nonZeros();

    std::vector<int> row_nnz(H.rows, 0), col_nnz(H.cols, 0);

    // scan entries for counts and nonfinite
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            const double v = it.value();
            row_nnz[i]++; col_nnz[j]++;
            if (!std::isfinite(v)) {
                ++H.nonfinite;
                if (H.nonfinite <= print_limit && verbose)
                    std::cerr << "nonfinite at (" << i << "," << j << ") = " << v << "\n";
            }
        }
    }

    // zero rows and columns
    for (int i = 0; i < H.rows; ++i) {
        if (row_nnz[i] == 0) {
            ++H.zero_rows;
            if (H.zero_rows <= print_limit && verbose) std::cerr << "zero row " << i << "\n";
        }
    }
    for (int j = 0; j < H.cols; ++j) {
        if (col_nnz[j] == 0) {
            ++H.zero_cols;
            if (H.zero_cols <= print_limit && verbose) std::cerr << "zero col " << j << "\n";
        }
    }

    // zero diagonal
    for (int i = 0; i < std::min(H.rows, H.cols); ++i) {
        if (A.coeff(i, i) == 0.0) {
            ++H.zero_diagonal;
            if (H.zero_diagonal <= print_limit && verbose) std::cerr << "zero diagonal at " << i << "\n";
        }
    }

    // optional inconsistency check with RHS
    if (Rtot && Rtot->size() == H.rows) {
        for (int i = 0; i < H.rows; ++i) {
            if (row_nnz[i] == 0 && std::abs((*Rtot)(i)) > tol) {
                ++H.inconsistent_zero_rows;
                if (H.inconsistent_zero_rows <= print_limit && verbose)
                    std::cerr << "zero row " << i << " but R(" << i << ") = " << (*Rtot)(i) << "\n";
            }
        }
    }

    // optional symmetry check
    if (expectSym) {
        Eigen::SparseMatrix<double> AT = A.transpose();
        Eigen::SparseMatrix<double> Skew = A - AT;
        const double nrmA = A.norm();
        const double nrmSkew = Skew.norm();
        H.rel_asym = nrmSkew / std::max(1e-30, nrmA);
        std::cerr << "relative asymmetry = " << H.rel_asym << "\n";
    }



    H.ok = (H.nonfinite == 0
            && H.zero_rows == 0
            && H.zero_cols == 0
            && (!expectSym || H.rel_asym < 1e-8));
    if (verbose) {
        std::cerr << "health: nnz=" << H.nnz
                  << " zero_rows=" << H.zero_rows
                  << " zero_cols=" << H.zero_cols
                  << " zero_diag=" << H.zero_diagonal
                  << " nonfinite=" << H.nonfinite
                  << " inconsistent_zero_rows=" << H.inconsistent_zero_rows
                  << " ok=" << (H.ok ? "yes" : "no") << "\n";

        // quick nullspace hint
        {
            Eigen::VectorXd ones = Eigen::VectorXd::Ones(H.cols);
            H.ones_resid = (A * ones).norm();
            std::cerr << "||K*1|| = " << H.ones_resid << "\n";
        }
    }

    return H;
}


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
    BCset = false;
    filename = oss.str();

    // === Assign boundary conditions ===
    Assign_BC(Corners);  // fills BC and DOFs
    //Assign_DOF_DBC();    // computes global DOF numbering

    // === Assign Gauss Point DOFs if requested ===
    if (GP_vals == "On") {
        Assign_GP_DOFs();
    }

    // === Output problem setup info ===
    problem_info();  // displays PD, mesh, DOFs, etc.
    solve();
}



void problem_coupled::Assign_BC(const std::string Corners) {
    //Timer t("Assigning BC");
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
        // minus/plus edges (X, Y, Z) and minus/plus faces (X, Y, Z)
        std::vector<int> MEX, PEX, MEY, PEY, MEZ, PEZ;
        std::vector<int> MFX, PFX, MFY, PFY, MFZ, PFZ;

        auto on = [&](double a, double b) { return std::abs(a - b) < tol; };

        for (int i = 0; i < NoNs; ++i) {
            const auto& Xv = Node_List[i].X;      // Xv(0)=x, Xv(1)=y, Xv(2)=z
            const double x = Xv(0), y = Xv(1), z = Xv(2);

            // corner nodes
            if ( (on(x,0) && on(y,0) && on(z,0)) || (on(x,0) && on(y,0) && on(z,D)) ||
                 (on(x,0) && on(y,H) && on(z,0)) || (on(x,0) && on(y,H) && on(z,D)) ||
                 (on(x,W) && on(y,0) && on(z,0)) || (on(x,W) && on(y,0) && on(z,D)) ||
                 (on(x,W) && on(y,H) && on(z,0)) || (on(x,W) && on(y,H) && on(z,D)) ) {
                NLC.push_back(i);
            }
                // X = 0 plane (minus X)
            else if (on(x,0)) {
                if      (on(y,0)) { MEZ.push_back(i); }     // edge: X=0, Y=0  (vary z)
                else if (on(y,H)) { PEZ.push_back(i); }     // edge: X=0, Y=H
                else if (on(z,0)) { MEY.push_back(i); }     // edge: X=0, Z=0  (vary y)
                else if (on(z,D)) { PEY.push_back(i); }     // edge: X=0, Z=D
                else              { MFX.push_back(i); }     // face interior: X=0
            }
                // Y = 0 plane (minus Y)
            else if (on(y,0)) {
                if      (on(x,W)) { PEZ.push_back(i); }     // edge: X=W, Y=0
                else if (on(z,0)) { MEX.push_back(i); }     // edge: Y=0, Z=0  (vary x)
                else if (on(z,D)) { PEX.push_back(i); }     // edge: Y=0, Z=D
                else              { MFY.push_back(i); }     // face interior: Y=0
            }
                // Z = 0 plane (minus Z)
            else if (on(z,0)) {
                if      (on(x,W)) { PEY.push_back(i); }     // edge: X=W, Z=0
                else if (on(y,H)) { PEX.push_back(i); }     // edge: Y=H, Z=0
                else              { MFZ.push_back(i); }     // face interior: Z=0
            }
                // X = W plane (plus X)
            else if (on(x,W)) {
                if      (on(y,H)) { PEZ.push_back(i); }     // edge: X=W, Y=H
                else if (on(z,D)) { PEY.push_back(i); }     // edge: X=W, Z=D
                else              { PFX.push_back(i); }     // face interior: X=W
            }
                // Y = H plane (plus Y)
            else if (on(y,H)) {
                if      (on(z,D)) { PEX.push_back(i); }     // edge: Y=H, Z=D
                else              { PFY.push_back(i); }     // face interior: Y=H
            }
                // Z = D plane (plus Z)
            else if (on(z,D)) {
                PFZ.push_back(i);                           // face interior: Z=D
            }
                // interior "star" nodes
            else {
                NLS.push_back(i);
            }
        }

        // X-direction edges: compare x only
        for (int pi : PEX) {
            for (int mi : MEX) {
                if (on(Node_List[pi].X(0), Node_List[mi].X(0))) {
                    NLM.push_back(mi);
                    NLP.push_back(pi);
                }
            }
        }

        // Y-direction edges: compare y only
        for (int pi : PEY) {
            for (int mi : MEY) {
                if (on(Node_List[pi].X(1), Node_List[mi].X(1))) {
                    NLM.push_back(mi);
                    NLP.push_back(pi);
                }
            }
        }

        // Z-direction edges: compare z only
        for (int pi : PEZ) {
            for (int mi : MEZ) {
                if (on(Node_List[pi].X(2), Node_List[mi].X(2))) {
                    NLM.push_back(mi);
                    NLP.push_back(pi);
                }
            }
        }

        // X-faces (X = W vs X = 0): match (y,z)
        for (int pi : PFX) {
            for (int mi : MFX) {
                if (on(Node_List[pi].X(1), Node_List[mi].X(1)) &&
                    on(Node_List[pi].X(2), Node_List[mi].X(2))) {
                    NLM.push_back(mi);
                    NLP.push_back(pi);
                }
            }
        }

        // Y-faces (Y = H vs Y = 0): match (x,z)
        for (int pi : PFY) {
            for (int mi : MFY) {
                if (on(Node_List[pi].X(0), Node_List[mi].X(0)) &&
                    on(Node_List[pi].X(2), Node_List[mi].X(2))) {
                    NLM.push_back(mi);
                    NLP.push_back(pi);
                }
            }
        }

        // Z-faces (Z = D vs Z = 0): match (x,y)
        for (int pi : PFZ) {
            for (int mi : MFZ) {
                if (on(Node_List[pi].X(0), Node_List[mi].X(0)) &&
                    on(Node_List[pi].X(1), Node_List[mi].X(1))) {
                    NLM.push_back(mi);
                    NLP.push_back(pi);
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
    int NoNs = static_cast<int>(Node_List.size());
    int dofs = 0;

    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd BC  = Node_List[i].BC;   // copy, like MATLAB
        Eigen::VectorXd DOF = Node_List[i].DOF;  // copy

        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1.0) {
                std::cout<< "node: "<< i<<std::endl;
                dofs = dofs + 1;
                DOF(p) = dofs;
            }
        }

        Node_List[i].DOF = DOF;  // write back
    }

    DOFs = dofs;  // total
}



// Assign DOFs with periodic BC (node index lists are already 0-based)
void problem_coupled::Assign_DOF_PBC(const std::vector<int>& NLC,
                                     const std::vector<int>& NLS,
                                     const std::vector<int>& NLM,
                                     const std::vector<int>& NLP)
{
    DOFs = 0;

    const int NoCNs = static_cast<int>(NLC.size());
    const int NoSNs = static_cast<int>(NLS.size());
    const int NoMNs = static_cast<int>(NLM.size());
    const int NoPNs = static_cast<int>(NLP.size());

    // ===== Corner nodes =====
    if (NoCNs != 0) {
        const int firstCorner = NLC[0];
        auto& DOF0 = Node_List[firstCorner].DOF;
        const auto& BC0 = Node_List[firstCorner].BC;

        // only for the first corner (as in MATLAB)
        for (Eigen::Index p = 0; p < BC0.size(); ++p) {
            if (BC0(p) == 1) {
                ++DOFs;
                DOF0(p) = static_cast<double>(DOFs);
            }
        }
        Node_List[firstCorner].DOF = DOF0;

        // copy first corner DOF to the other corners
        for (int i = 0; i < NoCNs; ++i) {
            const int idx = NLC[i];
            Node_List[idx].DOF = DOF0;
        }
    }

    // ===== Star nodes =====
    for (int i = 0; i < NoSNs; ++i) {
        const int idx = NLS[i];
        auto& DOF = Node_List[idx].DOF;
        const auto& BC = Node_List[idx].BC;

        for (Eigen::Index p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1) {
                ++DOFs;
                DOF(p) = static_cast<double>(DOFs);
            }
        }
        Node_List[idx].DOF = DOF;
    }

    // ===== Minus nodes =====
    for (int i = 0; i < NoMNs; ++i) {
        const int idx = NLM[i];

        // guard to avoid overwriting (the key fix)
        if (!Node_List[idx].BCset) {
            auto& DOF = Node_List[idx].DOF;
            const auto& BC = Node_List[idx].BC;

            for (Eigen::Index p = 0; p < BC.size(); ++p) {
                if (BC(p) == 1) {
                    ++DOFs;
                    DOF(p) = static_cast<double>(DOFs);
                }
            }
            Node_List[idx].DOF = DOF;
            Node_List[idx].BCset = true;
        }
    }

    // ===== Plus nodes =====

    const int nPair = std::min(NoPNs, NoMNs); // safety; MATLAB assumes equal length
    for (int i = 0; i < nPair; ++i) {
        const int idxP = NLP[i];
        const int idxM = NLM[i];

        Node_List[idxP].BC  = Node_List[idxM].BC;
        Node_List[idxP].DOF = Node_List[idxM].DOF;
    }
}

void problem_coupled::Assign_GP_DOFs() {
    int GP_dofs = 0;
    const int NoNs = static_cast<int>(Node_List.size());
    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC  = Node_List[i].GP_BC;   // length = NGP_val  (e.g., PD*PD)
        Eigen::VectorXd& DOF = Node_List[i].GP_DOF;  // length = NGP_val
        if ((BC.array() == 1).any()) {
            ++GP_dofs;
            // all GP components at the same node share ONE (the same) DOF id
            DOF.setConstant(BC.size(), static_cast<double>(GP_dofs));
        } else {
            DOF.setZero(BC.size());
        }
    }
    GP_DOFs = GP_dofs;
}



void problem_coupled::update(const Eigen::VectorXd& dx) {
    int NoNs = Node_List.size();
    int NoEs = Element_List.size();
    int PD   = this->PD;

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

        const int NPE1 = elem.NPE1;
        const int NPE2 = elem.NPE2;

        // === Scalar field c (keep EXACTLY as before) ===
        Eigen::VectorXd c(NPE1);
        for (int i = 0; i < NPE1; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL1(i)) - 1; // 1-based -> 0-based
            c(i) = Node_List[nodeIdx].u(0);                  // c is the first entry
        }
        elem.c = c;


        const int dim   = PD;
        int last0  = 0;          // after c at index 0 in C++
        int first0 = last0 + 1;  // v starts right after c -> index 1
        last0     += dim;        // end index of v (unused here, kept for clarity)

        Eigen::MatrixXd v(PD, NPE2);
        for (int i = 0; i < NPE2; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL2(i)) - 1; // 1-based -> 0-based
            v.col(i) = Node_List[nodeIdx].u.segment(first0, dim); // u(first0 : first0+PD-1)
        }

        elem.v = v;

    }

}

void problem_coupled::update_GP(const Eigen::MatrixXd& dx_gp) {
    const int NoNs = static_cast<int>(Node_List.size());
    if (dx_gp.rows() == 0) return;
    for (int i = 0; i < NoNs; ++i) {
        const auto& BC  = Node_List[i].GP_BC;
        const auto& DOF = Node_List[i].GP_DOF;
        if ((BC.array() == 1).any()) {
            const int d = static_cast<int>(DOF(0)) - 1;
            if (0 <= d && d < dx_gp.rows()) Node_List[i].GP_vals = dx_gp.row(d).transpose();
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


// Global residual assembly (R1 and R2)
Eigen::VectorXd problem_coupled::Residual(double dt)
{
    const int NoEs  = static_cast<int>(Element_List.size());
    if (NoEs == 0) return Eigen::VectorXd::Zero(DOFs);

    const int NPE1 = Element_List[0].NPE1;
    const int NPE2 = Element_List[0].NPE2;

    Eigen::VectorXd Rtot = Eigen::VectorXd::Zero(DOFs);

    for (int e = 0; e < NoEs; ++e) {
        auto [R1, R2] = Element_List[e].Residual(dt);
        const auto& NdL1 = Element_List[e].NdL1; // 1-based ids
        const auto& NdL2 = Element_List[e].NdL2;

        // R1 (scalar)
        {
            const int dim    = 1;          // cell
            const int first0 = 0;          // MATLAB first=1  ->  C++ 0
            const int last0  = first0 + dim - 1; // = 0

            for (int i = 0; i < NPE1; ++i) {
                const int nodeIdx = static_cast<int>(NdL1(i)) - 1;   // 1-based -> 0-based
                const auto& BC  = Node_List[nodeIdx].BC;
                const auto& DOF = Node_List[nodeIdx].DOF;

                for (int p0 = first0; p0 <= last0; ++p0) {          // only p0 = 0
                    if (BC(p0) == 1) {
                        const int g = static_cast<int>(DOF(p0)) - 1;    // global DOF 1-based -> 0-based
                        const int r = i * dim + (p0 - first0);          // = i
                        if (g >= 0 && g < Rtot.size() && r >= 0 && r < R1.size()) {
                            Rtot(g) += R1(r);
                        }
                    }
                }
            }
        }
        //R2 vector
        {
            const int dim   = this->PD;       // velocity block size
            const int first0 = 1;             // zero-based: v starts right after c at index 0
            const int last0  = first0 + dim - 1;

            for (int i = 0; i < NPE2; ++i) {
                const int nodeIdx = static_cast<int>(NdL2(i)) - 1;   // 1-based -> 0-based
                const auto& BC  = Node_List[nodeIdx].BC;
                const auto& DOF = Node_List[nodeIdx].DOF;

                for (int p0 = first0; p0 <= last0; ++p0) {          // p0 in [1, PD]
                    if (BC(p0) == 1) {
                        const int g = static_cast<int>(DOF(p0)) - 1;    // global DOF: 1-based -> 0-based
                        const int r = i * dim + (p0 - first0);          // R2 index: 0..(NPE2*PD-1)
                        if (g >= 0 && g < Rtot.size() && r >= 0 && r < R2.size()) {
                            Rtot(g) += R2(r);
                        }
                    }
                }
            }
        }

    }

    return Rtot;
}



void problem_coupled::assemble(double dt) {
    //Timer t("Assamble");
    // === MATLAB: NoEs = size(EL,2); NoNs = size(NL,2); ===
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());

    // Guard for empty meshes (optional, but safer)
    if (NoEs == 0 || NoNs == 0) {
        Rtot = Eigen::VectorXd::Zero(DOFs);
        Ktot = Eigen::SparseMatrix<double>(DOFs, DOFs);
        return;
    }

    // === MATLAB: NPE1 = EL(1).NPE1; NPE2 = EL(1).NPE2; ===
    const int NPE1 = Element_List[0].NPE1;
    const int NPE2 = Element_List[0].NPE2;

    // === MATLAB: Rtot = zeros(DOFs,1); ===
    Rtot = Eigen::VectorXd::Zero(DOFs);

    // === MATLAB: sprC = 0; % sparse counter ===
    // We'll track triplets instead of a manual counter.
    std::vector<Eigen::Triplet<double>> triplets;

    // === MATLAB: for e = 1:NoEs ... end  (empty body for now) ===
    for (int e = 0; e < NoEs; ++e) {
        // Assemble contribution of R1

            auto [R1, R2, K11, K12, K21, K22] = Element_List[e].RK(dt);


        // Per element e
        struct Slot { int g; int loc; }; // global dof, local index (within element block)

// Pre-allocate small fixed arrays for scalar rows/cols (NPE1 is small)
        std::array<Slot, 64> rows_scalar;  int n_rows_scalar = 0;
        std::array<Slot, 64> cols_scalar;  int n_cols_scalar = 0;

// Velocity rows/cols are dynamic (NPE2*PD)
        std::vector<Slot> rows_vel; rows_vel.reserve(NPE2 * PD);
        std::vector<Slot> cols_vel; cols_vel.reserve(NPE2 * PD);

// ---------- Collect scalar rows (NdL1, p=0) and do R1 residual ----------
        for (int i = 0; i < NPE1; ++i) {
            const int node = static_cast<int>(Element_List[e].NdL1(i)) - 1;
            const auto& BC  = Node_List[node].BC;
            if (BC(0) != 1) continue;

            const auto& DOF = Node_List[node].DOF;
            const int g  = static_cast<int>(DOF(0)) - 1;  // 0-based
            const int li = i;                              // dim=1 => local = i
            rows_scalar[n_rows_scalar++] = {g, li};
            Rtot(g) += R1(li);
        }

// ---------- Collect scalar cols (NdL1, q=0). Reused by K11 and K21 ----------
        for (int j = 0; j < NPE1; ++j) {
            const int node = static_cast<int>(Element_List[e].NdL1(j)) - 1;
            const auto& BC  = Node_List[node].BC;
            if (BC(0) != 1) continue;

            const auto& DOF = Node_List[node].DOF;
            const int g  = static_cast<int>(DOF(0)) - 1;
            const int lj = j; // dim=1
            cols_scalar[n_cols_scalar++] = {g, lj};
        }

// ---------- Collect velocity rows (NdL2, p=1..PD) and do R2 residual ----------
        for (int i = 0; i < NPE2; ++i) {
            const int node = static_cast<int>(Element_List[e].NdL2(i)) - 1;
            const auto& BC  = Node_List[node].BC;
            const auto& DOF = Node_List[node].DOF;

            for (int p = 1; p <= PD; ++p) {
                if (BC(p) != 1) continue;
                const int g   = static_cast<int>(DOF(p)) - 1;
                const int li  = i * PD + (p - 1);
                rows_vel.push_back({g, li});
                Rtot(g) += R2(li);
            }
        }

// ---------- Collect velocity cols (NdL2, q=1..PD). Reused by K12 and K22 ----------
        for (int j = 0; j < NPE2; ++j) {
            const int node = static_cast<int>(Element_List[e].NdL2(j)) - 1;
            const auto& BC  = Node_List[node].BC;
            const auto& DOF = Node_List[node].DOF;

            for (int q = 1; q <= PD; ++q) {
                if (BC(q) != 1) continue;
                const int g   = static_cast<int>(DOF(q)) - 1;
                const int lj  = j * PD + (q - 1);
                cols_vel.push_back({g, lj});
            }
        }

// ---------- Reserve triplets once for all four blocks ----------
        {
            const int rs = n_rows_scalar;
            const int cs = n_cols_scalar;
            const int rv = static_cast<int>(rows_vel.size());
            const int cv = static_cast<int>(cols_vel.size());
            triplets.reserve(triplets.size() + rs*cs + rs*cv + rv*cs + rv*cv);
        }

// ---------- K11: scalar rows x scalar cols ----------
        for (int r = 0; r < n_rows_scalar; ++r) {
            const int grow = rows_scalar[r].g;
            const int li   = rows_scalar[r].loc;
            for (int c = 0; c < n_cols_scalar; ++c) {
                const int gcol = cols_scalar[c].g;
                const int lj   = cols_scalar[c].loc;
                triplets.emplace_back(grow, gcol, K11(li, lj));
            }
        }

// ---------- K12: scalar rows x velocity cols ----------
        for (int r = 0; r < n_rows_scalar; ++r) {
            const int grow = rows_scalar[r].g;
            const int li   = rows_scalar[r].loc;
            for (const auto& cv : cols_vel) {
                triplets.emplace_back(grow, cv.g, K12(li, cv.loc));
            }
        }

// ---------- K21: velocity rows x scalar cols ----------
        for (const auto& rv_ : rows_vel) {
            for (int c = 0; c < n_cols_scalar; ++c) {
                const auto& cs = cols_scalar[c];
                triplets.emplace_back(rv_.g, cs.g, K21(rv_.loc, cs.loc));
            }
        }

// ---------- K22: velocity rows x velocity cols ----------
        for (const auto& rv_ : rows_vel) {
            for (const auto& cv : cols_vel) {
                triplets.emplace_back(rv_.g, cv.g, K22(rv_.loc, cv.loc));
            }
        }

        const auto& NdL1 = Element_List[e].NdL1;
        const auto& NdL2 = Element_List[e].NdL2;

    }

    // Build an (empty-for-now) global K to keep the function self-contained
    Ktot = Eigen::SparseMatrix<double>(DOFs, DOFs);
    Ktot.setFromTriplets(triplets.begin(), triplets.end());

}

void problem_coupled::assemble_GP(double dt) {
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());
    if (NoEs == 0 || NoNs == 0 || GP_DOFs == 0) {
        Rtot_GP.resize(0,0);
        Ktot_GP.resize(0,0);
        return;
    }
    const int NPE1 = Element_List[0].NPE1;
    const int NGP_val = static_cast<int>(Node_List[0].GP_BC.size()); // PD*PD

    Rtot_GP.setZero(GP_DOFs, NGP_val);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(NoEs * NPE1 * NPE1);

    for (int e = 0; e < NoEs; ++e) {
        auto ek = Element_List[e].RK_GP(dt, NGP_val);
        const Eigen::MatrixXd& R = ek.first;    // (NPE1 × NGP_val)
        const Eigen::MatrixXd& K = ek.second;   // (NPE1 × NPE1)
        const auto& NdL1 = Element_List[e].NdL1;

        // R
        for (int i = 0; i < NPE1; ++i) {
            const int node_i = static_cast<int>(NdL1(i)) - 1;
            const auto& BC  = Node_List[node_i].GP_BC;
            const auto& DOF = Node_List[node_i].GP_DOF;
            if ((BC.array() == 1).any()) {
                const int gi = static_cast<int>(DOF(0)) - 1;
                if (gi >= 0) Rtot_GP.row(gi) += R.row(i);
            }
        }

        // K
        for (int i = 0; i < NPE1; ++i) {
            const int node_i = static_cast<int>(NdL1(i)) - 1;
            const auto& BCi  = Node_List[node_i].GP_BC;
            const auto& DOFi = Node_List[node_i].GP_DOF;
            if ((BCi.array() == 1).any()) {
                const int gi = static_cast<int>(DOFi(0)) - 1;
                if (gi < 0) continue;
                for (int j = 0; j < NPE1; ++j) {
                    const int node_j = static_cast<int>(NdL1(j)) - 1;
                    const auto& BCj  = Node_List[node_j].GP_BC;
                    const auto& DOFj = Node_List[node_j].GP_DOF;
                    if ((BCj.array() == 1).any()) {
                        const int gj = static_cast<int>(DOFj(0)) - 1;
                        if (gj < 0) continue;
                        triplets.emplace_back(gi, gj, K(i, j));
                    }
                }
            }
        }
    }

    Ktot_GP.resize(GP_DOFs, GP_DOFs);
    Ktot_GP.setFromTriplets(triplets.begin(), triplets.end());
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



void problem_coupled::problem_info() {
    using namespace std;
    namespace fs = std::filesystem;

    if (filename.empty()) {
        throw std::runtime_error("filename está vacío");
    }
    int NoEs = Element_List.size();
    int NoNs = Node_List.size();
    cout << "======================================================" << endl;
    cout << "================  Problem information  ===============" << endl;
    cout << "======================================================" << endl;
    cout << "Problem dimension                   : " << PD << endl;
    cout << "Time increment                      : " << dt << endl;
    cout << "Final time                          : " << T << endl;
    cout << "Time increment method               : " << time_incr_method << endl;
    cout << "Number of nodes                     : " << NoNs << endl;
    cout << "Number of bulk elements             : " << NoEs << endl;
    cout << "Number of DOFs                      : " << DOFs << endl;

    cout << "Element order                       : [";
    for (size_t i = 0; i < element_order.size(); ++i) {
        cout << element_order[i] << (i + 1 < element_order.size() ? " " : "");
    }
    cout << "]" << endl;

    cout << "E  R  xi                            : [";
    for (size_t i = 0; i < parameters.size(); ++i) {
        cout << parameters[i] << (i + 1 < parameters.size() ? " " : "");
    }
    cout << "]" << endl;
    cout << "======================================================" << endl;


    fs::path outPath(filename);
    if (outPath.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(outPath.parent_path(), ec); // no lanza, revisa ec si quieres
        if (ec) {
            throw std::runtime_error("No se pudo crear el directorio: " + outPath.parent_path().string());
        }
    }


    cerr << "Working directory: " << fs::current_path().string() << "\n";
    cerr <<  "Writing in    : " << fs::absolute(outPath).string() << "\n";


    std::ofstream file;
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        file.open(outPath, std::ios::out | std::ios::trunc); // crea si no existe
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
        for (int v : element_order) file << v << " ";
        file << "\n";

        file << "E  R  xi                            : ";
        for (double v : parameters) file << v << " ";
        file << "\n";

        file << "======================================================\n\n\n";
        file.close();
    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error(std::string("Falló la escritura en '") + outPath.string() + "': " + e.what());
    }
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


// ------------ Post Process ------------



void problem_coupled::post_process() {
    // Minimal stub so you can compile/run. Hook your 1D/2D/3D exporters here.
    // write nodal c and v to file if you want.
    // Build "vtk/step_XXXX.vtk" (ParaView will group the series automatically)

        // --- 1. Configuration ---
    PostProcess::Config config;
    config.output_directory = "vtk_output";
    config.create_subdirectory = true;
    config.time_step = this->counter;
    config.problem_dimension = this->PD;
    config.write_binary_vtk = false;

    // --- 2. Instantiate the Post-Processor ---
    // Note: We now use the generalized PostProcessor class
    PostProcess::PostProcessor vtk_writer(config);

    // --- 3. Data Transfer: Nodes ---
    // Convert the simulation's Node_List into the format required by PostProcessor.
    std::vector<PostProcess::NodeData> nodes_for_vtk;
    nodes_for_vtk.reserve(Node_List.size());

    for (size_t i = 0; i < Node_List.size(); ++i) {
        const auto& sim_node = Node_List[i];
        PostProcess::NodeData vtk_node;

        vtk_node.id = static_cast<int>(i);

        // --- IMPORTANT: Map 1D/2D/3D coordinates to a 3D vector ---
        // The post-processor internally uses 3D vectors. We pad with zeros.
        vtk_node.x.setZero(); // Start with a zero vector
        for(int d=0; d < this->PD; ++d) {
            vtk_node.x(d) = sim_node.X(d);
        }

        vtk_node.u = sim_node.u;
        vtk_node.GP_vals = sim_node.GP_vals;

        // Determine active fields based on Boundary Condition flags (1 = free DOF)
        vtk_node.field.setZero();
        if (sim_node.BC.size() > 0 && sim_node.BC(0) == 1.0) {
            vtk_node.field(0) = 1; // Cell density field is active
        }

        bool velocity_active = false;
        for (int d = 1; d <= this->PD; ++d) {
            if (sim_node.BC.size() > d && sim_node.BC(d) == 1.0) {
                velocity_active = true;
                break;
            }
        }
        if (velocity_active) {
            vtk_node.field(1) = 1; // Velocity field is active
        }

        nodes_for_vtk.push_back(vtk_node);
    }
    vtk_writer.setNodeData(nodes_for_vtk);

    // --- 4. Data Transfer: Elements (no changes needed here) ---
    std::vector<PostProcess::ElementData> elements_for_vtk;
    elements_for_vtk.reserve(Element_List.size());
    for (size_t i = 0; i < Element_List.size(); ++i) {
        const auto& sim_elem = Element_List[i];
        PostProcess::ElementData vtk_elem;
        vtk_elem.id = static_cast<int>(i);
        vtk_elem.NPE1 = sim_elem.NPE1;
        vtk_elem.NPE2 = sim_elem.NPE2;
        vtk_elem.GP = sim_elem.GP;
        vtk_elem.connectivity.reserve(sim_elem.NdL1.size());
        for (Eigen::Index j = 0; j < sim_elem.NdL1.size(); ++j) {
            vtk_elem.connectivity.push_back(static_cast<int>(sim_elem.NdL1(j)) - 1);
        }
        elements_for_vtk.push_back(vtk_elem);
    }
    vtk_writer.setElementData(elements_for_vtk);

    // --- 5. Process and Write Files ---
   // std::cout << "\n--- Post-processing for step " << counter << " (" << this->PD << "D) ---" << std::endl;
    vtk_writer.process();
}

void problem_coupled::output_step_info() {
    std::cout << "\nStep number: " << counter
         << ",   Time increment: " << dt
         << ",   Current time: " << t << "\n\n";
    std::ofstream file(filename, std::ios::app);
    file << "\nStep number: " << counter
         << ",   Time increment: " << dt
         << ",   Current time: " << t << "\n\n";
}

void problem_coupled::solve() {
    // Drive the full Newton–Raphson time integration, mirroring your MATLAB flow.
    t = 0.0;
    counter = 1;

    while (t < T) {
        std::ofstream file(filename, std::ios::app);

        int  error_counter = 1;
        bool isNotAccurate = true;
        bool try_again     = false;

        while (isNotAccurate) {
            //Timer t("while loop");


            // ---------- Residual + Tangent ----------
            assemble(dt);  // fills: Ktot (Sparse), Rtot (Vector)
            double Norm_R0 = 1.0;
            auto H = check_matrix_health(Ktot);

            if (H.ok) {
                Solver = "SparseLU";
            } else {
                Solver = "LSQCG";//Least square problem using CG
            }

            // ---------- Initial output & predictor residual ----------
            if (error_counter == 1) {
                if (counter == 1 && GP_vals == "On" && GP_DOFs > 0) {
                    assemble_GP(dt);
                    const int NGP_val = static_cast<int>(Node_List[0].GP_BC.size());
                    Eigen::MatrixXd dx_GP_mat(GP_DOFs, NGP_val);
                    for (int s = 0; s < NGP_val; ++s) {
                        dx_GP_mat.col(s) = solve_dx_(Ktot_GP, Rtot_GP.col(s), false);
                    }
                    update_GP(dx_GP_mat);
                }

                Norm_R0 = Rtot.norm();
                std::cout << "Residual Norm at Predictor                : "
                          << std::scientific << Rtot.norm()
                          << " , normalized : 1\n";
                file << "Residual Norm at Predictor               : "
                     << std::scientific << Rtot.norm()
                     << " , normalized : 1\n";
            }
            Eigen::VectorXd dx = solve_dx_(Ktot, Rtot, true);

            update(dx);   // updates node and element unknowns from DOFs

            // ---------- Recompute residual with updated values ----------
            Rtot = Residual(dt);

            const double Rn = Rtot.norm();
            std::cout << "Residual Norm @ Increment " << error_counter
                      << " at Corrector : " << std::scientific << Rn
                      << " , normalized : " << (Norm_R0 > 0 ? Rn/Norm_R0 : 0.0) << "\n";
            file << "Residual Norm @ Increment " << error_counter
                 << " at Corrector : " << std::scientific << Rn
                 << " , normalized : " << (Norm_R0 > 0 ? Rn/Norm_R0 : 0.0) << "\n";

            // ---------- Convergence / failure checks ----------
            if (Rn < tol) {
                isNotAccurate = false;
                //writeVectorToFile(dx, "C:\\Users\\drva_\\CLionProjects\\coupled_field_problem_2\\vector.txt");

            } else if (error_counter > max_iter || Rn > 1e6) {
                isNotAccurate = false;
                std::cout << "Convergence is not obtained!\n";
                file << "Convergence is not obtained!\n";
                try_again = true;
            }

            ++error_counter;
        } // end Newton iterations

        // ---------- Early stop if velocities ~ 0 ----------
        {
            Eigen::MatrixXd velocity = Get_all_velocity();
            if (velocity.cwiseAbs().maxCoeff() < 1e-6) {
                post_process();
                break;
            }
        }

        // ---------- Time-step control ----------
        if (try_again) {
            if (dt < 1e-7) {
                std::cout << "\nSimulation terminated due to the very small time increment\n";
                file      << "\nSimulation terminated due to the very small time increment\n";
                break;
            } else {
                dt = dt / time_factor;
                std::cout << "\nReducing time step to: " << std::scientific << dt << "\n";
                file      << "\nReducing time step to: " << std::scientific << dt << "\n";
                downdate_time();
            }
        } else {
            // Accept the step
            t += dt;
            ++counter;

            if (GP_vals == "On") {
                assemble_GP(dt);
                Eigen::VectorXd dx_GP = solve_dx_(Ktot_GP, Rtot_GP, false);
                update_GP(dx_GP);
            }

            post_process();
            update_time(); // commit c→cn, v→vn, u→un

            if (time_incr_method == "Adaptive") {
                if (error_counter < 6) {
                    dt = dt * time_factor;
                    std::ofstream f2(filename, std::ios::app);
                    std::cout << "\nIncreasing time step to: " << std::scientific << dt << "\n\n";
                    f2        << "\nIncreasing time step to: " << std::scientific << dt << "\n\n";
                } else if (error_counter > 9) {
                    dt = dt / time_factor;
                    std::ofstream f2(filename, std::ios::app);
                    std::cout << "\nReducing time step to: " << std::scientific << dt << "\n\n";
                    f2        << "\nReducing time step to: " << std::scientific << dt << "\n\n";
                }
            }

            output_step_info();
        }

        file.close();
    } // while t < T
}

// problem.cpp
Eigen::VectorXd problem_coupled::solve_dx_(Eigen::SparseMatrix<double>& Ktot,
                                           const Eigen::VectorXd& R,
                                           bool verbose) {

    //Timer t("Time solving the system");
    auto K = Ktot;
    //if (!K.isCompressed()) K.makeCompressed();
    const Eigen::VectorXd b = -R;

    if (verbose) {
        std::cout << " using the following solver: " << Solver << std::endl;
    }

    Eigen::VectorXd dx;

    if (Solver == "SparseLU" || Solver == "slu") {
        //Timer t("time SLu");
        Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
        slu.analyzePattern(K);
        slu.factorize(K);
        dx = slu.solve(b);
        if (verbose || slu.info() != Eigen::Success) {
            //std::cerr << "[solve_dx_] SparseLU info=" << int(slu.info()) << "\n";
        }
    } else if (Solver == "LSQCG" || Solver == "LeastSquaresConjugateGradient") {
        using Sp = Eigen::SparseMatrix<double>;
        using Vec = Eigen::VectorXd;

        //static Vec dx_prev;                 // keep previous solution to warm-start
        if (dx_prev.size() != b.size())     // resize on first call / size change
            dx_prev = Vec::Zero(b.size());
        Eigen::LeastSquaresConjugateGradient<Sp, Eigen::IncompleteCholesky<double>> lsq;
        lsq.setTolerance(1e-10);            // tune; or make it relative to ||b||
        lsq.setMaxIterations(2 * K.cols()); // don’t let it spin forever
        lsq.compute(K);                     // recompute each time since entries change

        auto dx = lsq.solveWithGuess(b, dx_prev);
        dx_prev = dx;
    } else {
        if (verbose) {
            std::cerr << "[solve_dx_] Unknown Solver='" << Solver
                      << "', using BiCGSTAB.\n";
        }
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> bicg;
        bicg.compute(K);
        dx = bicg.solve(b);
        if (verbose || bicg.info() != Eigen::Success) {
            std::cerr << "[solve_dx_] BiCGSTAB info=" << int(bicg.info())
                      << " iters=" << bicg.iterations()
                      << " err=" << (K * dx - b).norm() << "\n";
        }
    }
    return dx;
}





void printSparseTriplets(const Eigen::SparseMatrix<double>& Ain,
                         std::ostream& os,
                         bool one_based)
{
    Eigen::SparseMatrix<double> A = Ain;
    A.makeCompressed();

    os << std::setprecision(17);
    for (int outer = 0; outer < A.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, outer); it; ++it) {
            int r = it.row();
            int c = it.col();
            if (one_based) { ++r; ++c; }
            os << r << " " << c << " " << it.value() << "\n";
        }
    }
    std::cout<<"print successful" << "\n";
}

void writeTripletsToFile(const Eigen::SparseMatrix<double>& A,
                         const std::string& path,
                         bool one_based)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    printSparseTriplets(A, f, one_based);
    f.close();
}



