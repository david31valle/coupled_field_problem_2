#include "problem.hpp"

#include <fstream>
#include <iomanip>
#include <set>
#include <array>
#include <sstream>

// fwd
void writeTripletsToFile(const Eigen::SparseMatrix<double>& A,
                         const std::string& path,
                         bool one_based = true);

class Timer {
public:
    Timer(const std::string& name = "Timer")
        : name(name), start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << " took " << ms << " ms\n";
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

    for (int i = 0; i < H.rows; ++i) if (row_nnz[i] == 0) {
        ++H.zero_rows; if (H.zero_rows <= print_limit && verbose) std::cerr << "zero row " << i << "\n";
    }
    for (int j = 0; j < H.cols; ++j) if (col_nnz[j] == 0) {
        ++H.zero_cols; if (H.zero_cols <= print_limit && verbose) std::cerr << "zero col " << j << "\n";
    }

    for (int i = 0; i < std::min(H.rows, H.cols); ++i) {
        if (A.coeff(i, i) == 0.0) {
            ++H.zero_diagonal;
            if (H.zero_diagonal <= print_limit && verbose) std::cerr << "zero diagonal at " << i << "\n";
        }
    }

    if (Rtot && Rtot->size() == H.rows) {
        for (int i = 0; i < H.rows; ++i) {
            if (row_nnz[i] == 0 && std::abs((*Rtot)(i)) > tol) {
                ++H.inconsistent_zero_rows;
                if (H.inconsistent_zero_rows <= print_limit && verbose)
                    std::cerr << "zero row " << i << " but R(" << i << ") = " << (*Rtot)(i) << "\n";
            }
        }
    }

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
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(H.cols);
        H.ones_resid = (A * ones).norm();
        std::cerr << "||K*1|| = " << H.ones_resid << "\n";
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
    // filename
    std::ostringstream oss;
    oss << PD << "D_Normal_" << EL.size() << "_EL=[";
    for (int eo : element_order) oss << eo;
    oss << "]_" << Initial_density << "_" << time_incr_method << "_" << BC_type << "_[";
    for (size_t i = 0; i < parameters.size(); ++i) {
        oss << parameters[i];
        if (i + 1 != parameters.size()) oss << ",";
    }
    oss << "].txt";
    filename = oss.str();

    // BCs
    Assign_BC(Corners);

    // GP DOFs
    if (GP_vals == "On") Assign_GP_DOFs();

    // info + run
    problem_info();
    solve();
}

void problem_coupled::Assign_BC(const std::string Corners) {
    const int NoNs = static_cast<int>(Node_List.size());
    const double tol = 1e-6;

    const double W = static_cast<double>(domain_size);
    const double H = W, D = W;

    if (BC_type == "DBC") {
        Assign_DOF_DBC();
        return;
    }

    // PBC
    std::vector<int> NLC, NLS, NLM, NLP;

    if (PD == 1) {
        for (int i = 0; i < NoNs; ++i) {
            const double x = Node_List[i].X(0);
            if (std::abs(x - 0.0) < tol || std::abs(x - W) < tol) NLC.push_back(i);
            else                                                  NLS.push_back(i);
        }
    } else if (PD == 2) {
        std::vector<int> MEX, PEX, MEY, PEY;
        for (int i = 0; i < NoNs; ++i) {
            const auto& X = Node_List[i].X;
            const double x = X(0), y = X(1);
            const bool onLeft   = std::abs(x - 0.0) < tol;
            const bool onRight  = std::abs(x - W)   < tol;
            const bool onBottom = std::abs(y - 0.0) < tol;
            const bool onTop    = std::abs(y - H)   < tol;
            const bool isCorner =
                (onLeft && onBottom) || (onRight && onBottom) ||
                (onRight && onTop)   || (onLeft  && onTop);
            if      (isCorner) NLC.push_back(i);
            else if (onLeft)   MEY.push_back(i);
            else if (onRight)  PEY.push_back(i);
            else if (onBottom) MEX.push_back(i);
            else if (onTop)    PEX.push_back(i);
            else               NLS.push_back(i);
        }
        for (int pi : PEX) for (int mi : MEX)
            if (std::abs(Node_List[pi].X(0) - Node_List[mi].X(0)) < tol) { NLP.push_back(pi); NLM.push_back(mi); }
        for (int pi : PEY) for (int mi : MEY)
            if (std::abs(Node_List[pi].X(1) - Node_List[mi].X(1)) < tol) { NLP.push_back(pi); NLM.push_back(mi); }
    } else if (PD == 3) {
        std::vector<int> MEX, PEX, MEY, PEY, MEZ, PEZ;
        std::vector<int> MFX, PFX, MFY, PFY, MFZ, PFZ;
        auto on = [&](double a, double b){ return std::abs(a-b) < tol; };

        for (int i = 0; i < NoNs; ++i) {
            const auto& Xv = Node_List[i].X;
            const double x = Xv(0), y = Xv(1), z = Xv(2);
            if ((on(x,0)&&on(y,0)&&on(z,0)) || (on(x,0)&&on(y,0)&&on(z,D)) ||
                (on(x,0)&&on(y,H)&&on(z,0)) || (on(x,0)&&on(y,H)&&on(z,D)) ||
                (on(x,W)&&on(y,0)&&on(z,0)) || (on(x,W)&&on(y,0)&&on(z,D)) ||
                (on(x,W)&&on(y,H)&&on(z,0)) || (on(x,W)&&on(y,H)&&on(z,D))) {
                NLC.push_back(i);
            } else if (on(x,0)) {
                if (on(y,0))      MEZ.push_back(i);
                else if (on(y,H)) PEZ.push_back(i);
                else if (on(z,0)) MEY.push_back(i);
                else if (on(z,D)) PEY.push_back(i);
                else              MFX.push_back(i);
            } else if (on(y,0)) {
                if (on(x,W))      PEZ.push_back(i);
                else if (on(z,0)) MEX.push_back(i);
                else if (on(z,D)) PEX.push_back(i);
                else              MFY.push_back(i);
            } else if (on(z,0)) {
                if (on(x,W))      PEY.push_back(i);
                else if (on(y,H)) PEX.push_back(i);
                else              MFZ.push_back(i);
            } else if (on(x,W)) {
                if (on(y,H))      PEZ.push_back(i);
                else if (on(z,D)) PEY.push_back(i);
                else              PFX.push_back(i);
            } else if (on(y,H)) {
                if (on(z,D))      PEX.push_back(i);
                else              PFY.push_back(i);
            } else if (on(z,D)) {
                PFZ.push_back(i);
            } else {
                NLS.push_back(i);
            }
        }
        for (int pi : PEX) for (int mi : MEX)
            if (on(Node_List[pi].X(0), Node_List[mi].X(0))) { NLM.push_back(mi); NLP.push_back(pi); }
        for (int pi : PEY) for (int mi : MEY)
            if (on(Node_List[pi].X(1), Node_List[mi].X(1))) { NLM.push_back(mi); NLP.push_back(pi); }
        for (int pi : PEZ) for (int mi : MEZ)
            if (on(Node_List[pi].X(2), Node_List[mi].X(2))) { NLM.push_back(mi); NLP.push_back(pi); }
        for (int pi : PFX) for (int mi : MFX)
            if (on(Node_List[pi].X(1), Node_List[mi].X(1)) && on(Node_List[pi].X(2), Node_List[mi].X(2))) { NLM.push_back(mi); NLP.push_back(pi); }
        for (int pi : PFY) for (int mi : MFY)
            if (on(Node_List[pi].X(0), Node_List[mi].X(0)) && on(Node_List[pi].X(2), Node_List[mi].X(2))) { NLM.push_back(mi); NLP.push_back(pi); }
        for (int pi : PFZ) for (int mi : MFZ)
            if (on(Node_List[pi].X(0), Node_List[mi].X(0)) && on(Node_List[pi].X(1), Node_List[mi].X(1))) { NLM.push_back(mi); NLP.push_back(pi); }
    }

    // periodic copies
    if (!NLC.empty()) {
        for (int i : NLC) {
            Node_List[i].U  = Node_List[NLC[0]].U;
            Node_List[i].u  = Node_List[NLC[0]].u;
            Node_List[i].un = Node_List[NLC[0]].un;
        }
    }
    for (size_t i = 0; i < NLP.size(); ++i) {
        Node_List[NLP[i]].U  = Node_List[NLM[i]].U;
        Node_List[NLP[i]].u  = Node_List[NLM[i]].u;
        Node_List[NLP[i]].un = Node_List[NLM[i]].un;
    }

    if (Corners == "Fixed") {
        for (int i : NLC) {
            Node_List[i].U.segment(1, PD).setZero();
            Node_List[i].u.segment(1, PD).setZero();
            Node_List[i].un.segment(1, PD).setZero();
            Node_List[i].BC.segment(1, PD).setZero();
        }
    }

    Assign_DOF_PBC(NLC, NLS, NLM, NLP);
}

void problem_coupled::Assign_DOF_DBC() {
    const int NoNs = static_cast<int>(Node_List.size());
    int dofs = 0;
    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd BC  = Node_List[i].BC;
        Eigen::VectorXd DOF = Node_List[i].DOF;
        for (int p = 0; p < BC.size(); ++p) {
            if (BC(p) == 1.0) {
                ++dofs;
                DOF(p) = dofs;
            }
        }
        Node_List[i].DOF = DOF;
    }
    DOFs = dofs;
}

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

    if (NoCNs != 0) {
        const int firstCorner = NLC[0];
        auto& DOF0 = Node_List[firstCorner].DOF;
        const auto& BC0 = Node_List[firstCorner].BC;
        for (Eigen::Index p = 0; p < BC0.size(); ++p) if (BC0(p) == 1) { ++DOFs; DOF0(p) = static_cast<double>(DOFs); }
        for (int idx : NLC) Node_List[idx].DOF = DOF0;
    }

    for (int idx : NLS) {
        auto& DOF = Node_List[idx].DOF;
        const auto& BC = Node_List[idx].BC;
        for (Eigen::Index p = 0; p < BC.size(); ++p) if (BC(p) == 1) { ++DOFs; DOF(p) = static_cast<double>(DOFs); }
    }

    for (int idx : NLM) {
        auto& DOF = Node_List[idx].DOF;
        const auto& BC = Node_List[idx].BC;
        for (Eigen::Index p = 0; p < BC.size(); ++p) if (BC(p) == 1) { ++DOFs; DOF(p) = static_cast<double>(DOFs); }
    }

    const int nCopy = std::min(NoPNs, NoMNs);
    for (int i = 0; i < nCopy; ++i) {
        int idxP = NLP[i], idxM = NLM[i];
        Node_List[idxP].BC  = Node_List[idxM].BC;
        Node_List[idxP].DOF = Node_List[idxM].DOF;
    }
}

void problem_coupled::Assign_GP_DOFs() {
    int GP_dofs = 0;
    const int NoNs = static_cast<int>(Node_List.size());
    for (int i = 0; i < NoNs; ++i) {
        Eigen::VectorXd& BC  = Node_List[i].GP_BC;
        Eigen::VectorXd& DOF = Node_List[i].GP_DOF;
        if ((BC.array() == 1).any()) {
            ++GP_dofs;
            DOF = Eigen::VectorXd::Constant(BC.size(), GP_dofs);
        }
    }
    GP_DOFs = GP_dofs;
}

void problem_coupled::update(const Eigen::VectorXd& dx) {
    const int NoNs = static_cast<int>(Node_List.size());
    const int NoEs = static_cast<int>(Element_List.size());

    for (int i = 0; i < NoNs; ++i) {
        auto& BC  = Node_List[i].BC;
        auto& DOF = Node_List[i].DOF;
        auto& u   = Node_List[i].u;
        for (int p = 0; p < BC.size(); ++p) if (BC(p) == 1) {
            int g = static_cast<int>(DOF(p)) - 1;
            if (0 <= g && g < dx.size()) u(p) += dx(g);
        }
    }

    for (int e = 0; e < NoEs; ++e) {
        auto& elem = Element_List[e];
        const int NPE1 = elem.NPE1;
        const int NPE2 = elem.NPE2;

        Eigen::VectorXd c(NPE1);
        for (int i = 0; i < NPE1; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL1(i)) - 1;
            c(i) = Node_List[nodeIdx].u(0);
        }
        elem.c = c;

        const int dim = PD;
        const int first0 = 1;
        Eigen::MatrixXd v(PD, NPE2);
        for (int i = 0; i < NPE2; ++i) {
            int nodeIdx = static_cast<int>(elem.NdL2(i)) - 1;
            v.col(i) = Node_List[nodeIdx].u.segment(first0, dim);
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
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());
    for (int e = 0; e < NoEs; ++e) { Element_List[e].cn = Element_List[e].c; Element_List[e].vn = Element_List[e].v; }
    for (int i = 0; i < NoNs; ++i)   Node_List[i].un = Node_List[i].u;
}

void problem_coupled::downdate_time() {
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());
    for (int e = 0; e < NoEs; ++e) { Element_List[e].c = Element_List[e].cn; Element_List[e].v = Element_List[e].vn; }
    for (int i = 0; i < NoNs; ++i)   Node_List[i].u = Node_List[i].un;
}

Eigen::VectorXd problem_coupled::Residual(double dt)
{
    const int NoEs  = static_cast<int>(Element_List.size());
    if (NoEs == 0) return Eigen::VectorXd::Zero(DOFs);

    const int NPE1 = Element_List[0].NPE1;
    const int NPE2 = Element_List[0].NPE2;

    Eigen::VectorXd Rtot = Eigen::VectorXd::Zero(DOFs);

    for (int e = 0; e < NoEs; ++e) {
        auto [R1, R2] = Element_List[e].Residual(dt);

        // R1 (scalar)
        for (int i = 0; i < NPE1; ++i) {
            const int nodeIdx = static_cast<int>(Element_List[e].NdL1(i)) - 1;
            const auto& BC  = Node_List[nodeIdx].BC;
            const auto& DOF = Node_List[nodeIdx].DOF;
            if (BC(0) == 1) {
                const int g = static_cast<int>(DOF(0)) - 1;
                if (0 <= g && g < Rtot.size()) Rtot(g) += R1(i);
            }
        }

        // R2 (velocity)
        for (int i = 0; i < NPE2; ++i) {
            const int nodeIdx = static_cast<int>(Element_List[e].NdL2(i)) - 1;
            const auto& BC  = Node_List[nodeIdx].BC;
            const auto& DOF = Node_List[nodeIdx].DOF;
            for (int p = 1; p <= PD; ++p) if (BC(p) == 1) {
                const int g  = static_cast<int>(DOF(p)) - 1;
                const int li = i * PD + (p - 1);
                if (0 <= g && g < Rtot.size()) Rtot(g) += R2(li);
            }
        }
    }

    return Rtot;
}

void problem_coupled::assemble(double dt) {
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());
    if (NoEs == 0 || NoNs == 0) {
        Rtot = Eigen::VectorXd::Zero(DOFs);
        Ktot = Eigen::SparseMatrix<double>(DOFs, DOFs);
        return;
    }

    const int NPE1 = Element_List[0].NPE1;
    const int NPE2 = Element_List[0].NPE2;

    Rtot = Eigen::VectorXd::Zero(DOFs);
    std::vector<Eigen::Triplet<double>> triplets;


    for (int e = 0; e < NoEs; ++e) {
        auto [R1, R2, K11, K12, K21, K22] = Element_List[e].RK(dt);

        struct Slot { int g; int loc; };
        std::array<Slot, 64> rows_scalar;  int n_rows_scalar = 0;
        std::array<Slot, 64> cols_scalar;  int n_cols_scalar = 0;
        std::vector<Slot> rows_vel; rows_vel.reserve(NPE2 * PD);
        std::vector<Slot> cols_vel; cols_vel.reserve(NPE2 * PD);

        // R1 + scalar rows
        for (int i = 0; i < NPE1; ++i) {
            const int node = static_cast<int>(Element_List[e].NdL1(i)) - 1;
            const auto& BC = Node_List[node].BC;
            if (BC(0) != 1) continue;
            const auto& DOF = Node_List[node].DOF;
            const int g  = static_cast<int>(DOF(0)) - 1;
            rows_scalar[n_rows_scalar++] = {g, i};
            Rtot(g) += R1(i);
        }

        // scalar cols
        for (int j = 0; j < NPE1; ++j) {
            const int node = static_cast<int>(Element_List[e].NdL1(j)) - 1;
            const auto& BC = Node_List[node].BC;
            if (BC(0) != 1) continue;
            const auto& DOF = Node_List[node].DOF;
            const int g  = static_cast<int>(DOF(0)) - 1;
            cols_scalar[n_cols_scalar++] = {g, j};
        }

        // R2 + vel rows
        for (int i = 0; i < NPE2; ++i) {
            const int node = static_cast<int>(Element_List[e].NdL2(i)) - 1;
            const auto& BC  = Node_List[node].BC;
            const auto& DOF = Node_List[node].DOF;
            for (int p = 1; p <= PD; ++p) if (BC(p) == 1) {
                const int g  = static_cast<int>(DOF(p)) - 1;
                const int li = i * PD + (p - 1);
                rows_vel.push_back({g, li});
                Rtot(g) += R2(li);
            }
        }

        // vel cols
        for (int j = 0; j < NPE2; ++j) {
            const int node = static_cast<int>(Element_List[e].NdL2(j)) - 1;
            const auto& BC  = Node_List[node].BC;
            const auto& DOF = Node_List[node].DOF;
            for (int q = 1; q <= PD; ++q) if (BC(q) == 1) {
                const int g  = static_cast<int>(DOF(q)) - 1;
                const int lj = j * PD + (q - 1);
                cols_vel.push_back({g, lj});
            }
        }

        // reserve
        triplets.reserve(triplets.size() +
                         n_rows_scalar*n_cols_scalar +
                         n_rows_scalar*static_cast<int>(cols_vel.size()) +
                         static_cast<int>(rows_vel.size())*n_cols_scalar +
                         static_cast<int>(rows_vel.size())*static_cast<int>(cols_vel.size()));

        // K11

        for (int r = 0; r < n_rows_scalar; ++r)
            for (int c = 0; c < n_cols_scalar; ++c)
                triplets.emplace_back(rows_scalar[r].g, cols_scalar[c].g,
                                      K11(rows_scalar[r].loc, cols_scalar[c].loc));
        // K12

        for (int r = 0; r < n_rows_scalar; ++r)
            for (const auto& cv : cols_vel)
                triplets.emplace_back(rows_scalar[r].g, cv.g, K12(rows_scalar[r].loc, cv.loc));
        // K21

        for (const auto& rv : rows_vel)
            for (int c = 0; c < n_cols_scalar; ++c)
                triplets.emplace_back(rv.g, cols_scalar[c].g, K21(rv.loc, cols_scalar[c].loc));
        // K22

        for (const auto& rv : rows_vel)
            for (const auto& cv : cols_vel)
                triplets.emplace_back(rv.g, cv.g, K22(rv.loc, cv.loc));
    }

    Ktot.resize(DOFs, DOFs);
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
    t = 0.0;
    counter = 1;

    while (t < T) {
        std::ofstream file(filename, std::ios::app);

        int  error_counter = 1;
        bool isNotAccurate = true;
        bool try_again     = false;
        double Norm_R0     = 1.0;   // per Newton loop, set on first corrector

        // --- Armijo-style backtracking line-search ---
        auto try_damped_update = [&](const Eigen::VectorXd& dx,
                                     double R0_norm,
                                     double c1 = 1e-4,
                                     double rho = 0.5,
                                     int max_bt = 6) -> std::pair<bool,double>
        {
            // Save nodal u (so we can revert between trials)
            const int NoNs = static_cast<int>(Node_List.size());
            std::vector<Eigen::VectorXd> u_save(NoNs);
            for (int i = 0; i < NoNs; ++i) u_save[i] = Node_List[i].u;

            double alpha = 1.0;
            Eigen::VectorXd scaled = dx;

            for (int k = 0; k <= max_bt; ++k) {
                // restore u to saved state then apply alpha*dx
                for (int i = 0; i < NoNs; ++i) Node_List[i].u = u_save[i];
                if (alpha != 1.0) scaled = alpha * dx;
                update(scaled);

                // evaluate residual at trial state
                Eigen::VectorXd Rnew = Residual(dt);
                const double Rn = Rnew.norm();

                // accept if sufficient decrease
                if (Rn <= (1.0 - c1 * alpha) * R0_norm) {
                    Rtot = std::move(Rnew);   // keep accepted residual
                    return {true, Rn};
                }

                alpha *= rho; // backtrack
            }

            // failed → revert u
            for (int i = 0; i < NoNs; ++i) Node_List[i].u = u_save[i];
            return {false, R0_norm};
        };

        // --- Predictor: warm-start current step from previous accepted state ---
        // Robust default (no extra history needed): u^{n+1,0} := u^n
        if (counter > 1) {
            for (auto& nd : Node_List) nd.u = nd.un;
        }

        while (isNotAccurate) {
            // Residual + Tangent
            assemble(dt);
            auto H = check_matrix_health(Ktot);
            // Let the linear solver auto-decide (LDLT vs LU vs iterative).
            Solver = "Auto";

            // Initial output & predictor residual
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
                          << std::scientific << Norm_R0
                          << " , normalized : 1\n";
                file << "Residual Norm at Predictor               : "
                     << std::scientific << Norm_R0
                     << " , normalized : 1\n";
            }

            // Solve for Newton step
            Eigen::VectorXd dx = solve_dx_(Ktot, Rtot, false);

            // Line-search (Armijo) for a safe step size
            auto [ok, Rn] = try_damped_update(dx, Norm_R0);
            if (!ok) {
                isNotAccurate = false;
                try_again = true;
                std::cout << "Line search failed; will reduce dt.\n";
                file      << "Line search failed; will reduce dt.\n";
                break; // exit Newton loop; time step will shrink
            }

            std::cout << "Residual Norm @ Increment " << error_counter
                      << " at Corrector : " << std::scientific << Rn
                      << " , normalized : " << (Norm_R0 > 0 ? Rn/Norm_R0 : 0.0) << "\n";
            file << "Residual Norm @ Increment " << error_counter
                 << " at Corrector : " << std::scientific << Rn
                 << " , normalized : " << (Norm_R0 > 0 ? Rn/Norm_R0 : 0.0) << "\n";

            if (Rn < tol) {
                isNotAccurate = false;
            } else if (error_counter > max_iter || Rn > 1e6) {
                isNotAccurate = false;
                std::cout << "Convergence is not obtained!\n";
                file << "Convergence is not obtained!\n";
                try_again = true;
            }

            ++error_counter;
        } // Newton

        // Early stop if velocities ~ 0
        {
            Eigen::MatrixXd velocity = Get_all_velocity();
            if (velocity.cwiseAbs().maxCoeff() < 1e-6) {
                post_process();
                break;
            }
        }

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
            // accept step
            t += dt;
            ++counter;

            if (GP_vals == "On" && GP_DOFs > 0) {
                assemble_GP(dt);
                const int NGP_val = static_cast<int>(Node_List[0].GP_BC.size());
                Eigen::MatrixXd dx_GP_mat(GP_DOFs, NGP_val);
                for (int s = 0; s < NGP_val; ++s) {
                    dx_GP_mat.col(s) = solve_dx_(Ktot_GP, Rtot_GP.col(s), false);
                }
                update_GP(dx_GP_mat);
            }

            post_process();
            update_time();

            if (time_incr_method == "Adaptive") {
                if (error_counter < 5) {
                    dt = dt * time_factor;
                    std::ofstream f2(filename, std::ios::app);
                    std::cout << "\nIncreasing time step to: " << std::scientific << dt << "\n\n";
                    f2        << "\nIncreasing time step to: " << std::scientific << dt << "\n\n";
                } else if (error_counter > 8) {
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

Eigen::VectorXd problem_coupled::solve_dx_(Eigen::SparseMatrix<double>& Ktot,
                                           const Eigen::VectorXd& R,
                                           bool verbose)
{
    // Ensure a clean, compressed matrix for factorization
    if (!Ktot.isCompressed()) Ktot.makeCompressed();

    const Eigen::VectorXd b = -R;
    Eigen::VectorXd dx;

    // Helper to measure relative asymmetry quickly (norm of K-K^T over norm of K)
    auto rel_asym = [](const Eigen::SparseMatrix<double>& A) -> double {
        // Guard against zero matrix
        const double nA = A.norm();
        if (nA == 0.0) return 0.0;
        Eigen::SparseMatrix<double> AT = A.transpose();
        Eigen::SparseMatrix<double> Skew = A - AT;
        return Skew.norm() / nA;
    };

    // Helper to log failures when needed
    auto log_fail = [&](const char* tag, const char* extra = nullptr){
        if (verbose) {
            std::cerr << "[solve_dx_] " << tag << " failed";
            if (extra) std::cerr << " (" << extra << ")";
            std::cerr << "\n";
        }
    };

    // Manual override: honor explicit solver choices if set
    if (Solver == "SparseLU" || Solver == "slu") {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
        slu.analyzePattern(Ktot);
        slu.factorize(Ktot);
        dx = slu.solve(b);
        if (verbose || slu.info() != Eigen::Success)
            std::cerr << "[solve_dx_] SparseLU info=" << int(slu.info()) << "\n";
        return dx;
    }

    if (Solver == "LSQCG" || Solver == "LeastSquaresConjugateGradient") {
        using Sp  = Eigen::SparseMatrix<double>;
        using Vec = Eigen::VectorXd;
        if (dx_prev.size() != b.size()) dx_prev = Vec::Zero(b.size());
        Eigen::LeastSquaresConjugateGradient<Sp, Eigen::IncompleteCholesky<double>> lsq;
        lsq.setTolerance(1e-10);
        lsq.setMaxIterations(2 * Ktot.cols());
        lsq.compute(Ktot);
        dx_prev = lsq.solveWithGuess(b, dx_prev);
        return dx_prev;
    }

    if (Solver == "BiCGSTAB") {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> it;
        it.setTolerance(1e-10);
        it.setMaxIterations(2 * Ktot.cols());
        it.compute(Ktot);
        dx = it.solve(b);
        if (verbose || it.info() != Eigen::Success) {
            std::cerr << "[solve_dx_] BiCGSTAB info=" << int(it.info())
                      << " iters=" << it.iterations()
                      << " err="   << (Ktot * dx - b).norm() << "\n";
        }
        return dx;
    }

    // === AUTO mode ===
    // 1) Check near-symmetry; if symmetric, try SimplicialLDLT (fast SPD solve)
    const double asym = rel_asym(Ktot);
    const bool nearlySym = (asym < 1e-12); // adjust threshold if needed

    if (nearlySym) {
        // Try SPD factorization (works for SPD; will fail if indefinite)
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        ldlt.analyzePattern(Ktot);
        ldlt.factorize(Ktot);
        if (ldlt.info() == Eigen::Success) {
            dx = ldlt.solve(b);
            if (verbose) {
                std::cout << "[solve_dx_] Auto -> SimplicialLDLT (asym=" << asym << ")\n";
            }
            return dx;
        } else {
            log_fail("SimplicialLDLT");
            // Fall back to a robust general solver
            Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
            slu.analyzePattern(Ktot);
            slu.factorize(Ktot);
            if (slu.info() == Eigen::Success) {
                dx = slu.solve(b);
                if (verbose) {
                    std::cout << "[solve_dx_] Auto fallback -> SparseLU (asym=" << asym << ")\n";
                }
                return dx;
            } else {
                log_fail("SparseLU", "after LDLT fail");
                // Last resort: iterative
                Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> it;
                it.setTolerance(1e-10);
                it.setMaxIterations(2 * Ktot.cols());
                it.compute(Ktot);
                dx = it.solve(b);
                if (verbose || it.info() != Eigen::Success) {
                    std::cerr << "[solve_dx_] BiCGSTAB info=" << int(it.info())
                              << " iters=" << it.iterations()
                              << " err="   << (Ktot * dx - b).norm() << "\n";
                }
                return dx;
            }
        }
    } else {
        // Non-symmetric: go to SparseLU first
        Eigen::SparseLU<Eigen::SparseMatrix<double>> slu;
        slu.analyzePattern(Ktot);
        slu.factorize(Ktot);
        if (slu.info() == Eigen::Success) {
            dx = slu.solve(b);
            if (verbose) {
                std::cout << "[solve_dx_] Auto -> SparseLU (asym=" << asym << ")\n";
            }
            return dx;
        } else {
            log_fail("SparseLU", "nonsymmetric");
            // Optional try: SparseQR (sometimes faster/robuster than LU on rectangular/ill-cond)
            // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
            // qr.compute(Ktot);
            // if (qr.info() == Eigen::Success) { dx = qr.solve(b); return dx; }

            // Iterative fallback
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> it;
            it.setTolerance(1e-10);
            it.setMaxIterations(2 * Ktot.cols());
            it.compute(Ktot);
            dx = it.solve(b);
            if (verbose || it.info() != Eigen::Success) {
                std::cerr << "[solve_dx_] BiCGSTAB info=" << int(it.info())
                          << " iters=" << it.iterations()
                          << " err="   << (Ktot * dx - b).norm() << "\n";
            }
            return dx;
        }
    }
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
    std::cout << "print successful\n";
}


// === Problem info: prints to console and writes a header to the output file ===
void problem_coupled::problem_info() {
    const int NoEs = static_cast<int>(Element_List.size());
    const int NoNs = static_cast<int>(Node_List.size());

    std::cout << "======================================================\n";
    std::cout << "================  Problem information  ===============\n";
    std::cout << "======================================================\n";
    std::cout << "Problem dimension                   : " << PD << "\n";
    std::cout << "Time increment                      : " << std::scientific << dt << "\n";
    std::cout << "Final time                          : " << std::scientific << T  << "\n";
    std::cout << "Time increment method               : " << time_incr_method << "\n";
    std::cout << "Number of nodes                     : " << NoNs << "\n";
    std::cout << "Number of bulk elements             : " << NoEs << "\n";
    std::cout << "Number of DOFs                      : " << DOFs << "\n";
    std::cout << "Element order                       : [";
    for (size_t i = 0; i < element_order.size(); ++i) {
        std::cout << element_order[i] << (i + 1 == element_order.size() ? "" : " ");
    }
    std::cout << "]\n";
    std::cout << "E  R  xi                            : [";
    for (size_t i = 0; i < parameters.size(); ++i) {
        std::cout << parameters[i] << (i + 1 == parameters.size() ? "" : " ");
    }
    std::cout << "]\n";
    std::cout << "======================================================\n";

    std::ofstream f(filename);
    f << "======================================================\n";
    f << "================  Problem information  ===============\n";
    f << "======================================================\n";
    f << "Problem dimension                   : " << PD << "\n";
    f << "Time increment                      : " << std::scientific << dt << "\n";
    f << "Final time                          : " << std::scientific << T  << "\n";
    f << "Time increment method               : " << time_incr_method << "\n";
    f << "Number of nodes                     : " << NoNs << "\n";
    f << "Number of bulk elements             : " << NoEs << "\n";
    f << "Number of DOFs                      : " << DOFs << "\n";
    f << "Element order                       : ";
    for (size_t i = 0; i < element_order.size(); ++i) {
        f << element_order[i] << (i + 1 == element_order.size() ? "" : " ");
    }
    f << "\nE  R  xi                            : ";
    for (size_t i = 0; i < parameters.size(); ++i) {
        f << parameters[i] << (i + 1 == parameters.size() ? "" : " ");
    }
    f << "\n======================================================\n\n\n";
    f.close();
}

// === Collect all nodal velocities (columns = nodes, rows = PD components) ===
Eigen::MatrixXd problem_coupled::Get_all_velocity() {
    const int NoNs = static_cast<int>(Node_List.size());
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(PD, NoNs);
    // convention in this project: u(0) = scalar (cell), u(1..PD) = velocity
    for (int i = 0; i < NoNs; ++i) {
        if (Node_List[i].u.size() >= (1 + PD)) {
            v.col(i) = Node_List[i].u.segment(1, PD);
        } else {
            v.col(i).setZero();
        }
    }
    return v;
}

// === Post-process  ===
#include "../postprocess/postprocess.hpp"
#include <filesystem>

void problem_coupled::post_process() {
    // Minimal stub so you can compile/run. Hook your 1D/2D/3D exporters here.
    // e.g., write nodal c and v to file if you want.
    // Build "vtk/step_XXXX.vtk" (ParaView will group the series automatically)
    // MATLAB-like outputs (1D/2D/3D) → VTK POLYDATA files in ./vtk
    // Auto-detects PD from nodes and writes the matching set for this step.
    // Create configuration for post-processing
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

void writeTripletsToFile(const Eigen::SparseMatrix<double>& A,
                         const std::string& path,
                         bool one_based)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);
    printSparseTriplets(A, f, one_based);
    f.close();
}
