//
// Created by Maitreya Limkar on 30-08-2025.
//

#ifndef COUPLED_FIELD_PROBLEM_2_POSTPROCESS_HPP
#define COUPLED_FIELD_PROBLEM_2_POSTPROCESS_HPP

#include <string>
#include <vector>
#include "../node/node.hpp"
#include "../element/element.hpp"

// Forward declarations; these are your project types.
struct Node;
struct element;

namespace vtkpp {

    struct Options {
        bool add_vertex_cells = true;   // also emit 1-pt cells so nodes can render as points
    };

    /// Write one legacy ASCII .vtk frame (unstructured grid).
    /// Expects:
    ///   - NL[i].X(j) : reference coordinates (j = 0..PD-1)
    ///   - NL[i].u(k) : c at k=0, v at k=1..PD, optional p at k=1+PD
    ///   - EL[e].NdL1(a) : 1-based connectivity (a = 0..NPE-1)
    ///   - EL[e].NPE1    : nodes per element (NPE)
    void write_step_vtk(const std::string& filepath,
                        const std::vector<Node>& NL,
                        const std::vector<element>& EL,
                        int PD, int step, double time,
                        const Options& opt = {});

    /// Simple .pvd series writer so ParaView groups frames automatically.
    class PvdSeries {
    public:
        explicit PvdSeries(const std::string& pvdPath);
        void add(double time, const std::string& relativeVtkFile); // call once per frame
        void close();                                              // call once at the end
        ~PvdSeries();
    private:
        std::string path_;
        bool open_ = false;
        void ensure_open_();
    };

} // namespace vtkpp

#endif //COUPLED_FIELD_PROBLEM_2_POSTPROCESS_HPP