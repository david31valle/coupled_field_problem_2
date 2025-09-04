#pragma once
#include <vector>
#include <string>

struct Node;     // expected fields: int PD; Eigen::VectorXd x,u,field,GP_vals;
struct element;  // not used here but kept for signature parity

namespace vtkio {

    /** Dimension-aware dispatcher (calls 1D/2D/3D based on NL.front().PD) */
    void Post_Process(const std::vector<Node>& NL,
                      const std::vector<element>& EL,
                      int step,
                      const std::string& out_dir = "vtk",
                      bool write_gp = true);

    /** 1D: writes c & v legacy POLYDATA (scatter + polyline) */
    void Post_Process_1D(const std::vector<Node>& NL,
                         const std::vector<element>& EL,
                         int step,
                         const std::string& out_dir = "vtk",
                         bool write_gp = true);

    /** 2D: writes six POLYDATA files (c/vx/vy at z=0 and z=scalar) */
    void Post_Process_2D(const std::vector<Node>& NL,
                         const std::vector<element>& EL,
                         int step,
                         const std::string& out_dir = "vtk",
                         bool write_gp = true);

    /** 3D: writes four POLYDATA files (c, vx, vy, vz) with MATLAB-like filtering */
    void Post_Process_3D(const std::vector<Node>& NL,
                         const std::vector<element>& EL,
                         int step,
                         const std::string& out_dir = "vtk",
                         bool write_gp = true);

} // namespace vtkio