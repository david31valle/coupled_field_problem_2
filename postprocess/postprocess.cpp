#include "postprocess.hpp"
#include "../node/node.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace vtkio {

namespace fs = std::filesystem;

static void ensure_dir(const std::string& dir){
  if (dir.empty()) return;
  if (!fs::exists(dir)) fs::create_directories(dir);
}

/* ---------- generic VTK writer (legacy POLYDATA) ----------

Writes:
- POINTS
- VERTICES
- optional single LINES polyline (order given by 'polyline_idx')
- POINT_DATA scalar arrays (each entry: {name, data})

If gp_arrays is non-null, writes gp0, gp1, ... too.
*/
static void write_polydata(
    const std::string& path,
    const std::vector<double>& X,
    const std::vector<double>& Y,
    const std::vector<double>& Z,
    const std::vector<int>* polyline_idx, // optional (sorted indices)
    const std::vector<std::pair<std::string, std::vector<double>>>& scalars,
    const std::vector<std::vector<double>>* gp_arrays = nullptr)
{
  const int N = (int)X.size();
  if (N == 0) return;

  std::ofstream f(path);
  if (!f) throw std::runtime_error("Cannot open " + path);

  f << "# vtk DataFile Version 3.0\n";
  f << "postprocess\nASCII\nDATASET POLYDATA\n";

  // POINTS
  f << "POINTS " << N << " float\n";
  f << std::setprecision(9);
  for (int i=0;i<N;++i) f << X[i] << " " << Y[i] << " " << Z[i] << "\n";

  // VERTICES for scatter
  f << "VERTICES " << N << " " << (N*2) << "\n";
  for (int i=0;i<N;++i) f << 1 << " " << i << "\n";

  // optional single polyline
  if (polyline_idx && !polyline_idx->empty()){
    f << "LINES 1 " << (int(polyline_idx->size()) + 1) << "\n";
    f << polyline_idx->size();
    for (int id : *polyline_idx) f << " " << id;
    f << "\n";
  } else {
    f << "LINES 0 0\n"; // keep reader happy
  }

  // Scalars
  f << "POINT_DATA " << N << "\n";
  auto dump_scalar = [&](const std::string& name, const std::vector<double>& a){
    if ((int)a.size()!=N) return;
    f << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
    for (int i=0;i<N;++i) f << a[i] << "\n";
  };
  for (const auto& s : scalars) dump_scalar(s.first, s.second);

  // gp arrays
  if (gp_arrays){
    int maxc = 0;
    for (auto& v : *gp_arrays) maxc = std::max<int>(maxc, (int)v.size());
    for (int c=0;c<maxc;++c){
      f << "SCALARS gp" << c << " float 1\nLOOKUP_TABLE default\n";
      for (int i=0;i<N;++i){
        if ((int)(*gp_arrays)[i].size() > c) f << (*gp_arrays)[i][c] << "\n";
        else f << 0.0 << "\n";
      }
    }
  }
}

/* ===================== 1D ===================== */

void Post_Process_1D(const std::vector<Node>& NL,
                     const std::vector<element>& /*EL*/,
                     int step,
                     const std::string& out_dir,
                     bool write_gp)
{
  if (NL.empty()) return;
  ensure_dir(out_dir);

  // Collect c-nodes
  std::vector<double> Xc, Yc, Zc, C;
  std::vector<std::vector<double>> GPc;
  // Collect v-nodes
  std::vector<double> Xv, Yv, Zv, V;
  const double Zplane = 100.0;

  for (const auto& n : NL){
    if (n.field.size() >= 1 && n.field(0) == 1){
      Xc.push_back(n.x(0)); Yc.push_back(0.0); Zc.push_back(Zplane);
      C.push_back(n.u(0));
      if (write_gp && n.GP_vals.size()>0){
        std::vector<double> gp(n.GP_vals.size());
        for (int i=0;i<n.GP_vals.size();++i) gp[i] = n.GP_vals(i);
        GPc.push_back(std::move(gp));
      } else GPc.emplace_back();
    }
    if (n.field.size() >= 2 && n.field(1) == 1){
      Xv.push_back(n.x(0)); Yv.push_back(0.0); Zv.push_back(Zplane);
      V.push_back(n.u.size()>=2 ? n.u(1) : 0.0); // x-component (1D)
    }
  }

  // polyline order = sort by x
  std::vector<int> order_c(Xc.size()); for (int i=0;i<(int)order_c.size();++i) order_c[i]=i;
  std::sort(order_c.begin(), order_c.end(), [&](int a,int b){ return Xc[a] < Xc[b]; });
  std::vector<int> order_v(Xv.size()); for (int i=0;i<(int)order_v.size();++i) order_v[i]=i;
  std::sort(order_v.begin(), order_v.end(), [&](int a,int b){ return Xv[a] < Xv[b]; });

  // c file (with gp arrays)
  {
    std::ostringstream oss; oss<<out_dir<<"/1D_c_"<<std::setw(4)<<std::setfill('0')<<step<<".vtk";
    write_polydata(oss.str(), Xc,Yc,Zc, &order_c,
      { {"c",C}, {"size", std::vector<double>(C.size(),80.0)} },
      &GPc);
  }
  // v file
  {
    std::ostringstream oss; oss<<out_dir<<"/1D_v_"<<std::setw(4)<<std::setfill('0')<<step<<".vtk";
    write_polydata(oss.str(), Xv,Yv,Zv, &order_v,
      { {"v",V}, {"size", std::vector<double>(V.size(),40.0)} },
      nullptr);
  }
}

/* ===================== 2D ===================== */

void Post_Process_2D(const std::vector<Node>& NL,
                     const std::vector<element>& /*EL*/,
                     int step,
                     const std::string& out_dir,
                     bool write_gp)
{
  if (NL.empty()) return;
  ensure_dir(out_dir);

  // c cloud (z=0 and z=c)
  std::vector<double> Xc,Yc,Zc0,C, size_c, Zc3D;
  std::vector<std::vector<double>> GPc;

  // v cloud (z=0 and z=vx/vy)
  std::vector<double> Xv,Yv,Zv0,Vx,Vy, Vmag, size_v, ZvX, ZvY;

  for (const auto& n : NL){
    if (n.field.size()>=1 && n.field(0)==1){
      const double x=n.x(0), y=n.x(1);
      Xc.push_back(x); Yc.push_back(y); Zc0.push_back(0.0);
      const double c = n.u.size()>=1 ? n.u(0) : 0.0;
      C.push_back(c);
      size_c.push_back(80.0);
      Zc3D.push_back(c);
      if (write_gp && n.GP_vals.size()>0){
        std::vector<double> gp(n.GP_vals.size());
        for (int i=0;i<n.GP_vals.size();++i) gp[i]=n.GP_vals(i);
        GPc.push_back(std::move(gp));
      } else GPc.emplace_back();
    }
    if (n.field.size()>=2 && n.field(1)==1){
      const double x=n.x(0), y=n.x(1);
      const double vx = n.u.size()>=2 ? n.u(1) : 0.0;
      const double vy = n.u.size()>=3 ? n.u(2) : 0.0;
      Xv.push_back(x); Yv.push_back(y); Zv0.push_back(0.0);
      Vx.push_back(vx); Vy.push_back(vy);
      Vmag.push_back(std::sqrt(vx*vx+vy*vy));
      size_v.push_back(40.0);
      ZvX.push_back(vx); ZvY.push_back(vy);
    }
  }

  auto tag = [&](const std::string& base){
    std::ostringstream oss; oss<<out_dir<<"/"<<base<<"_"<<std::setw(4)<<std::setfill('0')<<step<<".vtk"; return oss.str();
  };

  // XY (z=0)
  if (!Xc.empty()) write_polydata(tag("2D_c_xy"), Xc,Yc,Zc0,nullptr,
      { {"c",C}, {"v_mag",std::vector<double>(C.size(),0)}, {"size",size_c} }, &GPc);
  if (!Xv.empty()) write_polydata(tag("2D_vx_xy"), Xv,Yv,Zv0,nullptr,
      { {"vx",Vx}, {"vy",Vy}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);
  if (!Xv.empty()) write_polydata(tag("2D_vy_xy"), Xv,Yv,Zv0,nullptr,
      { {"vy",Vy}, {"vx",Vx}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);

  // XYZ (z=scalar)
  if (!Xc.empty()) write_polydata(tag("2D_c_xyz"), Xc,Yc,Zc3D,nullptr,
      { {"c",C}, {"size",size_c} }, &GPc);
  if (!Xv.empty()) write_polydata(tag("2D_vx_xyz"), Xv,Yv,ZvX,nullptr,
      { {"vx",Vx}, {"vy",Vy}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);
  if (!Xv.empty()) write_polydata(tag("2D_vy_xyz"), Xv,Yv,ZvY,nullptr,
      { {"vy",Vy}, {"vx",Vx}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);
}

/* ===================== 3D ===================== */

static inline bool region_filter(double x, double y, double z, int incr, double tol){
  if (incr < 4){
    return ((x > 0.69) && (std::abs(y - 0.0) < tol)) ||
           ((y > 0.69) && (std::abs(x - 0.0) < tol)) ||
           ((std::abs(z - 1.0) < tol) && (x > 0.69))  ||
           ((std::abs(z - 1.0) < tol) && (y > 0.69))  ||
           (std::abs(y - 0.7) < tol) ||
           (std::abs(x - 0.7) < tol);
  } else {
    return ((x > 0.49) && (std::abs(y - 0.0) < tol)) ||
           ((y > 0.49) && (std::abs(x - 0.0) < tol)) ||
           ((std::abs(z - 1.0) < tol) && (x > 0.49))  ||
           ((std::abs(z - 1.0) < tol) && (y > 0.49))  ||
           (std::abs(y - 0.5) < tol) ||
           (std::abs(x - 0.5) < tol);
  }
}

void Post_Process_3D(const std::vector<Node>& NL,
                     const std::vector<element>& /*EL*/,
                     int incr,
                     const std::string& out_dir,
                     bool write_gp)
{
  if (NL.empty()) return;
  ensure_dir(out_dir);
  constexpr double tol = 5e-2;

  // c cloud
  std::vector<double> Xc,Yc,Zc,C, Vxc,Vyc,Vzc, Vmag_c, size_c;
  std::vector<std::vector<double>> GPc;

  // v cloud
  std::vector<double> Xv,Yv,Zv,Vx,Vy,Vz,Vmag, size_v;

  for (const auto& n : NL){
    const double x=n.x(0), y=n.x(1), z=n.x(2);
    if (n.field.size()>=1 && n.field(0)==1 && region_filter(x,y,z,incr,tol)){
      Xc.push_back(x); Yc.push_back(y); Zc.push_back(z);
      const double c  = n.u.size()>=1 ? n.u(0) : 0.0;
      const double vx = n.u.size()>=2 ? n.u(1) : 0.0;
      const double vy = n.u.size()>=3 ? n.u(2) : 0.0;
      const double vz = n.u.size()>=4 ? n.u(3) : 0.0;
      C.push_back(c); Vxc.push_back(vx); Vyc.push_back(vy); Vzc.push_back(vz);
      Vmag_c.push_back(std::sqrt(vx*vx+vy*vy+vz*vz));
      size_c.push_back(130.0);
      if (write_gp && n.GP_vals.size()>0){
        std::vector<double> gp(n.GP_vals.size());
        for (int i=0;i<n.GP_vals.size();++i) gp[i]=n.GP_vals(i);
        GPc.push_back(std::move(gp));
      } else GPc.emplace_back();
    }
    if (n.field.size()>=2 && n.field(1)==1 && region_filter(x,y,z,incr,tol)){
      Xv.push_back(x); Yv.push_back(y); Zv.push_back(z);
      const double vx = n.u.size()>=2 ? n.u(1) : 0.0;
      const double vy = n.u.size()>=3 ? n.u(2) : 0.0;
      const double vz = n.u.size()>=4 ? n.u(3) : 0.0;
      Vx.push_back(vx); Vy.push_back(vy); Vz.push_back(vz);
      Vmag.push_back(std::sqrt(vx*vx+vy*vy+vz*vz));
      size_v.push_back(130.0);
    }
  }

  auto tag = [&](const std::string& base){
    std::ostringstream oss; oss<<out_dir<<"/"<<base<<"_"<<std::setw(4)<<std::setfill('0')<<incr<<".vtk"; return oss.str();
  };

  if (!Xc.empty()) write_polydata(tag("3D_c"), Xc,Yc,Zc,nullptr,
      { {"c",C}, {"vx",Vxc}, {"vy",Vyc}, {"vz",Vzc}, {"v_mag",Vmag_c}, {"size",size_c} }, &GPc);
  if (!Xv.empty()) write_polydata(tag("3D_vx"), Xv,Yv,Zv,nullptr,
      { {"vx",Vx}, {"vy",Vy}, {"vz",Vz}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);
  if (!Xv.empty()) write_polydata(tag("3D_vy"), Xv,Yv,Zv,nullptr,
      { {"vy",Vy}, {"vx",Vx}, {"vz",Vz}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);
  if (!Xv.empty()) write_polydata(tag("3D_vz"), Xv,Yv,Zv,nullptr,
      { {"vz",Vz}, {"vx",Vx}, {"vy",Vy}, {"v_mag",Vmag}, {"size",size_v} }, nullptr);
}

/* -------- dispatcher -------- */
void Post_Process(const std::vector<Node>& NL,
                  const std::vector<element>& EL,
                  int step,
                  const std::string& out_dir,
                  bool write_gp)
{
  if (NL.empty()) return;
  const int PD = NL.front().PD;
  if      (PD == 1) Post_Process_1D(NL,EL,step,out_dir,write_gp);
  else if (PD == 2) Post_Process_2D(NL,EL,step,out_dir,write_gp);
  else if (PD == 3) Post_Process_3D(NL,EL,step,out_dir,write_gp);
  else throw std::runtime_error("Unsupported PD in Post_Process()");
}

} // namespace vtkio
