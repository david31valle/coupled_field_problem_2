// mesh_3D.cpp
#include "mesh_3D.hpp"
#include <algorithm>
#include <iostream>

// lexicographic compare for 3D
static bool cmp3(const std::array<double,3>& a,
                 const std::array<double,3>& b) {
    if (a[0]<b[0]) return true;
    if (a[0]>b[0]) return false;
    if (a[1]<b[1]) return true;
    if (a[1]>b[1]) return false;
    return a[2]<b[2];
}

Mesh_3D::Result Mesh_3D::generate(double domain_size,
                                  int partition,
                                  const std::vector<int>& element_orders) const
{
    // 1) Build each individual mesh
    std::vector<NodeMat> NL_list;
    std::vector<ElemMat> EL_list;
    NL_list.reserve(element_orders.size());
    EL_list.reserve(element_orders.size());
    for (int deg : element_orders) {
        NodeMat NL_i;
        ElemMat EL_i;
        individual(domain_size, partition, deg, NL_i, EL_i);
        NL_list.push_back(std::move(NL_i));
        EL_list.push_back(std::move(EL_i));
    }

    // 2) Union of all node-rows
    std::vector<std::array<double,3>> all_nodes;
    for (auto& NL_i : NL_list)
        for (int r = 0; r < NL_i.rows(); ++r)
            all_nodes.push_back({NL_i(r,0), NL_i(r,1), NL_i(r,2)});
    std::sort(all_nodes.begin(), all_nodes.end(), cmp3);
    all_nodes.erase(std::unique(all_nodes.begin(), all_nodes.end()), all_nodes.end());

    // 3) Pack global NL
    Result R;
    R.NL.resize(all_nodes.size(), 3);
    for (size_t i = 0; i < all_nodes.size(); ++i)
        R.NL.row(i) = Eigen::Vector3d(
            all_nodes[i][0],
            all_nodes[i][1],
            all_nodes[i][2]
        );

    // 4) Remap each EL into global indices
    for (size_t m = 0; m < EL_list.size(); ++m) {
        const auto& NL_i = NL_list[m];
        const auto& Eold = EL_list[m];
        ElemMat Enew = Eold;
        for (int e = 0; e < Eold.rows(); ++e) {
            for (int k = 0; k < Eold.cols(); ++k) {
                int old_idx = Eold(e,k);  // 0-based
                auto xyz = std::array<double,3>{
                    NL_i(old_idx,0),
                    NL_i(old_idx,1),
                    NL_i(old_idx,2)
                };
                auto it = std::lower_bound(all_nodes.begin(),
                                           all_nodes.end(),
                                           xyz, cmp3);
                Enew(e,k) = int(std::distance(all_nodes.begin(), it));
            }
        }
        R.EL.push_back(std::move(Enew));
    }

    // 5) Convert to 1-based indexing
    for (auto& E : R.EL)
        E.array() += 1;

    return R;
}

void Mesh_3D::individual(double domain_size,
                         int partition,
                         int degree,
                         NodeMat& NL,
                         ElemMat& EL) const
{
    int p   = partition;
    int dp  = degree * p;
    int nps = dp + 1;                   // nodes per side
    int Nn  = nps * nps * nps;
    int Ne  = p * p * p;
    int Npe = (degree+1)*(degree+1)*(degree+1);

    // 1) Nodes: exactly MATLAB's ndgrid + (:)
    NL.resize(Nn, 3);
    double dd = domain_size / double(dp);
    int idx = 0;
    for (int k = 0; k < nps; ++k)
        for (int j = 0; j < nps; ++j)
            for (int i = 0; i < nps; ++i)
                NL.row(idx++) = Eigen::Vector3d(i*dd, j*dd, k*dd);

    // 2) Elements: fully replicate MATLAB switch-case
    EL.setZero(Ne, Npe);
    int plane = (p+1)*(p+1);
    int sq2   = (2*p+1)*(2*p+1);
    int sq3   = (3*p+1)*(3*p+1);
    int sq4   = (4*p+1)*(4*p+1);

    for (int i = 1; i <= p; ++i) {
        for (int j = 1; j <= p; ++j) {
            for (int k = 1; k <= p; ++k) {
                int eid = (i-1)*p*p + (j-1)*p + (k-1);

                if (i==1 && k==1) {
                    switch (degree) {
                        case 1: {
                            int n4 = (j-1)*(p+1)+(k-1);
                            int n3 = n4+1;
                            int n8 = n4+(p+1);
                            int n7 = n8+1;
                            int n1 = n4 + plane;
                            int n2 = n3 + plane;
                            int n5 = n8 + plane;
                            int n6 = n7 + plane;
                            // [1 2 3 4 5 6 7 8]
                            EL(eid,0)=n1;
                            EL(eid,1)=n2;
                            EL(eid,2)=n3;
                            EL(eid,3)=n4;
                            EL(eid,4)=n5;
                            EL(eid,5)=n6;
                            EL(eid,6)=n7;
                            EL(eid,7)=n8;
                        } break;

                    case 2: {
                            int p1    = 2*p + 1;         // = (degree*p+1)
                            int plane = p1 * p1;         // (2*p+1)^2
                            // j,k are zero-based here (MATLAB’s j-1,k-1)
                            int b = 2*j*p1 + k;          // = 2*(j)*(2*p+1) + k
                            // Face at i==1, k==1 in MATLAB, here i==0,k==0
                            // MATLAB EL(:,4)  → C++ EL(eid,3)
                            EL(eid,3)  = b;
                            // MATLAB EL(:,11) → C++ EL(eid,10)
                            EL(eid,10) = b + 1;
                            // MATLAB EL(:,3)  → C++ EL(eid,2)
                            EL(eid,2)  = b + 2;

                            // Next row in j-direction
                            EL(eid,15) = b + p1;            // MATLAB col16
                            EL(eid,23) = EL(eid,15) + 1;    // col24
                            EL(eid,14) = EL(eid,15) + 2;    // col15

                            // Next row in j-direction again
                            EL(eid,7)  = b + 2*p1;          // col8
                            EL(eid,18) = EL(eid,7) + 1;     // col19
                            EL(eid,6)  = EL(eid,7) + 2;     // col7

                            // Now the “top” layer (add (2*p+1)^2)
                            EL(eid,11) = b + plane;                 // col12
                            EL(eid,20) = EL(eid,10) + plane;        // col21
                            EL(eid,9)  = EL(eid,2) + plane;         // col10

                            EL(eid,24) = EL(eid,15) + plane;        // col25
                            EL(eid,26) = EL(eid,23) + plane;        // col27
                            EL(eid,22) = EL(eid,14) + plane;        // col23
                            EL(eid,19) = EL(eid,7) + plane;         // col20
                            EL(eid,25) = EL(eid,18) + plane;        // col26
                            EL(eid,17) = EL(eid,6) + plane;         // col18

                            // Finally, the very top layer (add 2*(2*p+1)^2)
                            EL(eid,0)  = EL(eid,11) + plane;        // col1
                            EL(eid,8)  = EL(eid,20) + plane;        // col9
                            EL(eid,1)  = EL(eid,9)  + plane;        // col2
                            EL(eid,12) = EL(eid,24) + plane;        // col13
                            EL(eid,21) = EL(eid,26) + plane;        // col22
                            EL(eid,13) = EL(eid,22) + plane;        // col14
                            EL(eid,4)  = EL(eid,19) + plane;        // col5
                            EL(eid,16) = EL(eid,25) + plane;        // col17
                            EL(eid,5)  = EL(eid,17) + plane;        // col6
                        } break;

                        case 3: {
                            int p3    = 3*p + 1;
                            int plane = p3 * p3;
                            int b     = 3*j*p3 + k;

                            EL(eid, 3)  = b;
                            EL(eid,13)  = b + 1;
                            EL(eid,12)  = b + 2;
                            EL(eid, 2)  = b + 3;

                            int n23 = b + p3;
                            EL(eid,22)  = n23;
                            EL(eid,45)  = n23 + 1;
                            EL(eid,44)  = n23 + 2;
                            EL(eid,20)  = n23 + 3;

                            int n24 = n23 + p3;
                            EL(eid,23)  = n24;
                            EL(eid,46)  = n24 + 1;
                            EL(eid,47)  = n24 + 2;
                            EL(eid,21)  = n24 + 3;

                            int n8 = n24 + p3;
                            EL(eid, 7)  = n8;
                            EL(eid,29)  = n8 + 1;
                            EL(eid,28)  = n8 + 2;
                            EL(eid, 6)  = n8 + 3;

                            EL(eid,14)  = b + plane;
                            EL(eid,35)  = b + 1 + plane;
                            EL(eid,34)  = b + 2 + plane;
                            EL(eid,11)  = b + 3 + plane;

                            EL(eid,48)  = n23 + plane;
                            EL(eid,59)  = n23 + 1 + plane;
                            EL(eid,58)  = n23 + 2 + plane;
                            EL(eid,41)  = n23 + 3 + plane;

                            EL(eid,51)  = n24 + plane;
                            EL(eid,63)  = n24 + 1 + plane;
                            EL(eid,62)  = n24 + 2 + plane;
                            EL(eid,42)  = n24 + 3 + plane;

                            EL(eid,30)  = n8 + plane;
                            EL(eid,55)  = n8 + 1 + plane;
                            EL(eid,54)  = n8 + 2 + plane;
                            EL(eid,27)  = n8 + 3 + plane;

                            EL(eid,15)  = EL(eid,14) + plane;
                            EL(eid,32)  = EL(eid,35) + plane;
                            EL(eid,33)  = EL(eid,34) + plane;
                            EL(eid,10)  = EL(eid,11) + plane;

                            EL(eid,49)  = EL(eid,48) + plane;
                            EL(eid,56)  = EL(eid,59) + plane;
                            EL(eid,57)  = EL(eid,58) + plane;
                            EL(eid,40)  = EL(eid,41) + plane;

                            EL(eid,50)  = EL(eid,51) + plane;
                            EL(eid,61)  = EL(eid,63) + plane;
                            EL(eid,60)  = EL(eid,62) + plane;
                            EL(eid,43)  = EL(eid,42) + plane;

                            EL(eid,31)  = EL(eid,30) + plane;
                            EL(eid,53)  = EL(eid,55) + plane;
                            EL(eid,52)  = EL(eid,54) + plane;
                            EL(eid,26)  = EL(eid,27) + plane;

                            EL(eid, 0)  = EL(eid,15) + plane;
                            EL(eid, 8)  = EL(eid,32) + plane;
                            EL(eid, 9)  = EL(eid,33) + plane;
                            EL(eid, 1)  = EL(eid,10) + plane;

                            EL(eid,16)  = EL(eid,49) + plane;
                            EL(eid,36)  = EL(eid,56) + plane;
                            EL(eid,37)  = EL(eid,57) + plane;
                            EL(eid,18)  = EL(eid,40) + plane;

                            EL(eid,17)  = EL(eid,50) + plane;
                            EL(eid,39)  = EL(eid,61) + plane;
                            EL(eid,38)  = EL(eid,60) + plane;
                            EL(eid,19)  = EL(eid,43) + plane;

                            EL(eid, 4)  = EL(eid,27) + plane;
                        } break;

                        case 4: {
                            // build 5×5×5 element by explicit nested loops to mirror MATLAB’s ordering
                            int dp  = 4*p + 1;       // nodes per side for degree 4
                            int nps = dp;
                            // local index runs 0..124 in the same pattern as MATLAB’s xx(:),yy(:),zz(:)
                            int cnt = 0;
                            for (int kz = 0; kz <= 4; ++kz) {
                                for (int jy = 0; jy <= 4; ++jy) {
                                    for (int ix = 0; ix <= 4; ++ix) {
                                        // global node index (0-based)
                                        int x = (i-1)*4 + ix;
                                        int y = (j-1)*4 + jy;
                                        int z = (k-1)*4 + kz;
                                        int gidx = (z * nps + y) * nps + x;
                                        EL(eid, cnt++) = gidx;
                                    }
                                }
                            }
                        } break;
                    }
                }
                else if (i==1) {
                    // k>1: shift previous element by degree
                    for (int c=0; c<Npe; ++c)
                        EL(eid,c) = EL(eid-1,c) + degree;
                }
                else {
                    // i>1: shift element p^2 behind by degree*plane
                    for (int c=0; c<Npe; ++c)
                        EL(eid,c) = EL(eid - p*p, c) + degree*plane;
                }
            }
        }
    }
}

// MATLAB-style overloads
std::pair<Mesh_3D::NodeMat,Mesh_3D::ElemMat>
mesh_3D(int, double ds, int p, int o) {
    Mesh_3D m; auto R = m.generate(ds,p,{o}); return{R.NL,R.EL[0]};}
std::tuple<Mesh_3D::NodeMat,Mesh_3D::ElemMat,Mesh_3D::ElemMat>
mesh_3D(int, double ds, int p, const std::array<int,2>& o) {
    Mesh_3D m; auto R = m.generate(ds,p,{o[0],o[1]}); return{R.NL,R.EL[0],R.EL[1]};}
std::tuple<Mesh_3D::NodeMat,Mesh_3D::ElemMat,Mesh_3D::ElemMat,Mesh_3D::ElemMat>
mesh_3D(int, double ds, int p, const std::array<int,3>& o) {
    Mesh_3D m; auto R = m.generate(ds,p,{o[0],o[1],o[2]});
    return{R.NL,R.EL[0],R.EL[1],R.EL[2]};}

// Debug printer
void printMesh3D(const Mesh_3D::NodeMat& NL,
                 const Mesh_3D::ElemMat& EL)
{
    std::cout<<"=== 3D Mesh ===\n";
    std::cout<<"Nodes("<<NL.rows()<<"):\n";
    for(int i=0;i<NL.rows();++i)
        std::cout<<" ["<<NL(i,0)<<','<<NL(i,1)<<','<<NL(i,2)<<"]\n";
    std::cout<<"\nElements("<<EL.rows()<<"):\n";
    for(int e=0;e<EL.rows();++e){
        std::cout<<"  [ ";
        for(int c=0;c<EL.cols();++c)
            std::cout<<EL(e,c)<<(c+1<EL.cols()? ", ": " ]\n");
    }
    std::cout<<"===============\n";
}