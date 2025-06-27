#include "NaturalElement.h"
#include <iostream>

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> build(int NPE, int PD, int NGP, const Eigen::MatrixXd &GP){
    Eigen::ArrayXXd N;
    Eigen::ArrayXXd gradN_xi;

    if(PD==1){
            Eigen::ArrayXXd xi = GP.row(0);
            Eigen::ArrayXXd w = GP.row(1);
            Eigen::ArrayXXd ones = Eigen::ArrayXXd::Ones(1,NGP*PD);
            switch (NPE){
                case 2:
                    // num rows = num of shape functions = NPE ///  num cols = NGP 
                    N.row(0) = -0.5 * (xi - 1);
                    N.row(1) =  0.5 * (xi + 1);
                    
                    // gradients evaluated at each GP
                    gradN_xi.row(0) << -0.5 * ones;
                    gradN_xi.row(1) << 0.5 * ones;

                    break;

                case 3:
                    N.row(0) = 1.0/2 * (xi)   * (xi-1);
                    N.row(1) = -1.0  * (xi-1) * (xi+1);
                    N.row(2) = 1.0/2 * (xi)   * (xi+1);
                    
                    gradN_xi.row(0) =  1.0/2 * (2*xi-1);
                    gradN_xi.row(1) =  -2.0  * (xi);
                    gradN_xi.row(2) =  1.0/2 * (2*xi+1);
                    break;

                case 4:
                    N.row(0) = -9.0/16  * (xi+1.0/3) * (xi-1.0/3) * (xi-1);
                    N.row(1) =  27.0/16 * (xi+1)   * (xi-1.0/3) * (xi-1);
                    N.row(2) = -27.0/16 * (xi+1)   * (xi+1.0/3) * (xi-1);
                    N.row(3) =  9.0/16  * (xi+1)   * (xi+1.0/3) * (xi-1.0/3);

                    gradN_xi.row(0) = -9.0/16  * (xi - 1)   * (xi - 1.0/3) - ((9 *xi)/16.0  + 3/16.0)  * (xi - 1)   - ((9 *xi)/16.0 + 3/16.0)  * (xi - 1.0/3);
                    gradN_xi.row(1) =  27.0/16 * (xi - 1)   * (xi - 1.0/3) + ((27*xi)/16.0  + 27/16.0) * (xi - 1)   + ((27*xi)/16.0 + 27/16.0) * (xi - 1.0/3);
                    gradN_xi.row(2) = -27.0/16 * (xi - 1)   * (xi + 1.0/3) - ((27*xi)/16.0  + 27/16.0) * (xi - 1)   - ((27*xi)/16.0 + 27/16.0) * (xi + 1.0/3);
                    gradN_xi.row(3) =  9.0/16  * (xi - 1.0/3) * (xi + 1.0/3) + ((9 *xi)/16.0  + 9/16.0)  * (xi - 1.0/3) + ((9 *xi)/16.0 + 9/16.0)  * (xi + 1.0/3);

                    break;
            }
    }

    else if(PD==2){
            Eigen::ArrayXXd xi = GP.row(0);
            Eigen::ArrayXXd eta = GP.row(1);
            Eigen::ArrayXXd w = GP.row(2);

            Eigen::ArrayXXd dN_dxi(NPE, NGP);
            Eigen::ArrayXXd dN_deta(NPE, NGP);

            switch (NPE){
                case 4:
                    // num rows = num of shape functions = NPE ///  num cols = NGP 
                    N.row(0) = 1.0/4 * (1-xi) * (1-eta);
                    N.row(1) = 1.0/4 * (1+xi) * (1-eta);
                    N.row(2) = 1.0/4 * (1+xi) * (1+eta);
                    N.row(3) = 1.0/4 * (1-xi) * (1+eta);

                    // gradients evaluated at each GP            
                    dN_dxi.row(0) = -1.0/4 * (1-eta);
                    dN_dxi.row(1) =  1.0/4 * (1-eta);
                    dN_dxi.row(2) =  1.0/4 * (1+eta);
                    dN_dxi.row(3) = -1.0/4 * (1+eta);
                    
                    dN_deta.row(0) = -1.0/4 * (1-xi);
                    dN_deta.row(1) = -1.0/4 * (1+xi);
                    dN_deta.row(2) =  1.0/4 * (1+xi);
                    dN_deta.row(3) =  1.0/4 * (1-xi);
                    break;

                case 9:
                    N.row(0) =  1/4.0    * (1-xi) * xi      * (1-eta) * eta;
                    N.row(1) = -1/4.0    * (1+xi) * xi      * (1-eta) * eta;
                    N.row(2) =  1/4.0    * (1+xi) * xi      * (1+eta) * eta;
                    N.row(3) = -1/4.0    * (1-xi) * xi      * (1+eta) * eta;
                    N.row(4) = -1/2.0    * (1-xi) * (1+xi)  * (1-eta) * eta;
                    N.row(5) =  1/2.0    * (1+xi) * xi      * (1+eta) * (1-eta);
                    N.row(6) =  1/2.0    * (1-xi) * (1+xi)  * (1+eta) * eta;
                    N.row(7) = -1/2.0    * (1-xi) * xi      * (1+eta) * (1-eta);
                    N.row(8) = (1-xi) * (1+xi) * (1+eta) * (1-eta);

                    dN_dxi.row(0) =  1/4.0 * (1-2*xi) * (1-eta) * eta;
                    dN_dxi.row(1) = -1/4.0 * (1+2*xi) * (1-eta) * eta;
                    dN_dxi.row(2) =  1/4.0 * (1+2*xi) * (1+eta) * eta;
                    dN_dxi.row(3) = -1/4.0 * (1-2*xi) * (1+eta) * eta;
                    dN_dxi.row(4) =  xi * eta * (1-eta);
                    dN_dxi.row(5) =  1/2.0 * (1+2*xi) * (1-eta) * (1+eta);
                    dN_dxi.row(6) = -xi * eta * (1+eta);
                    dN_dxi.row(7) = -1/2.0 * (1-2*xi) * (1-eta) * (1+eta);
                    dN_dxi.row(8) = -2 * xi * (1-eta) * (1+eta);
                    
                    dN_deta.row(0) =  1/4.0 * (1-xi) * xi * (1-2*eta);
                    dN_deta.row(1) = -1/4.0 * (1+xi) * xi * (1-2*eta);
                    dN_deta.row(2) =  1/4.0 * (1+xi) * xi * (1+2*eta);
                    dN_deta.row(3) = -1/4.0 * (1-xi) * xi * (1+2*eta);
                    dN_deta.row(4) =  1/2.0 * (1-xi) * (1+xi) * (2*eta-1);
                    dN_deta.row(5) = -(1+xi) * xi * eta;
                    dN_deta.row(6) =  1/2.0 * (1-xi) * (1+xi) * (1+2*eta);
                    dN_deta.row(7) =  (1-xi) * xi * eta;
                    dN_deta.row(8) =  -2 * (1-xi) * (1+xi) * eta;
                    break;
                
                case 16:
                    N.row(0)  =  ((9*eta)/16.0 + 3/16.0)  * ((9*xi)/16.0 + 3/16.0)  * (eta - 1)  * (eta - 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(1)  = -((9*eta)/16.0 + 3/16.0)  * ((9*xi)/16.0 + 9/16.0)  * (eta - 1)  * (eta - 1/3.0)* (xi - 1/3.0)* (xi + 1/3.0);
                    N.row(2)  =  ((9*eta)/16.0 + 9/16.0)  * ((9*xi)/16.0 + 9/16.0)  * (eta - 1/3.0)* (eta + 1/3.0)* (xi - 1/3.0)* (xi + 1/3.0);
                    N.row(3)  = -((9*eta)/16.0 + 9/16.0)  * ((9*xi)/16.0 + 3/16.0)  * (eta - 1/3.0)* (eta + 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(4)  = -((9*eta)/16.0 + 3/16.0)  * ((27*xi)/16.0 + 27/16.0)* (eta - 1)  * (eta - 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(5)  =  ((9*eta)/16.0 + 3/16.0)  * ((27*xi)/16.0 + 27/16.0)* (eta - 1)  * (eta - 1/3.0)* (xi - 1)  * (xi + 1/3.0);
                    N.row(6)  =  ((27*eta)/16.0 + 27/16.0)* ((9*xi)/16.0 + 9/16.0)  * (eta - 1)  * (eta - 1/3.0)* (xi - 1/3.0)* (xi + 1/3.0);
                    N.row(7)  = -((27*eta)/16.0 + 27/16.0)* ((9*xi)/16.0 + 9/16.0)  * (eta - 1)  * (eta + 1/3.0)* (xi - 1/3.0)* (xi + 1/3.0);
                    N.row(8)  = -((9*eta)/16.0 + 9/16.0)  * ((27*xi)/16.0 + 27/16.0)* (eta - 1/3.0)* (eta + 1/3.0)* (xi - 1)  * (xi + 1/3.0);
                    N.row(9) =  ((9*eta)/16.0 + 9/16.0)  * ((27*xi)/16.0 + 27/16.0)* (eta - 1/3.0)* (eta + 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(10) =  ((27*eta)/16.0 + 27/16.0)* ((9*xi)/16.0 + 3/16.0)  * (eta - 1)  * (eta + 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(11) = -((27*eta)/16.0 + 27/16.0)* ((9*xi)/16.0 + 3/16.0)  * (eta - 1)  * (eta - 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(12) =  ((27*eta)/16.0 + 27/16.0)* ((27*xi)/16.0 + 27/16.0)* (eta - 1)  * (eta - 1/3.0)* (xi - 1)  * (xi - 1/3.0);
                    N.row(13) = -((27*eta)/16.0 + 27/16.0)* ((27*xi)/16.0 + 27/16.0)* (eta - 1)  * (eta - 1/3.0)* (xi - 1)  * (xi + 1/3.0);
                    N.row(14) =  ((27*eta)/16.0 + 27/16.0)* ((27*xi)/16.0 + 27/16.0)* (eta - 1)  * (eta + 1/3.0)* (xi - 1)  * (xi + 1/3.0);
                    N.row(15) = -((27*eta)/16.0 + 27/16.0)* ((27*xi)/16.0 + 27/16.0)* (eta - 1)  * (eta + 1/3.0)* (xi - 1)  * (xi - 1/3.0);

                    Eigen::ArrayXXd xi2 = xi * xi;
                    Eigen::ArrayXXd xi3 = xi * xi * xi;
                    Eigen::ArrayXXd eta2 = eta * eta;
                    Eigen::ArrayXXd eta3 = eta * eta * eta;
                    
                    dN_dxi.row(0)  =  ((- 27*xi2 + 18*xi + 1)  * (- 9*eta3 + 9*eta2 + eta - 1))/256.0;
                    dN_dxi.row(1)  =  ((27*xi2 + 18*xi - 1)    * (- 9*eta3 + 9*eta2 + eta - 1))/256.0;
                    dN_dxi.row(2)  = -((27*xi2 + 18*xi - 1)    * (- 9*eta3 - 9*eta2 + eta + 1))/256.0;
                    dN_dxi.row(3)  = -((- 27*xi2 + 18*xi + 1)  * (- 9*eta3 - 9*eta2 + eta + 1))/256.0;
                    dN_dxi.row(4)  = -(9*(- 9*xi2 + 2*xi + 3)  * (- 9*eta3 + 9*eta2 + eta - 1))/256.0;
                    dN_dxi.row(5)  = -(9*(9*xi2 + 2*xi - 3)    * (- 9*eta3 + 9*eta2 + eta - 1))/256.0;
                    dN_dxi.row(6)  = -(9*(27*xi2 + 18*xi - 1)  * (- 3*eta3 + eta2 + 3*eta - 1))/256.0;
                    dN_dxi.row(7)  =  (9*(27*xi2 + 18*xi - 1)  * (- 3*eta3 - eta2 + 3*eta + 1))/256.0;
                    dN_dxi.row(8)  =  (9*(9*xi2 + 2*xi - 3)    * (- 9*eta3 - 9*eta2 + eta + 1))/256.0;
                    dN_dxi.row(9) =  (9*(- 9*xi2 + 2*xi + 3)  * (- 9*eta3 - 9*eta2 + eta + 1))/256.0;
                    dN_dxi.row(10) =  (9*(- 27*xi2 + 18*xi + 1)* (- 3*eta3 - eta2 + 3*eta + 1))/256.0;
                    dN_dxi.row(11) = -(9*(- 27*xi2 + 18*xi + 1)* (- 3*eta3 + eta2 + 3*eta - 1))/256.0;
                    dN_dxi.row(12) =  (81*(- 9*xi2 + 2*xi + 3) * (- 3*eta3 + eta2 + 3*eta - 1))/256.0;
                    dN_dxi.row(13) =  (81*(9*xi2 + 2*xi - 3)   * (- 3*eta3 + eta2 + 3*eta - 1))/256.0;
                    dN_dxi.row(14) = -(81*(9*xi2 + 2*xi - 3)   * (- 3*eta3 - eta2 + 3*eta + 1))/256.0;
                    dN_dxi.row(15) = -(81*(- 9*xi2 + 2*xi + 3) * (- 3*eta3 - eta2 + 3*eta + 1))/256.0;

                    dN_deta.row(0)  =  ((- 27*eta2 + 18*eta + 1)   * (- 9*xi3 + 9*xi2 + xi - 1))/256.0;
                    dN_deta.row(1)  = -((- 27*eta2 + 18*eta + 1)   * (- 9*xi3 - 9*xi2 + xi + 1))/256.0;
                    dN_deta.row(2)  = -((27*eta2 + 18*eta - 1)     * (- 9*xi3 - 9*xi2 + xi + 1))/256.0;
                    dN_deta.row(3)  =  ((27*eta2 + 18*eta - 1)     * (- 9*xi3 + 9*xi2 + xi - 1))/256.0;
                    dN_deta.row(4)  = -(9*(- 27*eta2 + 18*eta + 1) * (- 3*xi3 + xi2 + 3*xi - 1))/256.0;
                    dN_deta.row(5)  =  (9*(- 27*eta2 + 18*eta + 1) * (- 3*xi3 - xi2 + 3*xi + 1))/256.0;
                    dN_deta.row(6)  =  (9*(- 9*eta2 + 2*eta + 3)   * (- 9*xi3 - 9*xi2 + xi + 1))/256.0;
                    dN_deta.row(7)  =  (9*(9*eta2 + 2*eta - 3)     * (- 9*xi3 - 9*xi2 + xi + 1))/256.0;
                    dN_deta.row(8)  =  (9*(27*eta2 + 18*eta - 1)   * (- 3*xi3 - xi2 + 3*xi + 1))/256.0;
                    dN_deta.row(9) = -(9*(27*eta2 + 18*eta - 1)   * (- 3*xi3 + xi2 + 3*xi - 1))/256.0;
                    dN_deta.row(10) = -(9*(9*eta2 + 2*eta - 3)     * (- 9*xi3 + 9*xi2 + xi - 1))/256.0;
                    dN_deta.row(11) = -(9*(- 9*eta2 + 2*eta + 3)   * (- 9*xi3 + 9*xi2 + xi - 1))/256.0;
                    dN_deta.row(12) =  (81*(- 9*eta2 + 2*eta + 3)  * (- 3*xi3 + xi2 + 3*xi - 1))/256.0;
                    dN_deta.row(13) = -(81*(- 9*eta2 + 2*eta + 3)  * (- 3*xi3 - xi2 + 3*xi + 1))/256.0;
                    dN_deta.row(14) = -(81*(9*eta2 + 2*eta - 3)    * (- 3*xi3 - xi2 + 3*xi + 1))/256.0;
                    dN_deta.row(15) =  (81*(9*eta2 + 2*eta - 3)    * (- 3*xi3 + xi2 + 3*xi - 1))/256.0;
                    break;
            }

            for(int gp=0; gp<NGP; ++gp){
                gradN_xi.col(2*gp) = dN_dxi.col(gp);
                gradN_xi.col(2*gp+1) = dN_deta.col(gp);
            }
    }

    else if(PD==3){
            Eigen::ArrayXXd xi = GP.row(0);
            Eigen::ArrayXXd eta = GP.row(1);
            Eigen::ArrayXXd zeta = GP.row(2);
            Eigen::ArrayXXd w = GP.row(3);

            Eigen::ArrayXXd dN_dxi(NPE, NGP);
            Eigen::ArrayXXd dN_deta(NPE, NGP);
            Eigen::ArrayXXd dN_dzeta(NPE, NGP);

            switch (NPE){
                case 8:

                    N.row(0) = 1/8.0 * (1-xi) * (1-eta) * (1-zeta);
                    N.row(1) = 1/8.0 * (1+xi) * (1-eta) * (1-zeta);
                    N.row(2) = 1/8.0 * (1+xi) * (1+eta) * (1-zeta);
                    N.row(3) = 1/8.0 * (1-xi) * (1+eta) * (1-zeta);
                    N.row(4) = 1/8.0 * (1-xi) * (1-eta) * (1+zeta);
                    N.row(5) = 1/8.0 * (1+xi) * (1-eta) * (1+zeta);
                    N.row(6) = 1/8.0 * (1+xi) * (1+eta) * (1+zeta);
                    N.row(7) = 1/8.0 * (1-xi) * (1+eta) * (1+zeta);

                    dN_dxi.row(0) = -1/8.0 * (1-eta) * (1-zeta);
                    dN_dxi.row(1) =  1/8.0 * (1-eta) * (1-zeta);
                    dN_dxi.row(2) =  1/8.0 * (1+eta) * (1-zeta);
                    dN_dxi.row(3) = -1/8.0 * (1+eta) * (1-zeta);
                    dN_dxi.row(4) = -1/8.0 * (1-eta) * (1+zeta);
                    dN_dxi.row(5) =  1/8.0 * (1-eta) * (1+zeta);
                    dN_dxi.row(6) =  1/8.0 * (1+eta) * (1+zeta);
                    dN_dxi.row(7) = -1/8.0 * (1+eta) * (1+zeta);

                    dN_deta.row(0) = -1/8.0 * (1-xi) * (1-zeta);
                    dN_deta.row(1) = -1/8.0 * (1+xi) * (1-zeta);
                    dN_deta.row(2) =  1/8.0 * (1+xi) * (1-zeta);
                    dN_deta.row(3) =  1/8.0 * (1-xi) * (1-zeta);
                    dN_deta.row(4) = -1/8.0 * (1-xi) * (1+zeta);
                    dN_deta.row(5) = -1/8.0 * (1+xi) * (1+zeta);
                    dN_deta.row(6) =  1/8.0 * (1+xi) * (1+zeta);
                    dN_deta.row(7) =  1/8.0 * (1-xi) * (1+zeta);

                    dN_dzeta.row(0) = -1/8.0 * (1-xi) * (1-eta);
                    dN_dzeta.row(1) = -1/8.0 * (1+xi) * (1-eta);
                    dN_dzeta.row(2) = -1/8.0 * (1+xi) * (1+eta);
                    dN_dzeta.row(3) = -1/8.0 * (1-xi) * (1+eta);
                    dN_dzeta.row(4) =  1/8.0 * (1-xi) * (1-eta);
                    dN_dzeta.row(5) =  1/8.0 * (1+xi) * (1-eta);
                    dN_dzeta.row(6) =  1/8.0 * (1+xi) * (1+eta);
                    dN_dzeta.row(7) =  1/8.0 * (1-xi) * (1+eta);

                    break;

                case 27:
                    N.row(0) =  (1/8.0) * (xi)   * (xi-1) * (eta)   * (eta-1) * (zeta)   * (zeta-1);
                    N.row(1) =  (1/8.0) * (xi)   * (xi+1) * (eta)   * (eta-1) * (zeta)   * (zeta-1);
                    N.row(2) =  (1/8.0) * (xi)   * (xi+1) * (eta)   * (eta+1) * (zeta)   * (zeta-1);
                    N.row(3) =  (1/8.0) * (xi)   * (xi-1) * (eta)   * (eta+1) * (zeta)   * (zeta-1);
                    N.row(4) =  (1/8.0) * (xi)   * (xi-1) * (eta)   * (eta-1) * (zeta)   * (zeta+1);
                    N.row(5) =  (1/8.0) * (xi)   * (xi+1) * (eta)   * (eta-1) * (zeta)   * (zeta+1);
                    N.row(6) =  (1/8.0) * (xi)   * (xi+1) * (eta)   * (eta+1) * (zeta)   * (zeta+1);
                    N.row(7) =  (1/8.0) * (xi)   * (xi-1) * (eta)   * (eta+1) * (zeta)   * (zeta+1);
                    N.row(8) = -(1/4.0) * (xi-1) * (xi+1) * (eta)   * (eta-1) * (zeta)   * (zeta-1);
                    N.row(9) = -(1/4.0) * (xi)   * (xi+1) * (eta-1) * (eta+1) * (zeta)   * (zeta-1);
                    N.row(10) = -(1/4.0) * (xi-1) * (xi+1) * (eta)   * (eta+1) * (zeta)   * (zeta-1);
                    N.row(11) = -(1/4.0) * (xi)   * (xi-1) * (eta-1) * (eta+1) * (zeta)   * (zeta-1);
                    N.row(12) = -(1/4.0) * (xi)   * (xi-1) * (eta)   * (eta-1) * (zeta-1) * (zeta+1);
                    N.row(13) = -(1/4.0) * (xi)   * (xi+1) * (eta)   * (eta-1) * (zeta-1) * (zeta+1);
                    N.row(14) = -(1/4.0) * (xi)   * (xi+1) * (eta)   * (eta+1) * (zeta-1) * (zeta+1);
                    N.row(15) = -(1/4.0) * (xi)   * (xi-1) * (eta)   * (eta+1) * (zeta-1) * (zeta+1);
                    N.row(16) = -(1/4.0) * (xi-1) * (xi+1) * (eta)   * (eta-1) * (zeta)   * (zeta+1);
                    N.row(17) = -(1/4.0) * (xi)   * (xi+1) * (eta-1) * (eta+1) * (zeta)   * (zeta+1);
                    N.row(18) = -(1/4.0) * (xi-1) * (xi+1) * (eta)   * (eta+1) * (zeta)   * (zeta+1);
                    N.row(19) = -(1/4.0) * (xi)   * (xi-1) * (eta-1) * (eta+1) * (zeta)   * (zeta+1);
                    N.row(20) =  (1/2.0) * (xi-1) * (xi+1) * (eta-1) * (eta+1) * (zeta)   * (zeta-1);
                    N.row(21) =  (1/2.0) * (xi-1) * (xi+1) * (eta)   * (eta-1) * (zeta-1) * (zeta+1);
                    N.row(22) =  (1/2.0) * (xi)   * (xi+1) * (eta-1) * (eta+1) * (zeta-1) * (zeta+1);
                    N.row(23) =  (1/2.0) * (xi-1) * (xi+1) * (eta)   * (eta+1) * (zeta-1) * (zeta+1);
                    N.row(24) =  (1/2.0) * (xi)   * (xi-1) * (eta-1) * (eta+1) * (zeta-1) * (zeta+1);
                    N.row(25) =  (1/2.0) * (xi-1) * (xi+1) * (eta-1) * (eta+1) * (zeta)   * (zeta+1);
                    N.row(26) = - (1)  * (xi-1) * (xi+1) * (eta-1) * (eta+1) * (zeta-1) * (zeta+1);

                    dN_dxi.row(0)  =  (eta*zeta*(2*xi - 1)*(eta - 1)*(zeta - 1))/8.0;
                    dN_dxi.row(1)  =  (eta*zeta*(2*xi + 1)*(eta - 1)*(zeta - 1))/8.0;
                    dN_dxi.row(2)  =  (eta*zeta*(2*xi + 1)*(eta + 1)*(zeta - 1))/8.0;
                    dN_dxi.row(3)  =  (eta*zeta*(2*xi - 1)*(eta + 1)*(zeta - 1))/8.0;
                    dN_dxi.row(4)  =  (eta*zeta*(2*xi - 1)*(eta - 1)*(zeta + 1))/8.0;
                    dN_dxi.row(5)  =  (eta*zeta*(2*xi + 1)*(eta - 1)*(zeta + 1))/8.0;
                    dN_dxi.row(6)  =  (eta*zeta*(2*xi + 1)*(eta + 1)*(zeta + 1))/8.0;
                    dN_dxi.row(7)  =  (eta*zeta*(2*xi - 1)*(eta + 1)*(zeta + 1))/8.0;
                    dN_dxi.row(8)  = -(eta*xi*zeta*(eta - 1)*(zeta - 1))/2.0;
                    dN_dxi.row(9)  = -(zeta*(eta*eta - 1)*(2*xi + 1)*(zeta - 1))/4.0;
                    dN_dxi.row(10) = -(eta*xi*zeta*(eta + 1)*(zeta - 1))/2.0;
                    dN_dxi.row(11) = -(zeta*(eta*eta - 1)*(2*xi - 1)*(zeta - 1))/4.0;
                    dN_dxi.row(12) = -(eta*(2*xi - 1)*(zeta*zeta - 1)*(eta - 1))/4.0;
                    dN_dxi.row(13) = -(eta*(2*xi + 1)*(zeta*zeta - 1)*(eta - 1))/4.0;
                    dN_dxi.row(14) = -(eta*(2*xi + 1)*(zeta*zeta - 1)*(eta + 1))/4.0;
                    dN_dxi.row(15) = -(eta*(2*xi - 1)*(zeta*zeta - 1)*(eta + 1))/4.0;
                    dN_dxi.row(16) = -(eta*xi*zeta*(eta - 1)*(zeta + 1))/2.0;
                    dN_dxi.row(17) = -(zeta*(eta*eta - 1)*(2*xi + 1)*(zeta + 1))/4.0;
                    dN_dxi.row(18) = -(eta*xi*zeta*(eta + 1)*(zeta + 1))/2.0;
                    dN_dxi.row(19) = -(zeta*(eta*eta - 1)*(2*xi - 1)*(zeta + 1))/4.0;
                    dN_dxi.row(20) = xi*zeta*(eta*eta - 1)*(zeta - 1);
                    dN_dxi.row(21) = eta*xi*(zeta*zeta - 1)*(eta - 1);
                    dN_dxi.row(22) = ((eta*eta - 1)*(2*xi + 1)*(zeta*zeta - 1))/2.0;
                    dN_dxi.row(23) = eta*xi*(zeta*zeta - 1)*(eta + 1);
                    dN_dxi.row(24) = ((eta*eta - 1)*(2*xi - 1)*(zeta*zeta - 1))/2.0;
                    dN_dxi.row(25) = xi*zeta*(eta*eta - 1)*(zeta + 1);
                    dN_dxi.row(26) = -2*xi*(eta*eta - 1)*(zeta*zeta - 1);

                    dN_deta.row(0)  =  (xi*zeta*(2*eta - 1)*(xi - 1)*(zeta - 1))/8.0;
                    dN_deta.row(1)  =  (xi*zeta*(2*eta - 1)*(xi + 1)*(zeta - 1))/8.0;
                    dN_deta.row(2)  =  (xi*zeta*(2*eta + 1)*(xi + 1)*(zeta - 1))/8.0;
                    dN_deta.row(3)  =  (xi*zeta*(2*eta + 1)*(xi - 1)*(zeta - 1))/8.0;
                    dN_deta.row(4)  =  (xi*zeta*(2*eta - 1)*(xi - 1)*(zeta + 1))/8.0;
                    dN_deta.row(5)  =  (xi*zeta*(2*eta - 1)*(xi + 1)*(zeta + 1))/8.0;
                    dN_deta.row(6)  =  (xi*zeta*(2*eta + 1)*(xi + 1)*(zeta + 1))/8.0;
                    dN_deta.row(7)  =  (xi*zeta*(2*eta + 1)*(xi - 1)*(zeta + 1))/8.0;
                    dN_deta.row(8)  = -(zeta*(2*eta - 1)*(xi*xi - 1)*(zeta - 1))/4.0;
                    dN_deta.row(9)  = -(eta*xi*zeta*(xi + 1)*(zeta - 1))/2.0;
                    dN_deta.row(10) = -(zeta*(2*eta + 1)*(xi*xi - 1)*(zeta - 1))/4.0;
                    dN_deta.row(11) = -(eta*xi*zeta*(xi - 1)*(zeta - 1))/2.0;
                    dN_deta.row(12) = -(xi*(2*eta - 1)*(zeta*zeta - 1)*(xi - 1))/4.0;
                    dN_deta.row(13) = -(xi*(2*eta - 1)*(zeta*zeta - 1)*(xi + 1))/4.0;
                    dN_deta.row(14) = -(xi*(2*eta + 1)*(zeta*zeta - 1)*(xi + 1))/4.0;
                    dN_deta.row(15) = -(xi*(2*eta + 1)*(zeta*zeta - 1)*(xi - 1))/4.0;
                    dN_deta.row(16) = -(zeta*(2*eta - 1)*(xi*xi - 1)*(zeta + 1))/4.0;
                    dN_deta.row(17) = -(eta*xi*zeta*(xi + 1)*(zeta + 1))/2.0;
                    dN_deta.row(18) = -(zeta*(2*eta + 1)*(xi*xi - 1)*(zeta + 1))/4.0;
                    dN_deta.row(19) = -(eta*xi*zeta*(xi - 1)*(zeta + 1))/2.0;
                    dN_deta.row(20) = eta*zeta*(xi*xi - 1)*(zeta - 1);
                    dN_deta.row(21) = ((2*eta - 1)*(xi*xi - 1)*(zeta*zeta - 1))/2.0;
                    dN_deta.row(22) = eta*xi*(zeta*zeta - 1)*(xi + 1);
                    dN_deta.row(23) = ((2*eta + 1)*(xi*xi - 1)*(zeta*zeta - 1))/2.0;
                    dN_deta.row(24) = eta*xi*(zeta*zeta - 1)*(xi - 1);
                    dN_deta.row(25) = eta*zeta*(xi*xi - 1)*(zeta + 1);
                    dN_deta.row(26) = -2*eta*(xi*xi - 1)*(zeta*zeta - 1);

                    dN_dzeta.row(0)  =  (eta*xi*(2*zeta - 1)*(eta - 1)*(xi - 1))/8.0;
                    dN_dzeta.row(1)  =  (eta*xi*(2*zeta - 1)*(eta - 1)*(xi + 1))/8.0;
                    dN_dzeta.row(2)  =  (eta*xi*(2*zeta - 1)*(eta + 1)*(xi + 1))/8.0;
                    dN_dzeta.row(3)  =  (eta*xi*(2*zeta - 1)*(eta + 1)*(xi - 1))/8.0;
                    dN_dzeta.row(4)  =  (eta*xi*(2*zeta + 1)*(eta - 1)*(xi - 1))/8.0;
                    dN_dzeta.row(5)  =  (eta*xi*(2*zeta + 1)*(eta - 1)*(xi + 1))/8.0;
                    dN_dzeta.row(6)  =  (eta*xi*(2*zeta + 1)*(eta + 1)*(xi + 1))/8.0;
                    dN_dzeta.row(7)  =  (eta*xi*(2*zeta + 1)*(eta + 1)*(xi - 1))/8.0;
                    dN_dzeta.row(8)  = -(eta*(xi*xi - 1)*(2*zeta - 1)*(eta - 1))/4.0;
                    dN_dzeta.row(9)  = -(xi*(eta*eta - 1)*(2*zeta - 1)*(xi + 1))/4.0;
                    dN_dzeta.row(10) = -(eta*(xi*xi - 1)*(2*zeta - 1)*(eta + 1))/4.0;
                    dN_dzeta.row(11) = -(xi*(eta*eta - 1)*(2*zeta - 1)*(xi - 1))/4.0;
                    dN_dzeta.row(12) = -(eta*xi*zeta*(eta - 1)*(xi - 1))/2.0;
                    dN_dzeta.row(13) = -(eta*xi*zeta*(eta - 1)*(xi + 1))/2.0;
                    dN_dzeta.row(14) = -(eta*xi*zeta*(eta + 1)*(xi + 1))/2.0;
                    dN_dzeta.row(15) = -(eta*xi*zeta*(eta + 1)*(xi - 1))/2.0;
                    dN_dzeta.row(16) = -(eta*(xi*xi - 1)*(2*zeta + 1)*(eta - 1))/4.0;
                    dN_dzeta.row(17) = -(xi*(eta*eta - 1)*(2*zeta + 1)*(xi + 1))/4.0;
                    dN_dzeta.row(18) = -(eta*(xi*xi - 1)*(2*zeta + 1)*(eta + 1))/4.0;
                    dN_dzeta.row(19) = -(xi*(eta*eta - 1)*(2*zeta + 1)*(xi - 1))/4.0;
                    dN_dzeta.row(20) = ((eta*eta - 1)*(xi*xi - 1)*(2*zeta - 1))/2.0;
                    dN_dzeta.row(21) = eta*zeta*(xi*xi - 1)*(eta - 1);
                    dN_dzeta.row(22) = xi*zeta*(eta*eta - 1)*(xi + 1);
                    dN_dzeta.row(23) = eta*zeta*(xi*xi - 1)*(eta + 1);
                    dN_dzeta.row(24) = xi*zeta*(eta*eta - 1)*(xi - 1);
                    dN_dzeta.row(25) = ((eta*eta - 1)*(xi*xi - 1)*(2*zeta + 1))/2.0;
                    dN_dzeta.row(26) = -2*zeta*(eta*eta - 1)*(xi*xi - 1);
                    break;
                
                case 64:
                    N.row(0)  = -((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(1)  =  ((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(2)  = -((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(3)  =  ((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(4)  =  ((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(5)  = -((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(6)  =  ((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(7)  = -((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(8)  =  ((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(9) = -((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(10) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(11) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(12) =  ((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(13) = -((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(14) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(15) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(16) =  ((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(17) = -((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(18) = -((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(19) =  ((9*eta)/16.0  + 3/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(20) =  ((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(21) = -((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(22) = -((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(23) =  ((9*eta)/16.0  + 9/16.0)  * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(24) = -((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(25) =  ((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(26) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(27) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(28) = -((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(29) =  ((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(30) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(31) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(32) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(33) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(34) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(35) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 3/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(36) = -((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(37) =  ((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(38) = -((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(39) =  ((9*eta)/16.0  + 3/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(40) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(41) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(42) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(43) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 9/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1/3.0) * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(44) = -((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(45) =  ((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(46) = -((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(47) =  ((9*eta)/16.0  + 9/16.0)  * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1/3.0) * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(48) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(49) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(50) =  ((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(51) = -((27*eta)/16.0 + 27/16.0) * ((9*xi)/16.0  + 3/16.0)  * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(52) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(53) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(54) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(55) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((9*zeta)/16.0  + 9/16.0)  * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1/3.0) * (zeta + 1/3.0);
                    N.row(56) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(57) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(58) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(59) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta - 1/3.0);
                    N.row(60) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(61) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta - 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(62) = -((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi + 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);
                    N.row(63) =  ((27*eta)/16.0 + 27/16.0) * ((27*xi)/16.0 + 27/16.0) * ((27*zeta)/16.0 + 27/16.0) * (eta - 1)   * (eta + 1/3.0) * (xi - 1)   * (xi - 1/3.0) * (zeta - 1)   * (zeta + 1/3.0);

                    Eigen::ArrayXXd xi2 = xi * xi;
                    Eigen::ArrayXXd xi3 = xi * xi * xi;
                    Eigen::ArrayXXd eta2 = eta * eta;
                    Eigen::ArrayXXd eta3 = eta * eta * eta;
                    Eigen::ArrayXXd zeta2 = zeta * zeta;
                    Eigen::ArrayXXd zeta3 = zeta * zeta * zeta;

                    dN_dxi.row(0)  =  ((- 27*xi2 + 18*xi + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(1)  =  ((27*xi2 + 18*xi - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(2)  = -((27*xi2 + 18*xi - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(3)  = -((- 27*xi2 + 18*xi + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(4)  = -((- 27*xi2 + 18*xi + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(5)  = -((27*xi2 + 18*xi - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(6)  =  ((27*xi2 + 18*xi - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(7)  =  ((- 27*xi2 + 18*xi + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(8)  = -(9*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(9) = -(9*(9*xi2 + 2*xi - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(10) = -(9*(27*xi2 + 18*xi - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(11) =  (9*(27*xi2 + 18*xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(12) =  (9*(9*xi2 + 2*xi - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(13) =  (9*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(14) =  (9*(- 27*xi2 + 18*xi + 1)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(15) = -(9*(- 27*xi2 + 18*xi + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(16) = -(9*(- 27*xi2 + 18*xi + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(17) =  (9*(- 27*xi2 + 18*xi + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(18) = -(9*(27*xi2 + 18*xi - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(19) =  (9*(27*xi2 + 18*xi - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(20) =  (9*(27*xi2 + 18*xi - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(21) = -(9*(27*xi2 + 18*xi - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(22) =  (9*(- 27*xi2 + 18*xi + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(23) = -(9*(- 27*xi2 + 18*xi + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(24) =  (9*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(25) =  (9*(9*xi2 + 2*xi - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(26) =  (9*(27*xi2 + 18*xi - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(27) = -(9*(27*xi2 + 18*xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(28) = -(9*(9*xi2 + 2*xi - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(29) = -(9*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(30) = -(9*(- 27*xi2 + 18*xi + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(31) =  (9*(- 27*xi2 + 18*xi + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(32) =  (81*(- 9*xi2 + 2*xi + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(33) =  (81*(9*xi2 + 2*xi - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0 ; 
                    dN_dxi.row(34) = -(81*(9*xi2 + 2*xi - 3)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(35) = -(81*(- 9*xi2 + 2*xi + 3)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(36) =  (81*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(37) =  (81*(9*xi2 + 2*xi - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(38) = -(81*(9*xi2 + 2*xi - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(39) = -(81*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(40) =  (81*(27*xi2 + 18*xi - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(41) = -(81*(27*xi2 + 18*xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(42) =  (81*(27*xi2 + 18*xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(43) = -(81*(27*xi2 + 18*xi - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(44) = -(81*(9*xi2 + 2*xi - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(45) = -(81*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(46) =  (81*(- 9*xi2 + 2*xi + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(47) =  (81*(9*xi2 + 2*xi - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(48) = -(81*(- 27*xi2 + 18*xi + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(49) =  (81*(- 27*xi2 + 18*xi + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(50) = -(81*(- 27*xi2 + 18*xi + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(51) =  (81*(- 27*xi2 + 18*xi + 1)*(- 3*eta3 - eta2 + 3*eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(52) = -(81*(- 9*xi2 + 2*xi + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(53) = -(81*(9*xi2 + 2*xi - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(54) =  (81*(9*xi2 + 2*xi - 3)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(55) =  (81*(- 9*xi2 + 2*xi + 3)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(56) = -(729*(- 9*xi2 + 2*xi + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(57) = -(729*(9*xi2 + 2*xi - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0 ; 
                    dN_dxi.row(58) =  (729*(9*xi2 + 2*xi - 3)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(59) =  (729*(- 9*xi2 + 2*xi + 3)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0 ; 
                    dN_dxi.row(60) =  (729*(- 9*xi2 + 2*xi + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(61) =  (729*(9*xi2 + 2*xi - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(62) = -(729*(9*xi2 + 2*xi - 3)*(- 3*eta3 - eta2 + 3*eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ; 
                    dN_dxi.row(63) = -(729*(- 9*xi2 + 2*xi + 3)*(- 3*eta3 - eta2 + 3*eta + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0 ;

                    dN_deta.row(0)  =  ((- 27*eta2 + 18*eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(1)  = -((- 27*eta2 + 18*eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(2)  = -((27*eta2 + 18*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(3)  =  ((27*eta2 + 18*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(4)  = -((- 27*eta2 + 18*eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(5)  =  ((- 27*eta2 + 18*eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(6)  =  ((27*eta2 + 18*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(7)  = -((27*eta2 + 18*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(8)  = -(9*(- 27*eta2 + 18*eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(9) =  (9*(- 27*eta2 + 18*eta + 1)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(10) =  (9*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(11) =  (9*(9*eta2 + 2*eta - 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(12) =  (9*(27*eta2 + 18*eta - 1)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(13) = -(9*(27*eta2 + 18*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(14) = -(9*(9*eta2 + 2*eta - 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(15) = -(9*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(16) = -(9*(- 27*eta2 + 18*eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(17) =  (9*(- 27*eta2 + 18*eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(18) =  (9*(- 27*eta2 + 18*eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(19) = -(9*(- 27*eta2 + 18*eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(20) =  (9*(27*eta2 + 18*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(21) = -(9*(27*eta2 + 18*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(22) = -(9*(27*eta2 + 18*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(23) =  (9*(27*eta2 + 18*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(24) =  (9*(- 27*eta2 + 18*eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(25) = -(9*(- 27*eta2 + 18*eta + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(26) = -(9*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(27) = -(9*(9*eta2 + 2*eta - 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(28) = -(9*(27*eta2 + 18*eta - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(29) =  (9*(27*eta2 + 18*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(30) =  (9*(9*eta2 + 2*eta - 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(31) =  (9*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(32) =  (81*(- 9*eta2 + 2*eta + 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(33) = -(81*(- 9*eta2 + 2*eta + 3)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(34) = -(81*(9*eta2 + 2*eta - 3)*(9*zeta2 - 9*zeta3 + zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(35) =  (81*(9*eta2 + 2*eta - 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(9*zeta2 - 9*zeta3 + zeta - 1))/4096.0;
                    dN_deta.row(36) =  (81*(- 27*eta2 + 18*eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(37) = -(81*(- 27*eta2 + 18*eta + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(38) =  (81*(- 27*eta2 + 18*eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(39) = -(81*(- 27*eta2 + 18*eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(40) = -(81*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(41) = -(81*(9*eta2 + 2*eta - 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(42) =  (81*(9*eta2 + 2*eta - 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(43) =  (81*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(44) = -(81*(27*eta2 + 18*eta - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(45) =  (81*(27*eta2 + 18*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(46) = -(81*(27*eta2 + 18*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(47) =  (81*(27*eta2 + 18*eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(48) =  (81*(9*eta2 + 2*eta - 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(49) =  (81*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(50) = -(81*(- 9*eta2 + 2*eta + 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(51) = -(81*(9*eta2 + 2*eta - 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(52) = -(81*(- 9*eta2 + 2*eta + 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(53) =  (81*(- 9*eta2 + 2*eta + 3)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(54) =  (81*(9*eta2 + 2*eta - 3)*(zeta - 9*zeta2 - 9*zeta3 + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(55) = -(81*(9*eta2 + 2*eta - 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta - 9*zeta2 - 9*zeta3 + 1))/4096.0;
                    dN_deta.row(56) = -(729*(- 9*eta2 + 2*eta + 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(57) =  (729*(- 9*eta2 + 2*eta + 3)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(58) =  (729*(9*eta2 + 2*eta - 3)*(zeta2 - 3*zeta3 + 3*zeta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_deta.row(59) = -(729*(9*eta2 + 2*eta - 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(zeta2 - 3*zeta3 + 3*zeta - 1))/4096.0;
                    dN_deta.row(60) =  (729*(- 9*eta2 + 2*eta + 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(61) = -(729*(- 9*eta2 + 2*eta + 3)*(- 3*xi3 - xi2 + 3*xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(62) = -(729*(9*eta2 + 2*eta - 3)*(- 3*xi3 - xi2 + 3*xi + 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;
                    dN_deta.row(63) =  (729*(9*eta2 + 2*eta - 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(3*zeta - zeta2 - 3*zeta3 + 1))/4096.0;

                    dN_dzeta.row(0)  =  ((18*zeta - 27*zeta2 + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(1)  = -((18*zeta - 27*zeta2 + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(2)  =  ((18*zeta - 27*zeta2 + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(3)  = -((18*zeta - 27*zeta2 + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(4)  =  ((27*zeta2 + 18*zeta - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(5)  = -((27*zeta2 + 18*zeta - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(6)  =  ((27*zeta2 + 18*zeta - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(7)  = -((27*zeta2 + 18*zeta - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(8)  = -(9*(18*zeta - 27*zeta2 + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(9) =  (9*(18*zeta - 27*zeta2 + 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(10) =  (9*(18*zeta - 27*zeta2 + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(11) = -(9*(18*zeta - 27*zeta2 + 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(12) = -(9*(18*zeta - 27*zeta2 + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(13) =  (9*(18*zeta - 27*zeta2 + 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(14) =  (9*(18*zeta - 27*zeta2 + 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(15) = -(9*(18*zeta - 27*zeta2 + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(16) = -(9*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(17) = -(9*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(18) =  (9*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(19) =  (9*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(20) = -(9*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(21) = -(9*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(22) =  (9*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(23) =  (9*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(24) = -(9*(27*zeta2 + 18*zeta - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(25) =  (9*(27*zeta2 + 18*zeta - 1)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(26) =  (9*(27*zeta2 + 18*zeta - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(27) = -(9*(27*zeta2 + 18*zeta - 1)*(- 9*xi3 - 9*xi2 + xi + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(28) = -(9*(27*zeta2 + 18*zeta - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(29) =  (9*(27*zeta2 + 18*zeta - 1)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(30) =  (9*(27*zeta2 + 18*zeta - 1)*(- 9*xi3 + 9*xi2 + xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(31) = -(9*(27*zeta2 + 18*zeta - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(32) =  (81*(18*zeta - 27*zeta2 + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(33) = -(81*(18*zeta - 27*zeta2 + 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(34) =  (81*(18*zeta - 27*zeta2 + 1)*(- 3*eta3 - eta2 + 3*eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(35) = -(81*(18*zeta - 27*zeta2 + 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(36) =  (81*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(37) = -(81*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(38) = -(81*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(39) =  (81*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 + 9*eta2 + eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(40) = -(81*(2*zeta - 9*zeta2 + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(41) =  (81*(2*zeta - 9*zeta2 + 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(42) =  (81*(9*zeta2 + 2*zeta - 3)*(- 9*xi3 - 9*xi2 + xi + 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(43) = -(81*(9*zeta2 + 2*zeta - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 - 9*xi2 + xi + 1))/4096.0;
                    dN_dzeta.row(44) =  (81*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(45) = -(81*(2*zeta - 9*zeta2 + 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(46) = -(81*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(47) =  (81*(9*zeta2 + 2*zeta - 3)*(- 9*eta3 - 9*eta2 + eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(48) = -(81*(2*zeta - 9*zeta2 + 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(49) =  (81*(2*zeta - 9*zeta2 + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(50) =  (81*(9*zeta2 + 2*zeta - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 9*xi3 + 9*xi2 + xi - 1))/4096.0;
                    dN_dzeta.row(51) = -(81*(9*zeta2 + 2*zeta - 3)*(- 9*xi3 + 9*xi2 + xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(52) =  (81*(27*zeta2 + 18*zeta - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(53) = -(81*(27*zeta2 + 18*zeta - 1)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(54) =  (81*(27*zeta2 + 18*zeta - 1)*(- 3*eta3 - eta2 + 3*eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(55) = -(81*(27*zeta2 + 18*zeta - 1)*(- 3*xi3 + xi2 + 3*xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(56) = -(729*(2*zeta - 9*zeta2 + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(57) =  (729*(2*zeta - 9*zeta2 + 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(58) = -(729*(2*zeta - 9*zeta2 + 3)*(- 3*eta3 - eta2 + 3*eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(59) =  (729*(2*zeta - 9*zeta2 + 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0;
                    dN_dzeta.row(60) = -(729*(9*zeta2 + 2*zeta - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 + xi2 + 3*xi - 1))/4096.0;
                    dN_dzeta.row(61) =  (729*(9*zeta2 + 2*zeta - 3)*(- 3*eta3 + eta2 + 3*eta - 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(62) = -(729*(9*zeta2 + 2*zeta - 3)*(- 3*eta3 - eta2 + 3*eta + 1)*(- 3*xi3 - xi2 + 3*xi + 1))/4096.0;
                    dN_dzeta.row(63) =  (729*(9*zeta2 + 2*zeta - 3)*(- 3*xi3 + xi2 + 3*xi - 1)*(- 3*eta3 - eta2 + 3*eta + 1))/4096.0; 
                    break;
            }

            for(int gp=0; gp<NGP; ++gp){
                gradN_xi.col(3*gp) = dN_dxi.col(gp);
                gradN_xi.col(3*gp+1) = dN_deta.col(gp);
                gradN_xi.col(3*gp+2) = dN_dzeta.col(gp);
            }
    }
    return {N, gradN_xi};
}

