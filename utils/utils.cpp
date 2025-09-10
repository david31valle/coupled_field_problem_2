

#include "utils.hpp"


std::vector<std::vector<double>> compute_gp(int NGP, int PD) {
    std::cout<<"here in compute_gp"<<std::endl;
    std::vector<std::vector<double>> GP;

    switch (PD) {
        case 1:
            if (NGP == 1) {
                std::vector<double> xi = {0.0};
                std::vector<double> w = {2.0};
                GP.push_back(xi);
                GP.push_back(w);
            }
            else if (NGP == 2) {
                std::vector<double> xi = {-sqrt(1.0 / 3.0), sqrt(1.0 / 3.0)};
                std::vector<double> w = {1.0,1.0};
                GP.push_back(xi);
                GP.push_back(w);
            }
            else if (NGP == 3) {
                std::vector<double> xi = {-sqrt(0.6), 0,  sqrt(0.6)};
                std::vector<double> w = { 5.0/9.0 , 8.0/9.0 , 5.0/9.0};
                GP.push_back(xi);
                GP.push_back(w);
            }
            else if (NGP == 4) {
                std::vector<double> xi = {-0.861136311594953 , -0.339981043584856 , 0.339981043584856 , 0.861136311594953};
                std::vector<double> w = { 0.347854845137454 , 0.652145154862546 , 0.652145154862546 , 0.347854845137454 };
                GP.push_back(xi);
                GP.push_back(w);
            }
            else if (NGP == 5) {
                std::vector<double> xi ={ -0.9061798459386640 , -0.5384693101056831 , 0 , 0.5384693101056831 , 0.9061798459386640 };
                std::vector<double> w = { 0.2369268850561891 , 0.4786286704993665 , 0.5688888888888889 , 0.4786286704993665 , 0.2369268850561891 };
                GP.push_back(xi);
                GP.push_back(w);
            }
            break;

        case 2:
            if (NGP == 4) {
                double a = sqrt(1.0/3.0);
                double w1 = 1.0;
                std::vector<double> xi  = { -a, a, a, -a};
                std::vector<double> eta = { -a, -a, a, a};
                std::vector<double> w   = { w1 * w1, w1 * w1, w1 * w1, w1 * w1 };
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(w);
            }
            else if (NGP == 9) {
                double a = sqrt(0.6);
                double  w1 = 5.0/9.0;
                double  w2 = 8.0/9.0;
                std::vector<double> xi  = { -a, 0, a, -a, 0, a, -a, 0, a};
                std::vector<double> eta = { -a, -a, -a, 0, 0, 0, a, a, a};
                std::vector<double> w   = { w1 * w1, w2 * w1, w1 * w1, w2 * w1, w2 * w2, w2 * w1, w1 * w1, w2 * w1, w1 * w1 };
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(w);
            }
            else if (NGP == 16) {
                double a  = 0.861136311594953;
                double b  = 0.339981043584856;
                double w1 = 0.347854845137454;
                double w2 = 0.652145154862546;
                std::vector<double> xi  = { -a ,-b , b , a , -a ,-b , b , a , -a ,-b , b , a , -a ,-b , b , a};
                std::vector<double> eta = { -a ,-a ,-a ,-a, -b ,-b ,-b ,-b ,b , b , b , b ,a , a , a , a};
                std::vector<double> w   = { w1 * w1 , w2 * w1 , w2 * w1 , w1 * w1 ,w1 * w2 , w2 * w2 , w2 * w2 , w1 * w2 ,w1 * w2 , w2 * w2 , w2 * w2 , w1 * w2 ,w1 * w1 , w2 * w1 , w2 * w1 , w1 * w1};
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(w);
            }
            else if (NGP == 25) {
                double a  = 0.9061798459;
                double b  = 0.538469310105683;
                double w1 = 0.236926885056189;
                double w2 = 0.478628670499366;
                double w3 = 0.568888888888889;
                std::vector<double> xi  = {-a , -b , 0 , b , a ,-a , -b , 0 , b , a ,-a , -b , 0 , b , a ,-a , -b , 0 , b , a ,-a , -b , 0 , b , a};
                std::vector<double> eta = {-a ,-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b ,-b ,0 , 0 , 0 , 0 , 0 ,b , b , b , b , b ,a , a , a , a , a };
                std::vector<double> w   = {w1 * w1 , w2 * w1 , w3 * w1 , w2 * w1 , w1 * w1 ,w1 * w2 , w2 * w2 , w3 * w2 , w2 * w2 , w1 * w2 ,w1 * w3 , w2 * w3 , w3 * w3 , w2 * w3 , w1 * w3 ,w1 * w2 , w2 * w2 , w3 * w2 , w2 * w2 , w1 * w2 ,w1 * w1 , w2 * w1 , w3 * w1 , w2 * w1 , w1 * w1};
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(w);
            }
            break;

        case 3:
            if (NGP == 8) {
                double a = sqrt(1.0/3.0);
                double w1 = 1.0;
                std::vector<double> xi   = { -a , a , a ,-a ,-a , a , a ,-a};
                std::vector<double> eta  = { -a ,-a , a , a ,-a ,-a , a , a };
                std::vector<double> zeta = { -a ,-a ,-a ,-a ,a , a , a , a };
                std::vector<double> w    = {w1 * w1 * w1 , w1 * w1 * w1 ,w1 * w1 * w1 , w1 * w1 * w1 ,w1 * w1 * w1 , w1 * w1 * w1 ,w1 * w1 * w1 , w1 * w1 * w1};
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(zeta);
                GP.push_back(w);
            }
            else if (NGP == 27) {
                double a = sqrt(0.6);
                double w1 = 5.0/9.0;
                double w2 = 8.0/9.0;
                std::vector<double> xi  = { -a , 0 , a ,-a , 0 , a ,-a , 0 , a , -a , 0 , a ,-a , 0 , a ,-a , 0 , a , -a , 0 , a ,-a , 0 , a ,-a , 0 , a };
                std::vector<double> eta = {-a ,-a ,-a , 0 , 0 , 0 , a , a , a ,-a ,-a ,-a , 0 , 0 , 0 , a , a , a ,-a ,-a ,-a , 0 , 0 , 0 , a , a , a};
                std::vector<double>zeta = {-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,a , a , a , a , a , a , a , a , a};
                std::vector<double> w   = { w1 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,w1 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,w1 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,w1 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,w1 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,w1 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,w1 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,w1 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,w1 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 };
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(zeta);
                GP.push_back(w);
            }
            else if (NGP == 64) {
                double a = 0.861136311594953;
                double b = 0.339981043584856;
                double w1= 0.347854845137454;
                double w2= 0.6521451875456;
                std::vector<double> xi = {-a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,
                                          -a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,
                                          -a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,
                                          -a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,
                                          -a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a ,-a ,-b , b , a};
                std::vector<double> eta = {-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b , b , b , b , b , a , a , a , a ,
                                           -a ,-a ,-a ,-a ,-b ,-b ,-b ,-b , b , b , b , b , a , a , a , a ,
                                           -a ,-a ,-a ,-a ,-b ,-b ,-b ,-b , b , b , b , b , a , a , a , a ,
                                           -a ,-a ,-a ,-a ,-b ,-b ,-b ,-b , b , b , b , b , a , a , a , a ,
                                           -a ,-a ,-a ,-a ,-b ,-b ,-b ,-b , b , b , b , b , a , a , a , a };
                std::vector<double> zeta {-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,
                                          -b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,
                                          0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                          b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b ,
                                          a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a};
                std::vector<double> w = {w1 * w1 * w1 , w2 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,
                                         w1 * w2 * w1 , w2 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                         w1 * w2 * w1 , w2 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                         w1 * w1 * w1 , w2 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,
                                         w1 * w1 * w2 , w2 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                         w1 * w2 * w2 , w2 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                         w1 * w2 * w2 , w2 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                         w1 * w1 * w2 , w2 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                         w1 * w1 * w2 , w2 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                         w1 * w2 * w2 , w2 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                         w1 * w2 * w2 , w2 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                         w1 * w1 * w2 , w2 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                         w1 * w1 * w1 , w2 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,
                                         w1 * w2 * w1 , w2 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                         w1 * w2 * w1 , w2 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                         w1 * w1 * w1 , w2 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 };
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(zeta);
                GP.push_back(w);
            }
            else if (NGP==125) {
                double a  = 0.9061798459;
                double b  = 0.538469310105683;
                double w1 = 0.236926885056189;
                double w2 = 0.478628670499366;
                double w3 = 0.568888888888889;

                std::vector<double> xi   = {-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,
                                            -a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,
                                            -a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,
                                            -a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,
                                            -a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a ,-a ,-b , 0 , b , a };

                std::vector<double> eta  = {-a ,-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b ,-b , 0 , 0 , 0 , 0 , 0 , b , b , b , b , b , a , a , a , a , a ,
                                            -a ,-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b ,-b , 0 , 0 , 0 , 0 , 0 , b , b , b , b , b , a , a , a , a , a ,
                                            -a ,-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b ,-b , 0 , 0 , 0 , 0 , 0 , b , b , b , b , b , a , a , a , a , a ,
                                            -a ,-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b ,-b , 0 , 0 , 0 , 0 , 0 , b , b , b , b , b , a , a , a , a , a ,
                                            -a ,-a ,-a ,-a ,-a ,-b ,-b ,-b ,-b ,-b , 0 , 0 , 0 , 0 , 0 , b , b , b , b , b , a , a , a , a , a };

                std::vector<double> zeta = {-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,-a ,
                                            -b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,-b ,
                                            0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,
                                            a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a , a ,
                                            b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b , b };

                std::vector<double> w   = { w1 * w1 * w1 , w2 * w1 * w1 , w3 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,
                                            w1 * w2 * w1 , w2 * w2 * w1 , w3 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                            w1 * w3 * w1 , w2 * w3 * w1 , w3 * w3 * w1 , w2 * w3 * w1 , w1 * w3 * w1 ,
                                            w1 * w2 * w1 , w2 * w2 * w1 , w3 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                            w1 * w1 * w1 , w2 * w1 * w1 , w3 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,
                                            w1 * w1 * w2 , w2 * w1 * w2 , w3 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w3 * w2 , w2 * w3 * w2 , w3 * w3 * w2 , w2 * w3 * w2 , w1 * w3 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w1 * w2 , w2 * w1 * w2 , w3 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                            w1 * w1 * w3 , w2 * w1 * w3 , w3 * w1 * w3 , w2 * w1 * w3 , w1 * w1 * w3 ,
                                            w1 * w2 * w3 , w2 * w2 * w3 , w3 * w2 * w3 , w2 * w2 * w3 , w1 * w2 * w3 ,
                                            w1 * w3 * w3 , w2 * w3 * w3 , w3 * w3 * w3 , w2 * w3 * w3 , w1 * w3 * w3 ,
                                            w1 * w2 * w3 , w2 * w2 * w3 , w3 * w2 * w3 , w2 * w2 * w3 , w1 * w2 * w3 ,
                                            w1 * w1 * w3 , w2 * w1 * w3 , w3 * w1 * w3 , w2 * w1 * w3 , w1 * w1 * w3 ,
                                            w1 * w1 * w2 , w2 * w1 * w2 , w3 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w3 * w2 , w2 * w3 * w2 , w3 * w3 * w2 , w2 * w3 * w2 , w1 * w3 * w2 ,
                                            w1 * w2 * w2 , w2 * w2 * w2 , w3 * w2 * w2 , w2 * w2 * w2 , w1 * w2 * w2 ,
                                            w1 * w1 * w2 , w2 * w1 * w2 , w3 * w1 * w2 , w2 * w1 * w2 , w1 * w1 * w2 ,
                                            w1 * w1 * w1 , w2 * w1 * w1 , w3 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 ,
                                            w1 * w2 * w1 , w2 * w2 * w1 , w3 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                            w1 * w3 * w1 , w2 * w3 * w1 , w3 * w3 * w1 , w2 * w3 * w1 , w1 * w3 * w1 ,
                                            w1 * w2 * w1 , w2 * w2 * w1 , w3 * w2 * w1 , w2 * w2 * w1 , w1 * w2 * w1 ,
                                            w1 * w1 * w1 , w2 * w1 * w1 , w3 * w1 * w1 , w2 * w1 * w1 , w1 * w1 * w1 };
                GP.push_back(xi);
                GP.push_back(eta);
                GP.push_back(zeta);
                GP.push_back(w);
            }
            break;
    }
    return GP;
}


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> compute_N_xi_gp(int degree, const std::vector<std::vector<double>>& GP, int PD) {
    std::cout<<"here in compute_N_xi"<<std::endl;
    int NGP = GP[0].size();
    std::vector<double> xi, w, eta, zeta;

    switch (PD) {
        case 1:
            xi = GP[0];
            w = GP[1];
            break;
        case 2:
            xi = GP[0];
            eta = GP[1];
            w = GP[2];
            break;
        case 3:
            xi = GP[0];
            eta = GP[1];
            zeta = GP[2];
            w = GP[3];
            break;
    }

    std::vector<std::vector<double>> N_xi_gp, GradN_xi_gp, GradN_eta_gp, GradN_zeta_gp;

    switch (PD) {
        case 1:
            switch (degree) {
                case 1:
                    N_xi_gp.resize(2, std::vector<double>(NGP));
                    GradN_xi_gp.resize(2, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {
                        N_xi_gp[0][i] = -0.5 * (xi[i] - 1.0);
                        N_xi_gp[1][i] = 0.5 * (xi[i] + 1.0);

                        GradN_xi_gp[0][i] = -0.5;
                        GradN_xi_gp[1][i] = 0.5;
                    }
                    break;

                case 2:
                    N_xi_gp.resize(3, std::vector<double>(NGP));
                    GradN_xi_gp.resize(3, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {
                        N_xi_gp[0][i] = 0.5 * xi[i] * (xi[i] - 1.0);
                        N_xi_gp[1][i] = -1.0 * (xi[i] - 1.0) * (xi[i] + 1.0);
                        N_xi_gp[2][i] = 0.5 * xi[i] * (xi[i] + 1.0);

                        GradN_xi_gp[0][i] = 0.5 * (2.0 * xi[i] - 1.0);  // Fixed
                        GradN_xi_gp[1][i] = -2.0 * xi[i];
                        GradN_xi_gp[2][i] = 0.5 * (2.0 * xi[i] + 1.0);  // Fixed
                    }
                    break;

                case 3:
                    N_xi_gp.resize(4, std::vector<double>(NGP));
                    GradN_xi_gp.resize(4, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {
                        N_xi_gp[0][i] = (-9.0 / 16.0) * (xi[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] - 1.0);
                        N_xi_gp[1][i] = (27.0 / 16.0) * (xi[i] + 1.0) * (xi[i] - 1.0 / 3.0) * (xi[i] - 1.0);
                        N_xi_gp[2][i] = (-27.0 / 16.0) * (xi[i] + 1.0) * (xi[i] + 1.0 / 3.0) * (xi[i] - 1.0);
                        N_xi_gp[3][i] = (9.0 / 16.0) * (xi[i] + 1.0) * (xi[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0);

                        GradN_xi_gp[0][i] = (-9.0 / 16.0) *
                                            ((xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) +
                                             (xi[i] + 1.0 / 3.0) * (xi[i] - 1.0) +
                                             (xi[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0));

                        GradN_xi_gp[1][i] = (27.0 / 16.0) *
                                            ((xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) +
                                             (xi[i] + 1.0) * (xi[i] - 1.0) +
                                             (xi[i] + 1.0) * (xi[i] - 1.0 / 3.0));

                        GradN_xi_gp[2][i] = (-27.0 / 16.0) *
                                            ((xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) +
                                             (xi[i] + 1.0) * (xi[i] - 1.0) +
                                             (xi[i] + 1.0) * (xi[i] + 1.0 / 3.0));

                        GradN_xi_gp[3][i] = (9.0 / 16.0) *
                                            ((xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) +
                                             (xi[i] + 1.0) * (xi[i] - 1.0 / 3.0) +
                                             (xi[i] + 1.0) * (xi[i] + 1.0 / 3.0));
                    }
                    break;

                case 4:
                    N_xi_gp.resize(5, std::vector<double>(NGP));
                    GradN_xi_gp.resize(5, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {
                        N_xi_gp[0][i] = (2.0 / 3.0) * (xi[i] + 0.5) * xi[i] * (xi[i] - 0.5) * (xi[i] - 1.0);
                        N_xi_gp[1][i] = (-8.0 / 3.0) * (xi[i] + 1.0) * xi[i] * (xi[i] - 0.5) * (xi[i] - 1.0);
                        N_xi_gp[2][i] = 4.0 * (xi[i] + 1.0) * (xi[i] + 0.5) * (xi[i] - 0.5) * (xi[i] - 1.0);
                        N_xi_gp[3][i] = (-8.0 / 3.0) * (xi[i] + 1.0) * (xi[i] + 0.5) * xi[i] * (xi[i] - 1.0);
                        N_xi_gp[4][i] = (2.0 / 3.0) * (xi[i] + 1.0) * (xi[i] + 0.5) * xi[i] * (xi[i] - 0.5);

                        GradN_xi_gp[0][i] =
                                (8.0 * std::pow(xi[i], 3)) / 3.0 - 2.0 * std::pow(xi[i], 2) - xi[i] / 3.0 + 1.0 / 6.0;
                        GradN_xi_gp[1][i] =
                                (-32.0 * std::pow(xi[i], 3)) / 3.0 + 4.0 * std::pow(xi[i], 2) + (16.0 * xi[i]) / 3.0 -
                                4.0 / 3.0;
                        GradN_xi_gp[2][i] = 16.0 * std::pow(xi[i], 3) - 10.0 * xi[i];
                        GradN_xi_gp[3][i] =
                                (-32.0 * std::pow(xi[i], 3)) / 3.0 - 4.0 * std::pow(xi[i], 2) + (16.0 * xi[i]) / 3.0 +
                                4.0 / 3.0;
                        GradN_xi_gp[4][i] =
                                (8.0 * std::pow(xi[i], 3)) / 3.0 + 2.0 * std::pow(xi[i], 2) - xi[i] / 3.0 - 1.0 / 6.0;
                    }
                    break;
            }
            break;

        case 2:
            switch (degree) {
                case 1:
                    N_xi_gp.resize(4, std::vector<double>(NGP));
                    GradN_xi_gp.resize(4, std::vector<double>(NGP));
                    GradN_eta_gp.resize(4, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {
                        N_xi_gp[0][i] = 0.25 * (1.0 - xi[i]) * (1.0 - eta[i]);
                        N_xi_gp[1][i] = 0.25 * (1.0 + xi[i]) * (1.0 - eta[i]);
                        N_xi_gp[2][i] = 0.25 * (1.0 + xi[i]) * (1.0 + eta[i]);
                        N_xi_gp[3][i] = 0.25 * (1.0 - xi[i]) * (1.0 + eta[i]);

                        GradN_xi_gp[0][i] = -0.25 * (1.0 - eta[i]);
                        GradN_xi_gp[1][i] = 0.25 * (1.0 - eta[i]);
                        GradN_xi_gp[2][i] = 0.25 * (1.0 + eta[i]);
                        GradN_xi_gp[3][i] = -0.25 * (1.0 + eta[i]);

                        GradN_eta_gp[0][i] = -0.25 * (1.0 - xi[i]);
                        GradN_eta_gp[1][i] = -0.25 * (1.0 + xi[i]);
                        GradN_eta_gp[2][i] = 0.25 * (1.0 + xi[i]);
                        GradN_eta_gp[3][i] = 0.25 * (1.0 - xi[i]);
                    }
                    break;

                case 2 :
                    N_xi_gp.resize(9, std::vector<double>(NGP));
                    GradN_xi_gp.resize(9, std::vector<double>(NGP));
                    GradN_eta_gp.resize(9, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {
                        N_xi_gp[0][i] = 0.25 * (1.0 - xi[i]) * xi[i] * (1.0 - eta[i]) * eta[i];
                        N_xi_gp[1][i] = -0.25 * (1.0 + xi[i]) * xi[i] * (1.0 - eta[i]) * eta[i];
                        N_xi_gp[2][i] = 0.25 * (1.0 + xi[i]) * xi[i] * (1.0 + eta[i]) * eta[i];
                        N_xi_gp[3][i] = -0.25 * (1.0 - xi[i]) * xi[i] * (1.0 + eta[i]) * eta[i];
                        N_xi_gp[4][i] = -0.5 * (1.0 - xi[i]) * (1.0 + xi[i]) * (1.0 - eta[i]) * eta[i];
                        N_xi_gp[5][i] = 0.5 * (1.0 + xi[i]) * xi[i] * (1.0 + eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[6][i] = 0.5 * (1.0 - xi[i]) * (1.0 + xi[i]) * (1.0 + eta[i]) * eta[i];
                        N_xi_gp[7][i] = -0.5 * (1.0 - xi[i]) * xi[i] * (1.0 + eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[8][i] = (1.0 - xi[i]) * (1.0 + xi[i]) * (1.0 + eta[i]) * (1.0 - eta[i]);

                        GradN_xi_gp[0][i] = 0.25 * (1.0 - 2.0 * xi[i]) * (1.0 - eta[i]) * eta[i]; // Fixed
                        GradN_xi_gp[1][i] = -0.25 * (1.0 + 2.0 * xi[i]) * (1.0 - eta[i]) * eta[i]; // Fixed
                        GradN_xi_gp[2][i] = 0.25 * (1.0 + 2.0 * xi[i]) * (1.0 + eta[i]) * eta[i]; // Fixed
                        GradN_xi_gp[3][i] = -0.25 * (1.0 - 2.0 * xi[i]) * (1.0 + eta[i]) * eta[i]; // Fixed
                        GradN_xi_gp[4][i] = xi[i] * eta[i] * (1.0 - eta[i]);
                        GradN_xi_gp[5][i] = 0.50 * (1.0 + 2.0 * xi[i]) * (1.0 - eta[i]) * (1.0 + eta[i]); // Fixed
                        GradN_xi_gp[6][i] = -xi[i] * eta[i] * (1.0 + eta[i]);
                        GradN_xi_gp[7][i] = -0.50 * (1.0 - 2.0 * xi[i]) * (1.0 - eta[i]) * (1.0 + eta[i]); // Fixed
                        GradN_xi_gp[8][i] = -2.00 * xi[i] * (1.0 - eta[i]) * (1.0 + eta[i]);

                        GradN_eta_gp[0][i] = 0.25 * (1.0 - xi[i]) * xi[i] * (1.0 - 2.0 * eta[i]); // Fixed
                        GradN_eta_gp[1][i] = -0.25 * (1.0 + xi[i]) * xi[i] * (1.0 - 2.0 * eta[i]); // Fixed
                        GradN_eta_gp[2][i] = 0.25 * (1.0 + xi[i]) * xi[i] * (1.0 + 2.0 * eta[i]); // Fixed
                        GradN_eta_gp[3][i] = -0.25 * (1.0 - xi[i]) * xi[i] * (1.0 + 2.0 * eta[i]); // Fixed
                        GradN_eta_gp[4][i] = 0.50 * (1.0 - xi[i]) * (1.0 + xi[i]) * (2.0 * eta[i] - 1.0); // Fixed
                        GradN_eta_gp[5][i] = -(1.0 + xi[i]) * xi[i] * eta[i];
                        GradN_eta_gp[6][i] = 0.50 * (1.0 - xi[i]) * (1.0 + xi[i]) * (1.0 + 2.0 * eta[i]); // Fixed
                        GradN_eta_gp[7][i] = (1.0 - xi[i]) * xi[i] * (eta[i]);
                        GradN_eta_gp[8][i] = -2.00 * (1.0 - xi[i]) * (1.0 + xi[i]) * (eta[i]);
                    }
                    break;

                case 3:
                    N_xi_gp.resize(16, std::vector<double>(NGP));
                    GradN_xi_gp.resize(16, std::vector<double>(NGP));
                    GradN_eta_gp.resize(16, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {

                        N_xi_gp[0][i] = (((9.0 * eta[i]) / 16.0) + (3.0 / 16.0)) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);
                        N_xi_gp[1][i] = -((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) *
                                        (xi[i] + 1.0 / 3.0);
                        N_xi_gp[2][i] = ((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        (eta[i] - 1.0 / 3.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) *
                                        (xi[i] + 1.0 / 3.0);
                        N_xi_gp[3][i] = -((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                        (eta[i] - 1.0 / 3.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) *
                                        (xi[i] - 1.0 / 3.0);
                        N_xi_gp[4][i] = -((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);
                        N_xi_gp[5][i] = ((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0);
                        N_xi_gp[6][i] = ((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) *
                                        (xi[i] + 1.0 / 3.0);
                        N_xi_gp[7][i] = -((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) *
                                        (xi[i] + 1.0 / 3.0);
                        N_xi_gp[8][i] = -((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        (eta[i] - 1.0 / 3.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) *
                                        (xi[i] + 1.0 / 3.0);
                        N_xi_gp[9][i] = ((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        (eta[i] - 1.0 / 3.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) *
                                        (xi[i] - 1.0 / 3.0);
                        N_xi_gp[10][i] = ((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);
                        N_xi_gp[11][i] = -((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);
                        N_xi_gp[11][i] =
                                ((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);
                        N_xi_gp[12][i] =
                                -((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0);
                        N_xi_gp[13][i] =
                                ((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0);
                        N_xi_gp[14][i] =
                                -((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);

                        N_xi_gp[15][i] =
                                -((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0);

                        GradN_xi_gp[0][i] = ((-27.0 * std::pow(xi[i], 2.0) + 18.0 * xi[i] + 3.0) *
                                             (-9.0 * std::pow(eta[i], 3.0) + 9.0 * std::pow(eta[i], 2.0) + eta[i] -
                                              1.0)) / 256.0;

                        GradN_xi_gp[1][i] = ((27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] - 1.0) *
                                             (-9.0 * std::pow(eta[i], 3) + 9.0 * std::pow(eta[i], 2) + eta[i] - 1.0)) /
                                            256.0;

                        GradN_xi_gp[2][i] = -((27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] - 1.0) *
                                              (-9.0 * std::pow(eta[i], 3) - 9.0 * std::pow(eta[i], 2) + eta[i] + 1.0)) /
                                            256.0;

                        GradN_xi_gp[3][i] = -((-27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] + 1.0) *
                                              (-9.0 * std::pow(eta[i], 3) - 9.0 * std::pow(eta[i], 2) + eta[i] + 1.0)) /
                                            256.0;

                        GradN_xi_gp[4][i] = -(9.0 * (-9.0 * std::pow(xi[i], 2) + 2.0 * xi[i] + 3.0) *
                                              (-9.0 * std::pow(eta[i], 3) + 9.0 * std::pow(eta[i], 2) + eta[i] - 1.0)) /
                                            256.0;

                        GradN_xi_gp[5][i] = -(9.0 * (9.0 * std::pow(xi[i], 2) - 2.0 * xi[i] - 3.0) *
                                              (-9.0 * std::pow(eta[i], 3) + 9.0 * std::pow(eta[i], 2) + eta[i] - 1.0)) /
                                            256.0;

                        GradN_xi_gp[6][i] = -(9.0 * (27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] - 1.0) *
                                              (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) *
                                              (xi[i] + 1.0 / 3.0));

                        GradN_xi_gp[7][i] = (9.0 * (27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] - 1.0) *
                                             (-3.0 * std::pow(eta[i], 3) - std::pow(eta[i], 2) - eta[i] + 1.0)) / 256.0;

                        GradN_xi_gp[8][i] = (9.0 * (-9.0 * std::pow(xi[i], 2) + 2.0 * xi[i] + 3.0) *
                                             (-9.0 * std::pow(eta[i], 3) - 9.0 * std::pow(eta[i], 2) + eta[i] + 1.0)) /
                                            256.0;

                        GradN_xi_gp[9][i] = (9.0 * (-9.0 * std::pow(xi[i], 2) + 2.0 * xi[i] + 3.0) *
                                             (-9.0 * std::pow(eta[i], 3) - 9.0 * std::pow(eta[i], 2) + eta[i] + 1.0)) /
                                            256.0;

                        GradN_xi_gp[10][i] = (9.0 * (-27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] + 1.0) *
                                              (-3.0 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3.0 * eta[i] + 1.0)) /
                                             256.0;

                        GradN_xi_gp[11][i] = -(9.0 * (-27.0 * std::pow(xi[i], 2) + 18.0 * xi[i] + 1.0) *
                                               (-3.0 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3.0 * eta[i] -
                                                1.0)) / 256.0;

                        GradN_xi_gp[12][i] = (81.0 * (-9.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] + 3.0) *
                                              (-3.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 3.0 * eta[i] -
                                               1.0)) /
                                             256.0;

                        GradN_xi_gp[13][i] = (81.0 * (9.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 3.0) *
                                              (-3.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 3.0 * eta[i] -
                                               1.0)) /
                                             256.0;

                        GradN_xi_gp[14][i] = -(81.0 * (9.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 3.0) *
                                               (-3.0 * std::pow(eta[i], 3.0) - std::pow(eta[i], 2.0) + 3.0 * eta[i] +
                                                1.0)) /
                                             256.0;

                        GradN_xi_gp[15][i] = -(81.0 * (-9.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] + 3.0) *
                                               (-3.0 * std::pow(eta[i], 3.0) - std::pow(eta[i], 2.0) + 3.0 * eta[i] +
                                                1.0)) /
                                             256.0;

                        GradN_eta_gp[0][i] = ((-27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] + 1.0) *
                                              (-9.0 * std::pow(xi[i], 3.0) + 9.0 * std::pow(xi[i], 2.0) + xi[i] -
                                               1.0)) /
                                             256.0;

                        GradN_eta_gp[1][i] = -((-27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] + 1.0) *
                                               (-9.0 * std::pow(xi[i], 3.0) - 9.0 * std::pow(xi[i], 2.0) + xi[i] +
                                                1.0)) /
                                             256.0;

                        GradN_eta_gp[2][i] = -((27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] - 1.0) *
                                               (-9.0 * std::pow(xi[i], 3.0) - 9.0 * std::pow(xi[i], 2.0) + xi[i] +
                                                1.0)) /
                                             256.0;

                        GradN_eta_gp[3][i] = ((27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] - 1.0) *
                                              (-9.0 * std::pow(xi[i], 3.0) + 9.0 * std::pow(xi[i], 2.0) + xi[i] -
                                               1.0)) /
                                             256.0;

                        GradN_eta_gp[4][i] = -(9.0 * (-27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] + 1.0) *
                                               (-3.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 3.0 * xi[i] -
                                                1.0)) /
                                             256.0;

                        GradN_eta_gp[5][i] = (9.0 * (-27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] + 1.0) *
                                              (-3.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 3.0 * xi[i] +
                                               1.0)) /
                                             256.0;

                        GradN_eta_gp[6][i] = (9.0 * (-9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 3.0) *
                                              (-9.0 * std::pow(xi[i], 3.0) - 9.0 * std::pow(xi[i], 2.0) + xi[i] +
                                               1.0)) /
                                             256.0;

                        GradN_eta_gp[7][i] = (9.0 * (9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 3.0) *
                                              (-9.0 * std::pow(xi[i], 3.0) - 9.0 * std::pow(xi[i], 2.0) + xi[i] +
                                               1.0)) /
                                             256.0;

                        GradN_eta_gp[8][i] = (9.0 * (27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] - 1.0) *
                                              (-3.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 3.0 * xi[i] +
                                               1.0)) /
                                             256.0;

                        GradN_eta_gp[9][i] = -(9.0 * (27.0 * std::pow(eta[i], 2.0) + 18.0 * eta[i] - 1.0) *
                                               (-3.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 3.0 * xi[i] -
                                                1.0)) /
                                             256.0;

                        GradN_eta_gp[10][i] = -(9.0 * (9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 3.0) *
                                                (-9.0 * std::pow(xi[i], 3.0) + 9.0 * std::pow(xi[i], 2.0) + xi[i] -
                                                 1.0)) /
                                              256.0;

                        GradN_eta_gp[11][i] = -(9.0 * (-9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 3.0) *
                                                (-9.0 * std::pow(xi[i], 3.0) + 9.0 * std::pow(xi[i], 2.0) + xi[i] -
                                                 1.0)) /
                                              256.0;

                        GradN_eta_gp[12][i] = (81.0 * (-9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 3.0) *
                                               (-3.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 3.0 * xi[i] -
                                                1.0)) /
                                              256.0;

                        GradN_eta_gp[13][i] = -(81.0 * (-9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 3.0) *
                                                (-3.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 3.0 * xi[i] +
                                                 1.0)) /
                                              256.0;

                        GradN_eta_gp[14][i] = -(81.0 * (9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 3.0) *
                                                (-3.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 3.0 * xi[i] +
                                                 1.0)) /
                                              256.0;

                        GradN_eta_gp[15][i] = (81.0 * (9.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 3.0) *
                                               (-3.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 3.0 * xi[i] -
                                                1.0)) /
                                              256.0;
                    }
                        break;

                case 4:
                    N_xi_gp.resize(25, std::vector<double>(NGP)); // Assuming N has 16 rows, adjust as necessary
                    GradN_xi_gp.resize(25, std::vector<double>(NGP));
                    GradN_eta_gp.resize(25, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; ++i) {

                        N_xi_gp[0][i] =
                                (4.0 / 9.0) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) * (0.5 + eta[i]) *
                                (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[1][i] = -(4.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) *
                                        (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[2][i] =
                                (4.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 + eta[i]) *
                                (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]);
                        N_xi_gp[3][i] = -(4.0 / 9.0) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                        (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]);
                        N_xi_gp[4][i] = -(16.0 / 9.0) * (1.0 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                        (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[5][i] = (8.0 / 3.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                        (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[6][i] = (16.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (1.0 - xi[i]) *
                                        (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[7][i] = (16.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) *
                                        (1.0 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[8][i] = -(8.0 / 3.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) *
                                        (1.0 + eta[i]) * (0.5 + eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[9][i] = -(16.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) *
                                        (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[10][i] = -(16.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]);
                        N_xi_gp[11][i] = -(8.0 / 3.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]);
                        N_xi_gp[12][i] = (16.0 / 9.0) * (1.0 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (0.5 - eta[i]);
                        N_xi_gp[13][i] = (16.0 / 9.0) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[14][i] =
                                (8.0 / 3.0) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) * (1.0 + eta[i]) *
                                (0.5 + eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[15][i] = -(16.0 / 9.0) * (0.5 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[16][i] = (64.0 / 9.0) * (1.0 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[17][i] = -(32.0 / 3.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[18][i] = -(64.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[19][i] = (32.0 / 3.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[20][i] = (64.0 / 9.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[21][i] = (32.0 / 3.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[22][i] = -(64.0 / 9.0) * (1.0 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[23][i] = -(32.0 / 3.0) * (1.0 + xi[i]) * (xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);
                        N_xi_gp[24][i] = (16.0) * (1.0 + xi[i]) * (0.5 + xi[i]) * (0.5 - xi[i]) * (1.0 - xi[i]) *
                                         (1.0 + eta[i]) * (0.5 + eta[i]) * (0.5 - eta[i]) * (1.0 - eta[i]);

                        GradN_xi_gp[0][i] =
                                (eta[i] * (-4.0 * std::pow(eta[i], 3.0) + 4.0 * std::pow(eta[i], 2.0) + eta[i] - 1.0) *
                                 (-16.0 * std::pow(xi[i], 3.0) + 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 36.0;

                        GradN_xi_gp[1][i] =
                                (eta[i] * (-4.0 * std::pow(eta[i], 3.0) + 4.0 * std::pow(eta[i], 2.0) + eta[i] - 1.0) *
                                 (-16.0 * std::pow(xi[i], 3.0) - 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 36.0;

                        GradN_xi_gp[2][i] =
                                (eta[i] * (-4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0) *
                                 (-16.0 * std::pow(xi[i], 3.0) - 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 36.0;

                        GradN_xi_gp[3][i] =
                                (eta[i] * (-4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0) *
                                 (-16.0 * std::pow(xi[i], 3.0) + 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 36.0;

                        GradN_xi_gp[4][i] =
                                -(2.0 * eta[i] * (-4.0 * std::pow(eta[i], 3.0) + 4.0 * std::pow(eta[i], 2.0) + eta[i] - 1.0) *
                                  (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[5][i] =
                                -(eta[i] * xi[i] * (8.0 * std::pow(xi[i], 2.0) - 5.0) *
                                  (-4.0 * std::pow(eta[i], 3.0) + 4.0 * std::pow(eta[i], 2.0) + eta[i] - 1.0)) / 3.0;

                        GradN_xi_gp[6][i] =
                                -(2.0 * eta[i] * (-4.0 * std::pow(eta[i], 3.0) + 4.0 * std::pow(eta[i], 2.0) + eta[i] - 1.0) *
                                  (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] + 1.0)) / 9.0;

                        GradN_xi_gp[7][i] =
                                -(2.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0) *
                                  (-16.0 * std::pow(xi[i], 3.0) - 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 9.0;

                        GradN_xi_gp[8][i] =
                                -( ( -16.0 * std::pow(xi[i], 3.0) - 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0 ) *
                                   ( -4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0 ) ) / 6.0;

                        GradN_xi_gp[9][i] =
                                -(2.0 * eta[i] * (-4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0) *
                                  (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] + 1.0)) / 9.0;

                        GradN_xi_gp[10][i] =
                                -(2.0 * eta[i] * (-4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0) *
                                  (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[11][i] =
                                -(eta[i] * xi[i] * (8.0 * std::pow(xi[i], 2.0) - 5.0) *
                                  (-4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0)) / 3.0;

                        GradN_xi_gp[12][i] =
                                -(2.0 * eta[i] * (-4.0 * std::pow(eta[i], 3.0) - 4.0 * std::pow(eta[i], 2.0) + eta[i] + 1.0) *
                                  (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[13][i] =
                                -(2.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0) *
                                  (-16.0 * std::pow(xi[i], 3.0) + 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[14][i] =
                                -((4.0 * std::pow(eta[i], 4.0) - 5.0 * std::pow(eta[i], 2.0) + 1.0) *
                                  (-16.0 * std::pow(xi[i], 3.0) + 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 6.0;

                        GradN_xi_gp[15][i] =
                                -(2.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0) *
                                  (-16.0 * std::pow(xi[i], 3.0) + 12.0 * std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[16][i] =
                                (16.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0) *
                                 (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[17][i] =
                                (8.0 * eta[i] * xi[i] * (8.0 * std::pow(xi[i], 2.0) - 5.0) *
                                 (-2.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0)) / 3.0;

                        GradN_xi_gp[18][i] =
                                (16.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) + std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0) *
                                 (-8.0 * std::pow(xi[i], 3.0) - 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] + 1.0)) / 9.0;

                        GradN_xi_gp[19][i] =
                                (4.0 * (4.0 * std::pow(eta[i], 4.0) - 5.0 * std::pow(eta[i], 2.0) + 1.0) *
                                 (-8.0 * std::pow(xi[i], 3.0) - 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] + 1.0)) / 6.0;

                        GradN_xi_gp[20][i] =
                                (16.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) - std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0) *
                                 (-8.0 * std::pow(xi[i], 3.0) - 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] + 1.0)) / 9.0;

                        GradN_xi_gp[21][i] =
                                (8.0 * eta[i] * xi[i] * (8.0 * std::pow(xi[i], 2.0) - 5.0) *
                                 (-2.0 * std::pow(eta[i], 3.0) - std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0)) / 3.0;

                        GradN_xi_gp[22][i] =
                                (16.0 * eta[i] * (-2.0 * std::pow(eta[i], 3.0) - std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0) *
                                 (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[23][i] =
                                (4.0 * (4.0 * std::pow(eta[i], 4.0) - 5.0 * std::pow(eta[i], 2.0) + 1.0) *
                                 (-8.0 * std::pow(xi[i], 3.0) + 3.0 * std::pow(xi[i], 2.0) + 4.0 * xi[i] - 1.0)) / 9.0;

                        GradN_xi_gp[24][i] =
                                2.0 * xi[i] * (8.0 * std::pow(xi[i], 2.0) - 5.0) *
                                (4.0 * std::pow(eta[i], 4.0) - 5.0 * std::pow(eta[i], 2.0) + 1.0);

                        GradN_eta_gp[0][i] =
                                (xi[i] * (-4.0 * std::pow(xi[i], 3.0) + 4.0 * std::pow(xi[i], 2.0) + xi[i] - 1.0) *
                                 (-16.0 * std::pow(eta[i], 3.0) + 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0)) / 36.0;

                        GradN_eta_gp[1][i] =
                                (xi[i] * (-4.0 * std::pow(xi[i], 3.0) - 4.0 * std::pow(xi[i], 2.0) + xi[i] + 1.0) *
                                 (-16.0 * std::pow(eta[i], 3.0) + 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0)) / 36.0;

                        GradN_eta_gp[2][i] =
                                (xi[i] * (-4.0 * std::pow(xi[i], 3.0) - 4.0 * std::pow(xi[i], 2.0) + xi[i] + 1.0) *
                                 (-16.0 * std::pow(eta[i], 3.0) - 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0)) / 36.0;

                        GradN_eta_gp[3][i] =
                                (xi[i] * (-4.0 * std::pow(xi[i], 3.0) + 4.0 * std::pow(xi[i], 2.0) + xi[i] - 1.0) *
                                 (-16.0 * std::pow(eta[i], 3.0) - 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0)) / 36.0;

                        GradN_eta_gp[4][i] =
                                -(2.0 * xi[i] * (-4.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 3.0 * xi[i] - 1.0) *
                                  (-16.0 * std::pow(eta[i], 3.0) + 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0)) / 9.0;

                        GradN_eta_gp[5][i] =
                                -((4.0 * std::pow(xi[i], 4.0) - 5.0 * std::pow(xi[i], 2.0) + 1.0) *
                                  (-16.0 * std::pow(eta[i], 3.0) + 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0)) / 6.0;

                        GradN_eta_gp[6][i] =
                                -(2.0 * xi[i] * (-16.0 * std::pow(eta[i], 3.0) + 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] - 1.0) *
                                  (-2.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 9.0;

                        GradN_eta_gp[7][i] =
                                -(2.0 * xi[i] * (-4.0 * std::pow(xi[i], 3.0) - 4.0 * std::pow(xi[i], 2.0) + xi[i] + 1.0) *
                                  (-8.0 * std::pow(eta[i], 3.0) + 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] - 1.0)) / 9.0;

                        GradN_eta_gp[8][i] =
                                -(eta[i] * xi[i] * (8.0 * std::pow(eta[i], 2.0) - 5.0) *
                                  (-4.0 * std::pow(xi[i], 3.0) - 4.0 * std::pow(xi[i], 2.0) + xi[i] + 1.0)) / 3.0;

                        GradN_eta_gp[9][i] =
                                -(2.0 * xi[i] * (-4.0 * std::pow(xi[i], 3.0) - 4.0 * std::pow(xi[i], 2.0) + xi[i] + 1.0) *
                                  (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] + 1.0)) / 9.0;

                        GradN_eta_gp[10][i] =
                                -(2.0 * xi[i] * (-2.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0) *
                                  (-16.0 * std::pow(eta[i], 3.0) - 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0)) / 9.0;

                        GradN_eta_gp[11][i] =
                                -((4.0 * std::pow(xi[i], 4.0) - 5.0 * std::pow(xi[i], 2.0) + 1.0) *
                                  (-16.0 * std::pow(eta[i], 3.0) - 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0)) / 6.0;

                        GradN_eta_gp[12][i] =
                                -(2.0 * xi[i] * (-2.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0) *
                                  (-16.0 * std::pow(eta[i], 3.0) - 12.0 * std::pow(eta[i], 2.0) + 2.0 * eta[i] + 1.0)) / 9.0;

                        GradN_eta_gp[13][i] =
                                -(2.0 * xi[i] * (-4.0 * std::pow(xi[i], 3.0) + 4.0 * std::pow(xi[i], 2.0) + xi[i] - 1.0) *
                                  (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] + 1.0)) / 9.0;

                        GradN_eta_gp[14][i] =
                                -(eta[i] * xi[i] * (8.0 * std::pow(eta[i], 2.0) - 5.0) *
                                  (-4.0 * std::pow(xi[i], 3.0) + 4.0 * std::pow(xi[i], 2.0) + xi[i] - 1.0)) / 3.0;

                        GradN_eta_gp[15][i] =
                                -(2.0 * xi[i] * (-4.0 * std::pow(xi[i], 3.0) + 4.0 * std::pow(xi[i], 2.0) + xi[i] - 1.0) *
                                  (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] - 1.0)) / 9.0;

                        GradN_eta_gp[16][i] =
                                (16.0 * xi[i] * (-2.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0) *
                                 (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] + 1.0)) / 9.0;

                        GradN_eta_gp[17][i] =
                                (4.0 * (4.0 * std::pow(xi[i], 4.0) - 5.0 * std::pow(xi[i], 2.0) + 1.0) *
                                 (-8.0 * std::pow(eta[i], 3.0) + 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] - 1.0)) / 6.0;

                        GradN_eta_gp[18][i] =
                                (16.0 * xi[i] * (-8.0 * std::pow(eta[i], 3.0) + 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] - 1.0) *
                                 (-2.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 9.0;

                        GradN_eta_gp[19][i] =
                                (8.0 * eta[i] * xi[i] * (8.0 * std::pow(eta[i], 2.0) - 5.0) *
                                 (-2.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 3.0;

                        GradN_eta_gp[20][i] =
                                (16.0 * xi[i] * (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] + 1.0) *
                                 (-2.0 * std::pow(xi[i], 3.0) - std::pow(xi[i], 2.0) + 2.0 * xi[i] + 1.0)) / 9.0;

                        GradN_eta_gp[21][i] =
                                (4.0 * (4.0 * std::pow(xi[i], 4.0) - 5.0 * std::pow(xi[i], 2.0) + 1.0) *
                                 (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] + 1.0)) / 3.0;

                        GradN_eta_gp[22][i] =
                                (16.0 * xi[i] * (-8.0 * std::pow(eta[i], 3.0) - 3.0 * std::pow(eta[i], 2.0) + 4.0 * eta[i] + 1.0) *
                                 (-2.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 9.0;

                        GradN_eta_gp[23][i] =
                                (8.0 * eta[i] * xi[i] * (8.0 * std::pow(eta[i], 2.0) - 5.0) *
                                 (-2.0 * std::pow(xi[i], 3.0) + std::pow(xi[i], 2.0) + 2.0 * xi[i] - 1.0)) / 3.0;

                        GradN_eta_gp[24][i] =
                                2.0 * eta[i] * (8.0 * std::pow(eta[i], 2.0) - 5.0) *
                                (4.0 * std::pow(xi[i], 4.0) - 5.0 * std::pow(xi[i], 2.0) + 1.0);

                    }
                    break;
            }
            break;

        case 3 :
            switch (degree) {
                case 1 :
                    N_xi_gp.assign(8, std::vector<double>(NGP, 0.0));
                    GradN_xi_gp.assign(8, std::vector<double>(NGP, 0.0));
                    GradN_eta_gp.assign(8, std::vector<double>(NGP, 0.0));
                    GradN_zeta_gp.assign(8, std::vector<double>(NGP, 0.0));

                    for (int i = 0; i < NGP; i++) {
                        N_xi_gp[0][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 - eta[i]) * (1.0 - zeta[i]);
                        N_xi_gp[1][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 - eta[i]) * (1.0 - zeta[i]);
                        N_xi_gp[2][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + eta[i]) * (1.0 - zeta[i]);
                        N_xi_gp[2][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + eta[i]) * (1.0 - zeta[i]);
                        N_xi_gp[3][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 + eta[i]) * (1.0 - zeta[i]);
                        N_xi_gp[4][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 - eta[i]) * (1.0 + zeta[i]);
                        N_xi_gp[5][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 - eta[i]) * (1.0 + zeta[i]);
                        N_xi_gp[6][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + eta[i]) * (1.0 + zeta[i]);
                        N_xi_gp[7][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 + eta[i]) * (1.0 + zeta[i]);

                        GradN_xi_gp[0][i] = (-1.0 / 8.0) * (1.0 - eta[i]) * (1.0 - zeta[i]);
                        GradN_xi_gp[1][i] = (1.0 / 8.0) * (1.0 - eta[i]) * (1.0 - zeta[i]);
                        GradN_xi_gp[2][i] = (1.0 / 8.0) * (1.0 + eta[i]) * (1.0 - zeta[i]);
                        GradN_xi_gp[3][i] = (-1.0 / 8.0) * (1.0 + eta[i]) * (1.0 - zeta[i]);
                        GradN_xi_gp[4][i] = (-1.0 / 8.0) * (1.0 - eta[i]) * (1.0 + zeta[i]);
                        GradN_xi_gp[5][i] = (1.0 / 8.0) * (1.0 - eta[i]) * (1.0 + zeta[i]);
                        GradN_xi_gp[6][i] = (1.0 / 8.0) * (1.0 + eta[i]) * (1.0 + zeta[i]);
                        GradN_xi_gp[7][i] = (-1.0 / 8.0) * (1.0 + eta[i]) * (1.0 + zeta[i]);

                        GradN_eta_gp[0][i] = (-1.0 / 8.0) * (1.0 - xi[i]) * (1.0 - zeta[i]);
                        GradN_eta_gp[1][i] = (-1.0 / 8.0) * (1.0 + xi[i]) * (1.0 - zeta[i]);
                        GradN_eta_gp[2][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 - zeta[i]);
                        GradN_eta_gp[3][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 - zeta[i]);
                        GradN_eta_gp[4][i] = (-1.0 / 8.0) * (1.0 - xi[i]) * (1.0 + zeta[i]);
                        GradN_eta_gp[5][i] = (-1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + zeta[i]);
                        GradN_eta_gp[6][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + zeta[i]);
                        GradN_eta_gp[7][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 + zeta[i]);

                        GradN_zeta_gp[0][i] = (-1.0 / 8.0) * (1.0 - xi[i]) * (1.0 - eta[i]);
                        GradN_zeta_gp[1][i] = (-1.0 / 8.0) * (1.0 + xi[i]) * (1.0 - eta[i]);
                        GradN_zeta_gp[2][i] = (-1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + eta[i]);
                        GradN_zeta_gp[3][i] = (-1.0 / 8.0) * (1.0 - xi[i]) * (1.0 + eta[i]);
                        GradN_zeta_gp[4][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 - eta[i]);
                        GradN_zeta_gp[5][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 - eta[i]);
                        GradN_zeta_gp[6][i] = (1.0 / 8.0) * (1.0 + xi[i]) * (1.0 + eta[i]);
                        GradN_zeta_gp[7][i] = (1.0 / 8.0) * (1.0 - xi[i]) * (1.0 + eta[i]);

                    }
                    break;
                case 2 :
                    N_xi_gp.resize(27, std::vector<double>(NGP));
                    GradN_xi_gp.resize(27, std::vector<double>(NGP));
                    GradN_eta_gp.resize(27, std::vector<double>(NGP));
                    GradN_zeta_gp.resize(27, std::vector<double>(NGP));

                    for (int i = 0; i < (NGP); i++) {
                        N_xi_gp[0][i] = (1.0 / 8.0) * xi[i] * (xi[i] - 1.0) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[1][i] = (1.0 / 8.0) * xi[i] * (xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[2][i] = (1.0 / 8.0) * xi[i] * (xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[3][i] = (1.0 / 8.0) * xi[i] * (xi[i] - 1.0) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[4][i] = (1.0 / 8.0) * xi[i] * (xi[i] - 1.0) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] + 1.0);
                        N_xi_gp[5][i] = (1.0 / 8.0) * xi[i] * (xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] + 1.0);
                        N_xi_gp[6][i] = (1.0 / 8.0) * xi[i] * (xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);
                        N_xi_gp[7][i] = (1.0 / 8.0) * xi[i] * (xi[i] - 1.0) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);

                        N_xi_gp[8][i] = -(1.0 / 4.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[9][i] = -(1.0 / 4.0) * xi[i] * (xi[i] + 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[10][i] = -(1.0 / 4.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[11][i] = -(1.0 / 4.0) * xi[i] * (xi[i] - 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[12][i] = -(1.0 / 4.0) * xi[i] * (xi[i] - 1.0) * eta[i] * (eta[i] - 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[13][i] = -(1.0 / 4.0) * xi[i] * (xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[14][i] = -(1.0 / 4.0) * xi[i] * (xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[15][i] = -(1.0 / 4.0) * xi[i] * (xi[i] - 1.0) * eta[i] * (eta[i] + 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[16][i] = -(1.0 / 4.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] + 1.0);
                        N_xi_gp[17][i] = -(1.0 / 4.0) * xi[i] * (xi[i] + 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);
                        N_xi_gp[18][i] = -(1.0 / 4.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);
                        N_xi_gp[19][i] = -(1.0 / 4.0) * xi[i] * (xi[i] - 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);

                        N_xi_gp[20][i] = (1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);
                        N_xi_gp[21][i] = (1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[22][i] = (1.0 / 2.0) * xi[i] * (xi[i] + 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[23][i] = (1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[24][i] = (1.0 / 2.0) * xi[i] * (xi[i] - 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);
                        N_xi_gp[25][i] = (1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);

                        N_xi_gp[26][i] = -(1.0) * (xi[i] - 1.0) * (xi[i] + 1.0) * (eta[i] - 1.0) * (eta[i] + 1.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0);


                        GradN_xi_gp[0][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i]) * (eta[i] - 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_xi_gp[1][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] + 1.0) * (eta[i] - 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_xi_gp[2][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] + 1.0) * (eta[i] + 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_xi_gp[3][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] - 1.0) * (eta[i] + 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_xi_gp[4][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] - 1.0) * (eta[i] - 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_xi_gp[5][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] + 1.0) * (eta[i] - 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_xi_gp[6][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] + 1.0) * (eta[i] + 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_xi_gp[7][i] =
                                (eta[i] * zeta[i] * (2.0 * xi[i] - 1.0) * (eta[i] + 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_xi_gp[8][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i]) * (eta[i]) * (eta[i] - 1.0) * zeta[i] * (zeta[i] - 1.0);

                        GradN_xi_gp[9][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] + 1.0) * (std::pow(eta[i], 2.0) - 1.0) * zeta[i] * (zeta[i] - 1.0);

                        GradN_xi_gp[10][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i]) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] - 1.0);

                        GradN_xi_gp[11][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] - 1.0) * (std::pow(eta[i], 2.0) - 1.0) * zeta[i] * (zeta[i] - 1.0);

                        GradN_xi_gp[12][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] - 1.0) * eta[i] * (eta[i] - 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[13][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] + 1.0) * eta[i] * (eta[i] - 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[14][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] + 1.0) * eta[i] * (eta[i] + 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[15][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] - 1.0) * eta[i] * (eta[i] + 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[16][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i]) * eta[i] * (eta[i] - 1.0) * zeta[i] * (zeta[i] + 1.0);

                        GradN_xi_gp[17][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] + 1.0) * (std::pow(eta[i], 2.0) - 1.0) * zeta[i] * (zeta[i] + 1.0);

                        GradN_xi_gp[18][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i]) * eta[i] * (eta[i] + 1.0) * zeta[i] * (zeta[i] + 1.0);

                        GradN_xi_gp[19][i] =
                                -(1.0 / 4.0) * (2.0 * xi[i] - 1.0) * (std::pow(eta[i], 2.0) - 1.0) * zeta[i] * (zeta[i] + 1.0);

                        GradN_xi_gp[20][i] =
                                xi[i] * zeta[i] * (std::pow(eta[i], 2.0) - 1.0) * (zeta[i] - 1.0);

                        GradN_xi_gp[21][i] =
                                eta[i] * xi[i] * (eta[i] - 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[22][i] =
                                (1.0 / 2.0) * (2.0 * xi[i] + 1.0) * (std::pow(eta[i], 2.0) - 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[23][i] =
                                eta[i] * xi[i] * (eta[i] + 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_xi_gp[24][i] =
                                ((std::pow(eta[i], 2.0) - 1.0) * (2.0 * xi[i] - 1.0) * (std::pow(zeta[i], 2.0) - 1.0)) / 2.0;

                        GradN_xi_gp[25][i] =
                                xi[i] * zeta[i] * (std::pow(eta[i], 2.0) - 1.0) * (zeta[i] + 1.0);

                        GradN_xi_gp[26][i] =
                                -2.0 * xi[i] * (std::pow(eta[i], 2.0) - 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_eta_gp[0][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] - 1.0) * (xi[i] - 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_eta_gp[1][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] - 1.0) * (xi[i] + 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_eta_gp[2][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] + 1.0) * (xi[i] + 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_eta_gp[3][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] + 1.0) * (xi[i] - 1.0) * (zeta[i] - 1.0)) / 8.0;

                        GradN_eta_gp[4][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] - 1.0) * (xi[i] - 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_eta_gp[5][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] - 1.0) * (xi[i] + 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_eta_gp[6][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] + 1.0) * (xi[i] + 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_eta_gp[7][i] =
                                (xi[i] * zeta[i] * (2.0 * eta[i] + 1.0) * (xi[i] - 1.0) * (zeta[i] + 1.0)) / 8.0;

                        GradN_eta_gp[8][i] =
                                -(zeta[i] * (2.0 * eta[i] - 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (zeta[i] - 1.0)) / 4.0;

                        GradN_eta_gp[9][i] =
                                -(eta[i] * xi[i] * zeta[i] * (xi[i] + 1.0) * (zeta[i] - 1.0)) / 2.0;

                        GradN_eta_gp[10][i] =
                                -(zeta[i] * (2.0 * eta[i] + 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (zeta[i] - 1.0)) / 4.0;

                        GradN_eta_gp[11][i] =
                                -(eta[i] * xi[i] * zeta[i] * (xi[i] - 1.0) * (zeta[i] - 1.0)) / 4.0;

                        GradN_eta_gp[12][i] =
                                -(xi[i] * (2.0 * eta[i] - 1.0) * (std::pow(zeta[i], 2.0) - 1.0) * (xi[i] - 1.0)) / 4.0;

                        GradN_eta_gp[13][i] =
                                -(xi[i] * (2.0 * eta[i] - 1.0) * (std::pow(zeta[i], 2.0) - 1.0) * (xi[i] + 1.0)) / 4.0;

                        GradN_eta_gp[14][i] =
                                -(xi[i] * (2.0 * eta[i] + 1.0) * (std::pow(zeta[i], 2.0) - 1.0) * (xi[i] + 1.0)) / 4.0;

                        GradN_eta_gp[15][i] =
                                -(xi[i] * (2.0 * eta[i] + 1.0) * (std::pow(zeta[i], 2.0) - 1.0) * (xi[i] - 1.0)) / 4.0;

                        GradN_eta_gp[16][i] =
                                -(zeta[i] * (2.0 * eta[i] - 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (zeta[i] + 1.0)) / 4.0;

                        GradN_eta_gp[17][i] =
                                -(eta[i] * xi[i] * zeta[i] * (xi[i] + 1.0) * (zeta[i] + 1.0)) / 2.0;

                        GradN_eta_gp[18][i] =
                                -(zeta[i] * (2.0 * eta[i] + 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (zeta[i] + 1.0)) / 4.0;

                        GradN_eta_gp[19][i] =
                                -(eta[i] * xi[i] * zeta[i] * (xi[i] - 1.0) * (zeta[i] + 1.0)) / 4.0;

                        GradN_eta_gp[20][i] =
                                xi[i] * zeta[i] * (std::pow(xi[i], 2.0) - 1.0) * (zeta[i] - 1.0);

                        GradN_eta_gp[21][i] =
                                ((2.0 * eta[i] - 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (std::pow(zeta[i], 2.0) - 1.0)) / 2.0;

                        GradN_eta_gp[22][i] =
                                eta[i] * xi[i] * (std::pow(zeta[i], 2.0) - 1.0) * (xi[i] + 1.0);

                        GradN_eta_gp[23][i] =
                                ((2.0 * eta[i] + 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (std::pow(zeta[i], 2.0) - 1.0)) / 2.0;

                        GradN_eta_gp[24][i] =
                                eta[i] * xi[i] * (std::pow(zeta[i], 2.0) - 1.0) * (xi[i] - 1.0);

                        GradN_eta_gp[25][i] =
                                eta[i] * zeta[i] * (std::pow(xi[i], 2.0) - 1.0) * (zeta[i] + 1.0);

                        GradN_eta_gp[26][i] =
                                -2.0 * eta[i] * (std::pow(xi[i], 2.0) - 1.0) * (std::pow(zeta[i], 2.0) - 1.0);

                        GradN_zeta_gp[0][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] - 1.0) * (eta[i] - 1.0) * (xi[i] - 1.0)) / 8.0;

                        GradN_zeta_gp[1][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] - 1.0) * (eta[i] - 1.0) * (xi[i] + 1.0)) / 8.0;

                        GradN_zeta_gp[2][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] - 1.0) * (eta[i] + 1.0) * (xi[i] + 1.0)) / 8.0;

                        GradN_zeta_gp[3][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] - 1.0) * (eta[i] + 1.0) * (xi[i] - 1.0)) / 8.0;

                        GradN_zeta_gp[4][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] + 1.0) * (eta[i] - 1.0) * (xi[i] - 1.0)) / 8.0;

                        GradN_zeta_gp[5][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] + 1.0) * (eta[i] - 1.0) * (xi[i] + 1.0)) / 8.0;

                        GradN_zeta_gp[6][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] + 1.0) * (eta[i] + 1.0) * (xi[i] + 1.0)) / 8.0;

                        GradN_zeta_gp[7][i] =
                                (eta[i] * xi[i] * (2.0 * zeta[i] + 1.0) * (eta[i] + 1.0) * (xi[i] - 1.0)) / 8.0;

                        GradN_zeta_gp[8][i] =
                                -(eta[i] * (std::pow(xi[i], 2.0) - 1.0) * (2.0 * zeta[i] - 1.0) * (eta[i] - 1.0)) / 4.0;

                        GradN_zeta_gp[9][i] =
                                -(xi[i] * (std::pow(eta[i], 2.0) - 1.0) * (2.0 * zeta[i] - 1.0) * (xi[i] + 1.0)) / 4.0;

                        GradN_zeta_gp[10][i] =
                                -(eta[i] * (std::pow(xi[i], 2.0) - 1.0) * (2.0 * zeta[i] - 1.0) * (eta[i] + 1.0)) / 4.0;

                        GradN_zeta_gp[11][i] =
                                -(xi[i] * (std::pow(eta[i], 2.0) - 1.0) * (2.0 * zeta[i] - 1.0) * (xi[i] - 1.0)) / 4.0;

                        GradN_zeta_gp[12][i] =
                                -(eta[i] * xi[i] * zeta[i] * (eta[i] - 1.0) * (xi[i] - 1.0)) / 2.0;

                        GradN_zeta_gp[13][i] =
                                -(eta[i] * xi[i] * zeta[i] * (eta[i] - 1.0) * (xi[i] + 1.0)) / 2.0;

                        GradN_zeta_gp[14][i] =
                                -(eta[i] * xi[i] * zeta[i] * (eta[i] + 1.0) * (xi[i] + 1.0)) / 2.0;

                        GradN_zeta_gp[15][i] =
                                -(eta[i] * xi[i] * zeta[i] * (eta[i] + 1.0) * (xi[i] - 1.0)) / 2.0;

                        GradN_zeta_gp[16][i] =
                                -(eta[i] * (std::pow(xi[i], 2.0) - 1.0) * (2.0 * zeta[i] + 1.0) * (eta[i] - 1.0)) / 4.0;

                        GradN_zeta_gp[17][i] =
                                -(xi[i] * (std::pow(eta[i], 2.0) - 1.0) * (2.0 * zeta[i] + 1.0) * (xi[i] + 1.0)) / 4.0;

                        GradN_zeta_gp[18][i] =
                                -(eta[i] * (std::pow(xi[i], 2.0) - 1.0) * (2.0 * zeta[i] + 1.0) * (eta[i] + 1.0)) / 4.0;

                        GradN_zeta_gp[19][i] =
                                -(xi[i] * (std::pow(eta[i], 2.0) - 1.0) * (2.0 * zeta[i] + 1.0) * (xi[i] - 1.0)) / 4.0;

                        GradN_zeta_gp[20][i] =
                                ((std::pow(eta[i], 2.0) - 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (2.0 * zeta[i] - 1.0)) / 2.0;

                        GradN_zeta_gp[21][i] =
                                eta[i] * zeta[i] * (std::pow(xi[i], 2.0) - 1.0) * (eta[i] - 1.0);

                        GradN_zeta_gp[22][i] =
                                xi[i] * zeta[i] * (std::pow(eta[i], 2.0) - 1.0) * (xi[i] + 1.0);

                        GradN_zeta_gp[23][i] =
                                eta[i] * zeta[i] * (std::pow(xi[i], 2.0) - 1.0) * (eta[i] + 1.0);

                        GradN_zeta_gp[24][i] =
                                xi[i] * zeta[i] * (std::pow(eta[i], 2.0) - 1.0) * (xi[i] - 1.0);

                        GradN_zeta_gp[25][i] =
                                ((std::pow(eta[i], 2.0) - 1.0) * (std::pow(xi[i], 2.0) - 1.0) * (2.0 * zeta[i] + 1.0)) / 2.0;

                        GradN_zeta_gp[26][i] =
                                -2.0 * zeta[i] * (std::pow(eta[i], 2.0) - 1.0) * (std::pow(xi[i], 2.0) - 1.0);
                    }
                    break;
                case 3:
                    N_xi_gp.resize(64, std::vector<double>(NGP)); // Assuming N has 16 rows, adjust as necessary
                    GradN_xi_gp.resize(64, std::vector<double>(NGP));
                    GradN_eta_gp.resize(64, std::vector<double>(NGP));
                    GradN_zeta_gp.resize(64, std::vector<double>(NGP));

                    for (int i = 0; i < 64; i++) {
                        N_xi_gp[0][i] = -((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                        (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[1][i] = ((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                        (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                        (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[2][i] = -((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                        (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) *
                                        (zeta[i] - 1.0 / 3.0) * (zeta[i] + 1.0 / 3.0);

                        N_xi_gp[3][i] = ((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                        (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) *
                                        (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[4][i] = ((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) *
                                        (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) *
                                        (zeta[i] - 1.0 / 3.0) * (zeta[i] + 1.0 / 3.0);

                        N_xi_gp[5][i] = -((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) *
                                        (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) *
                                        (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[6][i] = ((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                        (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) *
                                        (zeta[i] - 1.0 / 3.0) * (zeta[i] + 1.0 / 3.0);

                        N_xi_gp[7][i] = -((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                        (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) *
                                        (zeta[i] - 1.0 / 3.0) * (zeta[i] + 1.0 / 3.0);

                        N_xi_gp[8][i] = ((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) *
                                        (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) *
                                        (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[9][i] = -((9.0 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                        ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) *
                                        (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) *
                                        (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[10][i] = -((27.0 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9.0 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) *
                                         (eta[i] - 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[13][i] = -((9.0 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27.0 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9.0 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);

                        N_xi_gp[14][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[15][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[16][i] = ((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[17][i] = -((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[18][i] = -((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[19][i] = ((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[20][i] = ((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[21][i] = -((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[22][i] = -((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[23][i] = ((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[24][i] = -((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[25][i] = ((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[26][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[27][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[28][i] = -((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) *
                                         (zeta[i] - 1.0 / 3.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[29][i] = ((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) *
                                         (zeta[i] - 1.0 / 3.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[30][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[31][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[32][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[33][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[34][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[35][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 3.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[36][i] = -((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[37][i] = ((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[38][i] = -((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[39][i] = ((9 * eta[i]) / 16.0 + 3.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[40][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[41][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[42][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[43][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 9.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0 / 3.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[44][i] = -((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[45][i] = ((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[46][i] = -((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[47][i] = ((9 * eta[i]) / 16.0 + 9.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0 / 3.0) *
                                         (eta[i] + 1.0 / 3.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[48][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[49][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[50][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[51][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((9 * xi[i]) / 16.0 + 3.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[52][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[53][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[54][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[55][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((9 * zeta[i]) / 16.0 + 9.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0 / 3.0) *
                                         (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[56][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[57][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[58][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[59][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 3.0);
                        N_xi_gp[60][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[61][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[62][i] = -((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] + 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);
                        N_xi_gp[63][i] = ((27 * eta[i]) / 16.0 + 27.0 / 16.0) * ((27 * xi[i]) / 16.0 + 27.0 / 16.0) *
                                         ((27 * zeta[i]) / 16.0 + 27.0 / 16.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 3.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 3.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 3.0);

                        GradN_xi_gp[0][i] = ((-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                             (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                             (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            4096.0;
                        GradN_xi_gp[1][i] = ((27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                             (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                             (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            4096.0;
                        GradN_xi_gp[2][i] = -((27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            4096.0;
                        GradN_xi_gp[3][i] = -((-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            4096.0;
                        GradN_xi_gp[4][i] = -((-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                            4096.0;
                        GradN_xi_gp[5][i] = -((27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                            4096.0;
                        GradN_xi_gp[6][i] = ((27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                             (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                             (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                            4096.0;
                        GradN_xi_gp[7][i] = ((-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                             (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                             (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                            4096.0;
                        GradN_xi_gp[8][i] = -(9 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            4096.0;
                        GradN_xi_gp[9][i] = -(9 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            4096.0;
                        GradN_xi_gp[10][i] = -(9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[11][i] = (9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[12][i] = (9 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[13][i] = (9 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[14][i] = (9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[15][i] = -(9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[16][i] = -(9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[17][i] = (9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[18][i] = -(9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[19][i] = (9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[20][i] = (9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[21][i] = -(9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[22][i] = (9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[23][i] = -(9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[24][i] = (9 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[25][i] = (9 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[26][i] = (9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[27][i] = -(9 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[28][i] = -(9 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[29][i] = -(9 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[30][i] = -(9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[31][i] = (9 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[32][i] = (81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[33][i] = (81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[34][i] = -(81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[35][i] = -(81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[36][i] = (81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[37][i] = (81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[38][i] = -(81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[39][i] = -(81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[40][i] = (81 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[41][i] = -(81 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[42][i] = (81 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[43][i] = -(81 * (27 * std::pow(xi[i], 2) + 18 * xi[i] - 1.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[44][i] = -(81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[45][i] = -(81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[46][i] = (81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[47][i] = (81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[48][i] = -(81 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[49][i] = (81 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[50][i] = -(81 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[51][i] = (81 * (-27 * std::pow(xi[i], 2) + 18 * xi[i] + 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[52][i] = -(81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[53][i] = -(81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[54][i] = (81 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[55][i] = (81 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[56][i] = -(729 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[57][i] = -(729 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_xi_gp[58][i] = (729 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[59][i] = (729 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                              (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[60][i] = (729 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[61][i] = (729 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                              (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                              (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[62][i] = -(729 * (9 * std::pow(xi[i], 2) + 2 * xi[i] - 3.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_xi_gp[63][i] = -(729 * (-9 * std::pow(xi[i], 2) + 2 * xi[i] + 3.0) *
                                               (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;

                        GradN_eta_gp[0][i] = ((-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                              (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_eta_gp[1][i] = -((-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_eta_gp[2][i] = -((27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_eta_gp[3][i] = ((27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                              (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_eta_gp[4][i] = -((-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_eta_gp[5][i] = ((-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                              (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_eta_gp[6][i] = ((27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                              (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                              (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_eta_gp[7][i] = -((27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                             4096.0;
                        GradN_eta_gp[8][i] = -(9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             4096.0;
                        GradN_eta_gp[9][i] = (9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                              (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                             4096.0;
                        GradN_eta_gp[10][i] = (9 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[11][i] = (9 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[12][i] = (9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[13][i] = -(9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[14][i] = -(9 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[15][i] = -(9 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[16][i] = -(9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[17][i] = (9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[18][i] = (9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[19][i] = -(9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[20][i] = (9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[21][i] = -(9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[22][i] = -(9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[23][i] = (9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[24][i] = (9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[25][i] = -(9 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                                (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[26][i] = -(9 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[27][i] = -(9 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[28][i] = -(9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                                (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[29][i] = (9 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[30][i] = (9 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[31][i] = (9 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[32][i] = (81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[33][i] = -(81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[34][i] = -(81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[35][i] = (81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[36][i] = (81 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[37][i] = -(81 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[38][i] = (81 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[39][i] = -(81 * (-27 * std::pow(eta[i], 2) + 18 * eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[40][i] = -(81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[41][i] = -(81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[42][i] = (81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[43][i] = (81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[44][i] = -(81 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[45][i] = (81 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[46][i] = -(81 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[47][i] = (81 * (27 * std::pow(eta[i], 2) + 18 * eta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[48][i] = (81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[49][i] = (81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[50][i] = -(81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[51][i] = -(81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[52][i] = -(81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[53][i] = (81 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[54][i] = (81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[55][i] = -(81 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (zeta[i] - 9 * std::pow(zeta[i], 2) - 9 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[56][i] = -(729 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[57][i] = (729 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[58][i] = (729 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[59][i] = -(729 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 3 * zeta[i] - 1.0)) /
                                              4096.0;
                        GradN_eta_gp[60][i] = (729 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[61][i] = -(729 * (-9 * std::pow(eta[i], 2) + 2 * eta[i] + 3.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[62][i] = -(729 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0) *
                                                (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;
                        GradN_eta_gp[63][i] = (729 * (9 * std::pow(eta[i], 2) + 2 * eta[i] - 3.0) *
                                               (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                               (3 * zeta[i] - std::pow(zeta[i], 2) - 3 * std::pow(zeta[i], 3) + 1.0)) /
                                              4096.0;

                        GradN_zeta_gp[0][i] = ((18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[1][i] = -((18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[2][i] = ((18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[3][i] = -((18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[4][i] = ((27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[5][i] = -((27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[6][i] = ((27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                               (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[7][i] = -((27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[8][i] = -(9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[9][i] = (9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                               (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                              4096.0;
                        GradN_zeta_gp[10][i] = (9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[11][i] = -(9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                 (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[12][i] = -(9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                 (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[13][i] = (9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[14][i] = (9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[15][i] = -(9 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[16][i] = -(9 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[17][i] = -(9 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[18][i] = (9 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[19][i] = (9 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[20][i] = -(9 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[21][i] = -(9 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[22][i] = (9 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[23][i] = (9 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[24][i] = -(9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                 (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[25][i] = (9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[26][i] = (9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[27][i] = -(9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[28][i] = -(9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                 (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[29][i] = (9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[30][i] = (9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[31][i] = -(9 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[32][i] = (81 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[33][i] = -(81 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[34][i] = (81 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[35][i] = -(81 * (18 * zeta[i] - 27 * std::pow(zeta[i], 2) + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[36][i] = (81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[37][i] = -(81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[38][i] = -(81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[39][i] = (81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-9 * std::pow(eta[i], 3) + 9 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[40][i] = -(81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[41][i] = (81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[42][i] = (81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[43][i] = -(81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-9 * std::pow(xi[i], 3) - 9 * std::pow(xi[i], 2) + xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[44][i] = (81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[45][i] = -(81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[46][i] = -(81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[47][i] = (81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-9 * std::pow(eta[i], 3) - 9 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[48][i] = -(81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[49][i] = (81 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[50][i] = (81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[51][i] = -(81 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-9 * std::pow(xi[i], 3) + 9 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[52][i] = (81 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[53][i] = -(81 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[54][i] = (81 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[55][i] = -(81 * (27 * std::pow(zeta[i], 2) + 18 * zeta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[56][i] = -(729 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[57][i] = (729 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[58][i] = -(729 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[59][i] = (729 * (2 * zeta[i] - 9 * std::pow(zeta[i], 2) + 3.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[60][i] = -(729 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                 (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[61][i] = (729 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-3 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 3 * eta[i] - 1.0) *
                                                (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[62][i] = -(729 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                 (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0) *
                                                 (-3 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 3 * xi[i] + 1.0)) /
                                               4096.0;
                        GradN_zeta_gp[63][i] = (729 * (9 * std::pow(zeta[i], 2) + 2 * zeta[i] - 3.0) *
                                                (-3 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 3 * xi[i] - 1.0) *
                                                (-3 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 3 * eta[i] + 1.0)) /
                                               4096.0;

                    }
                    break;

                case 4:
                    N_xi_gp.resize(125, std::vector<double>(NGP));
                    GradN_xi_gp.resize(125, std::vector<double>(NGP));
                    GradN_eta_gp.resize(125, std::vector<double>(NGP));
                    GradN_zeta_gp.resize(125, std::vector<double>(NGP));

                    for (int i = 0; i < NGP; i++) {
                        N_xi_gp[0][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                        (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[1][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                        (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[2][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                        (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                        (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[3][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                        (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                        (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[4][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                        (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[5][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                        (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[6][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                        (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                        (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[7][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                        ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                        (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                        (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[8][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                        ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                        (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                        (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[9][i] = eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * (4 * xi[i] + 4.0) *
                                        ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                        (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                        (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[10][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[11][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[12][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[13][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[14][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[15][i] = eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[16][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[17][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[18][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[19][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[20][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[21][i] =
                                eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[22][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[23][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[24][i] =
                                eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[25][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[26][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[27][i] =
                                eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) *
                                (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) *
                                (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[28][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[29][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[30][i] =
                                eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[31][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[32][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[33][i] = eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[34][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[35][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[36][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[37][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[38][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[39][i] = eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[40][i] = -eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[41][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[42][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[43][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[44][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[45][i] = -eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[46][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[47][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[48][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[49][i] = -eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[50][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[51][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[52][i] =
                                zeta[i] * (4 * eta[i] + 4.0) * (4 * xi[i] + 4.0) * ((2 * zeta[i]) / 3.0 + 1.0 / 3.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[53][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[54][i] = -eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[55][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[56][i] =
                                -eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[57][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[58][i] = -eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[59][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[60][i] =
                                -eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[61][i] =
                                eta[i] * ((2 * eta[i]) / 3.0 + 1.0 / 3.0) * (4 * xi[i] + 4.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[62][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[63][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[64][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[65][i] =
                                -eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[66][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[67][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[68][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[69][i] =
                                -eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[70][i] =
                                xi[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 2.0 / 3.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[71][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[72][i] = -eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[73][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[74][i] =
                                -eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[75][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[76][i] = -eta[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[77][i] = eta[i] * xi[i] * zeta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                         (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[78][i] =
                                -eta[i] * xi[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[79][i] =
                                eta[i] * ((2 * eta[i]) / 3.0 + 2.0 / 3.0) * (4 * xi[i] + 4.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[80][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[81][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[82][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[83][i] =
                                -eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[84][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[85][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[86][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[87][i] =
                                -eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[88][i] =
                                xi[i] * (4 * eta[i] + 4.0) * ((2 * xi[i]) / 3.0 + 1.0 / 3.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[89][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[90][i] = -eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[91][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[92][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[93][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[94][i] = -eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[95][i] = eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[96][i] = -xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                         ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[97][i] =
                                zeta[i] * (4 * eta[i] + 4.0) * (4 * xi[i] + 4.0) * ((2 * zeta[i]) / 3.0 + 2.0 / 3.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0 / 2.0) *
                                (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[98][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                         ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                         (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                         (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[99][i] = eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                         ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                         (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                         (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[100][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[101][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                          (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[102][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[103][i] = eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) *
                                          (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[104][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[105][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                          (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[106][i] =
                                eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[107][i] =
                                -eta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[108][i] =
                                eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[109][i] =
                                -xi[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[110][i] =
                                eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[111][i] =
                                -eta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[112][i] =
                                eta[i] * xi[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                (4 * zeta[i] + 4.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[113][i] =
                                -xi[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * (4 * zeta[i] + 4.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[114][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[115][i] = eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                          (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[116][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[117][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                          (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[118][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] + 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[119][i] = eta[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) * (4 * xi[i] + 4.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) *
                                          (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[120][i] = -eta[i] * xi[i] * zeta[i] * ((8 * eta[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * xi[i]) / 3.0 + 8.0 / 3.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                          (eta[i] - 1.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                          (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[121][i] = xi[i] * zeta[i] * (4 * eta[i] + 4.0) * ((8 * xi[i]) / 3.0 + 8.0 / 3.0) *
                                          ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) * (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) *
                                          (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) * (zeta[i] - 1.0) *
                                          (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[122][i] =
                                -zeta[i] * (4 * eta[i] + 4.0) * (4 * xi[i] + 4.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0);
                        N_xi_gp[123][i] =
                                -zeta[i] * (4 * eta[i] + 4.0) * (4 * xi[i] + 4.0) * ((8 * zeta[i]) / 3.0 + 8.0 / 3.0) *
                                (eta[i] - 1.0) * (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) *
                                (xi[i] - 1.0 / 2.0) * (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] + 1.0 / 2.0);
                        N_xi_gp[124][i] =
                                (4 * eta[i] + 4.0) * (4 * xi[i] + 4.0) * (4 * zeta[i] + 4.0) * (eta[i] - 1.0) *
                                (eta[i] - 1.0 / 2.0) * (eta[i] + 1.0 / 2.0) * (xi[i] - 1.0) * (xi[i] - 1.0 / 2.0) *
                                (xi[i] + 1.0 / 2.0) * (zeta[i] - 1.0) * (zeta[i] - 1.0 / 2.0) * (zeta[i] + 1.0 / 2.0);


                        GradN_xi_gp[0][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                            216;
                        GradN_xi_gp[1][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                            216;
                        GradN_xi_gp[2][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                            216;
                        GradN_xi_gp[3][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                            216;
                        GradN_xi_gp[4][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                            216;
                        GradN_xi_gp[5][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                            216;
                        GradN_xi_gp[6][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                            216;
                        GradN_xi_gp[7][i] = -(eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                            216;
                        GradN_xi_gp[8][i] = (eta[i] * zeta[i] *
                                             (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                             (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                             (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) / 27;
                        GradN_xi_gp[9][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                             (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                             (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                            18;
                        GradN_xi_gp[10][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[11][i] = (eta[i] * zeta[i] *
                                              (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[12][i] = (zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             36.0;
                        GradN_xi_gp[13][i] = (eta[i] * zeta[i] *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[14][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[15][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             18;
                        GradN_xi_gp[16][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[17][i] = (eta[i] * zeta[i] *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[18][i] = (zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             36.0;
                        GradN_xi_gp[19][i] = (eta[i] * zeta[i] *
                                              (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[20][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[21][i] = (eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             36.0;
                        GradN_xi_gp[22][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                              (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[23][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[24][i] = (eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             36.0;
                        GradN_xi_gp[25][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                              (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[26][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[27][i] = (eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             36.0;
                        GradN_xi_gp[28][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                              (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[29][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[30][i] = (eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             36.0;
                        GradN_xi_gp[31][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                              (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[32][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[33][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                             18;
                        GradN_xi_gp[34][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[35][i] = (eta[i] * zeta[i] *
                                              (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[36][i] = (zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             36.0;
                        GradN_xi_gp[37][i] = (eta[i] * zeta[i] *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[38][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[39][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                             18;
                        GradN_xi_gp[40][i] = (eta[i] * zeta[i] *
                                              (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[41][i] = (eta[i] * zeta[i] *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[42][i] = (zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             36.0;
                        GradN_xi_gp[43][i] = (eta[i] * zeta[i] *
                                              (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                              (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                              (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[44][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[45][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[46][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[47][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             9;
                        GradN_xi_gp[48][i] = -(8 * eta[i] * zeta[i] *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[49][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 9;
                        GradN_xi_gp[50][i] = -(8 * eta[i] * zeta[i] *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[51][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[52][i] = -(xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                             3.0;
                        GradN_xi_gp[53][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[54][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[55][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[56][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                  (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[57][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[58][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             9;
                        GradN_xi_gp[59][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[60][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                  (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[61][i] = -(eta[i] * xi[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0)) /
                                             3.0;
                        GradN_xi_gp[62][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[63][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             9;
                        GradN_xi_gp[64][i] = -(8 * eta[i] * zeta[i] *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[65][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                  (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[66][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[67][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             9;
                        GradN_xi_gp[68][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[69][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                  (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[70][i] = -((4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-16 * std::pow(xi[i], 3) - 12 * std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) /
                                             6;
                        GradN_xi_gp[71][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[72][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[73][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[74][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                  (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[75][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[76][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             9;
                        GradN_xi_gp[77][i] = -(8 * eta[i] * zeta[i] *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[78][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                  (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[79][i] = -(eta[i] * xi[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0)) /
                                             3.0;
                        GradN_xi_gp[80][i] = -(8 * eta[i] * zeta[i] *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[81][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[82][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[83][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                  (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[84][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[85][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             9;
                        GradN_xi_gp[86][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                             27;
                        GradN_xi_gp[87][i] =
                                -(2 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                  (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[88][i] = -((4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-16 * std::pow(xi[i], 3) + 12 * std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) /
                                             6;
                        GradN_xi_gp[89][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[90][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                             9;
                        GradN_xi_gp[91][i] = -(8 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[92][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             9;
                        GradN_xi_gp[93][i] = -(8 * eta[i] * zeta[i] *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                             27;
                        GradN_xi_gp[94][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 9;
                        GradN_xi_gp[95][i] = -(8 * eta[i] * zeta[i] *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[96][i] = -(2 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[97][i] = -(xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                             3.0;
                        GradN_xi_gp[98][i] = (64 * eta[i] * zeta[i] *
                                              (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                              (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                             27;
                        GradN_xi_gp[99][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                              (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                              (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                             9;
                        GradN_xi_gp[100][i] = (64 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                              27;
                        GradN_xi_gp[101][i] =
                                (16 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                 (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[102][i] = (64 * eta[i] * zeta[i] *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                              27;
                        GradN_xi_gp[103][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 9;
                        GradN_xi_gp[104][i] = (64 * eta[i] * zeta[i] *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                              27;
                        GradN_xi_gp[105][i] =
                                (16 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                 (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[106][i] =
                                (16 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[107][i] = (8 * eta[i] * xi[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) /
                                              3.0;
                        GradN_xi_gp[108][i] =
                                (16 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[109][i] = (4 * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) /
                                              3.0;
                        GradN_xi_gp[110][i] =
                                (16 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0)) / 9;
                        GradN_xi_gp[111][i] = (8 * eta[i] * xi[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) /
                                              3.0;
                        GradN_xi_gp[112][i] =
                                (16 * eta[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) / 9;
                        GradN_xi_gp[113][i] = (4 * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0)) /
                                              3.0;
                        GradN_xi_gp[114][i] = (64 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_xi_gp[115][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_xi_gp[116][i] = (64 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_xi_gp[117][i] =
                                (16 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0) *
                                 (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) / 9;
                        GradN_xi_gp[118][i] = (64 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) - 3 * std::pow(xi[i], 2) + 4 * xi[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_xi_gp[119][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_xi_gp[120][i] = (64 * eta[i] * zeta[i] *
                                               (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                               (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_xi_gp[121][i] =
                                (16 * zeta[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (-8 * std::pow(xi[i], 3) + 3 * std::pow(xi[i], 2) + 4 * xi[i] - 1.0) *
                                 (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) / 9;
                        GradN_xi_gp[122][i] = (8 * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                              3.0;
                        GradN_xi_gp[123][i] = (8 * xi[i] * zeta[i] * (8 * std::pow(xi[i], 2) - 5) *
                                               (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                               (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              3.0;
                        GradN_xi_gp[124][i] = 2 * xi[i] * (8 * std::pow(xi[i], 2) - 5) *
                                              (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                              (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0);


                        GradN_eta_gp[0][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                  (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 216;
                        GradN_eta_gp[1][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                  (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 216;
                        GradN_eta_gp[2][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                  (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 216;
                        GradN_eta_gp[3][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                  (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 216;
                        GradN_eta_gp[4][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                  (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 216;
                        GradN_eta_gp[5][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                  (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 216;
                        GradN_eta_gp[6][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                  (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 216;
                        GradN_eta_gp[7][i] =
                                -(xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                  (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 216;
                        GradN_eta_gp[8][i] =
                                (xi[i] * zeta[i] * (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[9][i] = (zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                              (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                              (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                               1.0)) / 36.0;
                        GradN_eta_gp[10][i] = (xi[i] * zeta[i] *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                1.0) *
                                               (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[11][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                 (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[12][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                               (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              18;
                        GradN_eta_gp[13][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                 (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[14][i] = (xi[i] * zeta[i] *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                1.0) *
                                               (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[15][i] = (zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                               (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                1.0)) / 36.0;
                        GradN_eta_gp[16][i] =
                                (xi[i] * zeta[i] * (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[17][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                 (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[18][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                               (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              18;
                        GradN_eta_gp[19][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                 (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[20][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[21][i] = (xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                1.0)) / 36.0;
                        GradN_eta_gp[22][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_eta_gp[23][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[24][i] = (xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                1.0)) / 36.0;
                        GradN_eta_gp[25][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_eta_gp[26][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[27][i] = (xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                1.0)) / 36.0;
                        GradN_eta_gp[28][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_eta_gp[29][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[30][i] = (xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                               (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                1.0)) / 36.0;
                        GradN_eta_gp[31][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_eta_gp[32][i] =
                                (xi[i] * zeta[i] * (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                 (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[33][i] = (zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                1.0)) / 36.0;
                        GradN_eta_gp[34][i] = (xi[i] * zeta[i] *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                1.0) *
                                               (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[35][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                 (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[36][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                               (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                              18;
                        GradN_eta_gp[37][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                 (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[38][i] = (xi[i] * zeta[i] *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                1.0) *
                                               (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[39][i] = (zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                               (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                1.0)) / 36.0;
                        GradN_eta_gp[40][i] =
                                (xi[i] * zeta[i] * (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                 (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[41][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                 (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 27;
                        GradN_eta_gp[42][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                               (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                               (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                              18;
                        GradN_eta_gp[43][i] =
                                (xi[i] * zeta[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                 (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 27;
                        GradN_eta_gp[44][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 27;
                        GradN_eta_gp[45][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 9;
                        GradN_eta_gp[46][i] = -(8 * xi[i] * zeta[i] *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[47][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[48][i] = -(8 * xi[i] * zeta[i] *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[49][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 9;
                        GradN_eta_gp[50][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 27;
                        GradN_eta_gp[51][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              9;
                        GradN_eta_gp[52][i] = -(eta[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + zeta[i] - 1.0)) /
                                              3;
                        GradN_eta_gp[53][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0)) / 27;
                        GradN_eta_gp[54][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0)) / 9;
                        GradN_eta_gp[55][i] = -(8 * xi[i] * zeta[i] *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[56][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                  (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[57][i] = -(8 * xi[i] * zeta[i] *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[58][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_eta_gp[59][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[60][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                  (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) / 9;
                        GradN_eta_gp[61][i] = -((4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-16 * std::pow(eta[i], 3) + 12 * std::pow(eta[i], 2) + 2 * eta[i] -
                                                 1.0)) / 6;
                        GradN_eta_gp[62][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 27;
                        GradN_eta_gp[63][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                              9;
                        GradN_eta_gp[64][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 27;
                        GradN_eta_gp[65][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 9;
                        GradN_eta_gp[66][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[67][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_eta_gp[68][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[69][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 9;
                        GradN_eta_gp[70][i] = -(eta[i] * xi[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0)) / 3;
                        GradN_eta_gp[71][i] = -(8 * xi[i] * zeta[i] *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[72][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0)) / 9;
                        GradN_eta_gp[73][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0)) / 27;
                        GradN_eta_gp[74][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                  (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) / 9;
                        GradN_eta_gp[75][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[76][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_eta_gp[77][i] = -(8 * xi[i] * zeta[i] *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[78][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                  (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[79][i] = -((4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-16 * std::pow(eta[i], 3) - 12 * std::pow(eta[i], 2) + 2 * eta[i] +
                                                 1.0)) / 6;
                        GradN_eta_gp[80][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 27;
                        GradN_eta_gp[81][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                              9;
                        GradN_eta_gp[82][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 27;
                        GradN_eta_gp[83][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 9;
                        GradN_eta_gp[84][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[85][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_eta_gp[86][i] = -(8 * xi[i] * zeta[i] *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                              27;
                        GradN_eta_gp[87][i] =
                                -(2 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                  (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 9;
                        GradN_eta_gp[88][i] = -(eta[i] * xi[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0)) / 3;
                        GradN_eta_gp[89][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 27;
                        GradN_eta_gp[90][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 9;
                        GradN_eta_gp[91][i] = -(8 * xi[i] * zeta[i] *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[92][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[93][i] = -(8 * xi[i] * zeta[i] *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[94][i] = -(2 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 9;
                        GradN_eta_gp[95][i] = -(8 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 27;
                        GradN_eta_gp[96][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                              9;
                        GradN_eta_gp[97][i] = -(eta[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (zeta[i] - 4 * std::pow(zeta[i], 2) - 4 * std::pow(zeta[i], 3) + 1.0)) /
                                              3;
                        GradN_eta_gp[98][i] = (64 * xi[i] * zeta[i] *
                                               (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                1.0)) / 27;
                        GradN_eta_gp[99][i] = (16 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                               (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                               (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                1.0)) / 9;
                        GradN_eta_gp[100][i] = (64 * xi[i] * zeta[i] *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[101][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[102][i] = (64 * xi[i] * zeta[i] *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 27;
                        GradN_eta_gp[103][i] = (16 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 9;
                        GradN_eta_gp[104][i] = (64 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 27;
                        GradN_eta_gp[105][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                               9;
                        GradN_eta_gp[106][i] =
                                (16 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0)) / 9;
                        GradN_eta_gp[107][i] = (4 * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0)) / 3;
                        GradN_eta_gp[108][i] =
                                (16 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] - 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[109][i] = (8 * eta[i] * xi[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 3;
                        GradN_eta_gp[110][i] =
                                (16 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_eta_gp[111][i] = (4 * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0)) / 3;
                        GradN_eta_gp[112][i] =
                                (16 * xi[i] * (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] + 1.0)) / 9;
                        GradN_eta_gp[113][i] = (8 * eta[i] * xi[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0) *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 3;
                        GradN_eta_gp[114][i] = (64 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               27;
                        GradN_eta_gp[115][i] = (16 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               9;
                        GradN_eta_gp[116][i] = (64 * xi[i] * zeta[i] *
                                                (-8 * std::pow(eta[i], 3) + 3 * std::pow(eta[i], 2) + 4 * eta[i] -
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               27;
                        GradN_eta_gp[117][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               9;
                        GradN_eta_gp[118][i] = (64 * xi[i] * zeta[i] *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               27;
                        GradN_eta_gp[119][i] = (16 * zeta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               9;
                        GradN_eta_gp[120][i] = (64 * xi[i] * zeta[i] *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (-8 * std::pow(eta[i], 3) - 3 * std::pow(eta[i], 2) + 4 * eta[i] +
                                                 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               27;
                        GradN_eta_gp[121][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               9;
                        GradN_eta_gp[122][i] = (8 * eta[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) /
                                               3;
                        GradN_eta_gp[123][i] = (8 * eta[i] * zeta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                                (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (2 * zeta[i] - std::pow(zeta[i], 2) - 2 * std::pow(zeta[i], 3) + 1.0)) /
                                               3;
                        GradN_eta_gp[124][i] = 2 * eta[i] * (8 * std::pow(eta[i], 2) - 5) *
                                               (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                               (4 * std::pow(zeta[i], 4) - 5 * std::pow(zeta[i], 2) + 1.0);


                        GradN_zeta_gp[0][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                  (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 216;
                        GradN_zeta_gp[1][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                  (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 216;
                        GradN_zeta_gp[2][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                  (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 216;
                        GradN_zeta_gp[3][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                  (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 216;
                        GradN_zeta_gp[4][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                  (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 216;
                        GradN_zeta_gp[5][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                  (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 216;
                        GradN_zeta_gp[6][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                  (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                  (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 216;
                        GradN_zeta_gp[7][i] =
                                -(eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                  (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                  (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 216;
                        GradN_zeta_gp[8][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[9][i] = (eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                               (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                               (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                1.0)) / 36.0;
                        GradN_zeta_gp[10][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[11][i] =
                                (eta[i] * xi[i] * (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[12][i] = (xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[13][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[14][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[15][i] = (eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[16][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[17][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[18][i] = (xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[19][i] =
                                (eta[i] * xi[i] * (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[20][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[21][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0)) / 18;
                        GradN_zeta_gp[22][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[23][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[24][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0)) / 18;
                        GradN_zeta_gp[25][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[26][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[27][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0)) / 18;
                        GradN_zeta_gp[28][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[29][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] - 1.0)) / 27;
                        GradN_zeta_gp[30][i] = (eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0)) / 18;
                        GradN_zeta_gp[31][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[32][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[33][i] = (eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[34][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[35][i] =
                                (eta[i] * xi[i] * (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[36][i] = (xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[37][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[38][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[39][i] = (eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[40][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[41][i] =
                                (eta[i] * xi[i] * (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[42][i] = (xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                 1.0)) / 36.0;
                        GradN_zeta_gp[43][i] =
                                (eta[i] * xi[i] * (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 1.0)) / 27;
                        GradN_zeta_gp[44][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[45][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[46][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[47][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[48][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[49][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[50][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[51][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[52][i] = -((4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) + 2 * zeta[i] -
                                                  1.0)) / 6;
                        GradN_zeta_gp[53][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[54][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[55][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[56][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_zeta_gp[57][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[58][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[59][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[60][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 9;
                        GradN_zeta_gp[61][i] = -(eta[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-4 * std::pow(eta[i], 3) + 4 * std::pow(eta[i], 2) + eta[i] - 1.0)) /
                                               3;
                        GradN_zeta_gp[62][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[63][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[64][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[65][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) /
                                               9;
                        GradN_zeta_gp[66][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[67][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[68][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[69][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0)) / 9;
                        GradN_zeta_gp[70][i] = -(xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-4 * std::pow(xi[i], 3) - 4 * std::pow(xi[i], 2) + xi[i] + 1.0)) / 3;
                        GradN_zeta_gp[71][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[72][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[73][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[74][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 9;
                        GradN_zeta_gp[75][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[76][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[77][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[78][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_zeta_gp[79][i] = -(eta[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-4 * std::pow(eta[i], 3) - 4 * std::pow(eta[i], 2) + eta[i] + 1.0)) /
                                               3;
                        GradN_zeta_gp[80][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[81][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[82][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[83][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0)) / 9;
                        GradN_zeta_gp[84][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[85][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[86][i] = -(8 * eta[i] * xi[i] *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[87][i] = -(4 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) /
                                               9;
                        GradN_zeta_gp[88][i] = -(xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-4 * std::pow(xi[i], 3) + 4 * std::pow(xi[i], 2) + xi[i] - 1.0)) / 3;
                        GradN_zeta_gp[89][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[90][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[91][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[92][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[93][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[94][i] = -(2 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[95][i] = -(8 * eta[i] * xi[i] *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[96][i] = -(2 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[97][i] = -((4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (2 * zeta[i] - 12 * std::pow(zeta[i], 2) - 16 * std::pow(zeta[i], 3) +
                                                  1.0)) / 6;
                        GradN_zeta_gp[98][i] = (64 * eta[i] * xi[i] *
                                                (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                 1.0)) / 27;
                        GradN_zeta_gp[99][i] = (16 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                 1.0)) / 9;
                        GradN_zeta_gp[100][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[101][i] =
                                (16 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] - 1.0)) / 9;
                        GradN_zeta_gp[102][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[103][i] = (16 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 9;
                        GradN_zeta_gp[104][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 27;
                        GradN_zeta_gp[105][i] =
                                (16 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] - 1.0)) / 9;
                        GradN_zeta_gp[106][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 9;
                        GradN_zeta_gp[107][i] = (8 * eta[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0)) /
                                                3;
                        GradN_zeta_gp[108][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_zeta_gp[109][i] = (8 * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 3;
                        GradN_zeta_gp[110][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0)) / 9;
                        GradN_zeta_gp[111][i] = (8 * eta[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) /
                                                3;
                        GradN_zeta_gp[112][i] = (32 * eta[i] * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0)) /
                                                9;
                        GradN_zeta_gp[113][i] = (8 * xi[i] * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                 (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0)) / 3;
                        GradN_zeta_gp[114][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[115][i] = (16 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[116][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) + std::pow(eta[i], 2) + 2 * eta[i] - 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[117][i] =
                                (16 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 1.0)) / 9;
                        GradN_zeta_gp[118][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (-2 * std::pow(xi[i], 3) - std::pow(xi[i], 2) + 2 * xi[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[119][i] = (16 * eta[i] * (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 9;
                        GradN_zeta_gp[120][i] = (64 * eta[i] * xi[i] *
                                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                                 (-2 * std::pow(eta[i], 3) - std::pow(eta[i], 2) + 2 * eta[i] + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 27;
                        GradN_zeta_gp[121][i] =
                                (16 * xi[i] * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                 (-2 * std::pow(xi[i], 3) + std::pow(xi[i], 2) + 2 * xi[i] - 1.0) *
                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 1.0)) / 9;
                        GradN_zeta_gp[122][i] = (4 * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) + 4 * zeta[i] -
                                                  1.0)) / 3;
                        GradN_zeta_gp[123][i] = (4 * (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                 (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0) *
                                                 (4 * zeta[i] - 3 * std::pow(zeta[i], 2) - 8 * std::pow(zeta[i], 3) +
                                                  1.0)) / 3;
                        GradN_zeta_gp[124][i] = 2 * zeta[i] * (8 * std::pow(zeta[i], 2) - 5) *
                                                (4 * std::pow(eta[i], 4) - 5 * std::pow(eta[i], 2) + 1.0) *
                                                (4 * std::pow(xi[i], 4) - 5 * std::pow(xi[i], 2) + 1.0);

                    }
                    break;
            }
            break;
    }


    size_t rows = GradN_xi_gp.size();
    size_t cols = GradN_xi_gp[0].size();

// Declare result once, then resize based on PD.
    std::vector<std::vector<double>> result;
    if (PD == 1) {
        result.resize(rows, std::vector<double>(1 * cols, 0.0));
    } else if (PD == 2) {
        result.resize(rows, std::vector<double>(2 * cols, 0.0));
    } else if (PD == 3) {
        result.resize(rows, std::vector<double>(3 * cols, 0.0));
    } else {
        // Optionally handle unexpected PD values here.
        result.resize(rows, std::vector<double>(3 * cols, 0.0));
    }

// Populate 'result' using an offset based on PD
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t baseIdx = PD * j;  // Adjust offset using PD
            result[i][baseIdx] = GradN_xi_gp[i][j];
            if (PD >= 2) {
                result[i][baseIdx + 1] = GradN_eta_gp[i][j];
            }
            if (PD >= 3) {
                result[i][baseIdx + 2] = GradN_zeta_gp[i][j];
            }
        }
    }
    return std::make_pair(N_xi_gp , result);
}


void printMatrix(const std::vector<std::vector<double>>& matrix, const std::string& name = "Matrix") {
    std::cout << name << ":\n";
    for (size_t i = 0; i < matrix.size(); ++i) {
        std::cout << "Row " << i << ": ";
        for (double value : matrix[i]) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Eigen::VectorXd readVectorFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<double> values;
    double val;

    while (infile >> val) {
        values.push_back(val);
    }

    infile.close(); // ✅ Always close the file

    Eigen::VectorXd vec(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        vec(i) = values[i];
    }

    return vec;
}