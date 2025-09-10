#pragma once
#include <unordered_map>
#include <mutex>
#include <tuple>
#include <vector>
#include <utility>
#include <Eigen/Dense>

// forward decls of your existing helpers
std::vector<std::vector<double>> compute_gp(int NGP, int PD);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
compute_N_xi_gp(int degree, const std::vector<std::vector<double>>& GP, int PD);

namespace fem_cache {

    struct GPKey {
        int NGP, PD;
        bool operator==(const GPKey& o) const noexcept { return NGP==o.NGP && PD==o.PD; }
    };
    struct GPKeyHash {
        std::size_t operator()(const GPKey& k) const noexcept {
            return (std::hash<int>()(k.NGP) << 1) ^ std::hash<int>()(k.PD);
        }
    };

    struct NKey {
        int degree, NGP, PD;
        bool operator==(const NKey& o) const noexcept {
            return degree==o.degree && NGP==o.NGP && PD==o.PD;
        }
    };
    struct NKeyHash {
        std::size_t operator()(const NKey& k) const noexcept {
            // simple but fine
            return ((std::hash<int>()(k.degree)*131u) ^ (std::hash<int>()(k.NGP)*31u) ^ std::hash<int>()(k.PD));
        }
    };

    class ShapeCache {
    public:
        static const std::vector<std::vector<double>>& gp_vec(int NGP, int PD) {
            GPKey key{NGP, PD};
            {
                std::lock_guard<std::mutex> lk(m_);
                auto it = gp_vec_.find(key);
                if (it != gp_vec_.end()) return it->second;
            }
            // compute once, then store
            auto vec = compute_gp(NGP, PD);
            std::lock_guard<std::mutex> lk(m_);
            auto [it, _] = gp_vec_.emplace(key, std::move(vec));
            return it->second;
        }

        static const Eigen::MatrixXd& gp_mat(int NGP, int PD) {
            GPKey key{NGP, PD};
            {
                std::lock_guard<std::mutex> lk(m_);
                auto it = gp_mat_.find(key);
                if (it != gp_mat_.end()) return it->second;
            }
            const auto& gv = gp_vec(NGP, PD);
            Eigen::MatrixXd M(gv.size(), gv.empty()?0:gv[0].size());
            for (size_t i=0;i<gv.size();++i)
                for (size_t j=0;j<gv[0].size();++j)
                    M(i,j) = gv[i][j];
            std::lock_guard<std::mutex> lk(m_);
            auto [it, _] = gp_mat_.emplace(key, std::move(M));
            return it->second;
        }

        // Returns cached (N_vec, GradN_xi_vec)
        static const std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>&
        N_and_Grad_vec(int degree, int NGP, int PD) {
            NKey key{degree, NGP, PD};
            {
                std::lock_guard<std::mutex> lk(m_);
                auto it = N_grad_vec_.find(key);
                if (it != N_grad_vec_.end()) return it->second;
            }
            const auto& gpv = gp_vec(NGP, PD);
            auto pairNG = compute_N_xi_gp(degree, gpv, PD);
            std::lock_guard<std::mutex> lk(m_);
            auto [it, _] = N_grad_vec_.emplace(key, std::move(pairNG));
            return it->second;
        }

        // Convenience: get dense Eigen matrices too (built once)
        static const Eigen::MatrixXd& N_mat(int degree, int NGP, int PD) {
            NKey key{degree, NGP, PD};
            {
                std::lock_guard<std::mutex> lk(m_);
                auto it = N_mat_.find(key);
                if (it != N_mat_.end()) return it->second;
            }
            const auto& NG = N_and_Grad_vec(degree, NGP, PD).first;
            Eigen::MatrixXd M(NG.size(), NG.empty()?0:NG[0].size());
            for (size_t i=0;i<NG.size();++i)
                for (size_t j=0;j<NG[0].size();++j)
                    M(i,j) = NG[i][j];

            std::lock_guard<std::mutex> lk(m_);
            auto [it, _] = N_mat_.emplace(key, std::move(M));
            return it->second;
        }

    private:
        static std::mutex m_;
        static std::unordered_map<GPKey, std::vector<std::vector<double>>, GPKeyHash> gp_vec_;
        static std::unordered_map<GPKey, Eigen::MatrixXd, GPKeyHash> gp_mat_;
        static std::unordered_map<NKey, std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>, NKeyHash> N_grad_vec_;
        static std::unordered_map<NKey, Eigen::MatrixXd, NKeyHash> N_mat_;
    };

    inline std::mutex ShapeCache::m_;
    inline std::unordered_map<GPKey, std::vector<std::vector<double>>, GPKeyHash> ShapeCache::gp_vec_;
    inline std::unordered_map<GPKey, Eigen::MatrixXd, GPKeyHash> ShapeCache::gp_mat_;
    inline std::unordered_map<NKey, std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>, NKeyHash> ShapeCache::N_grad_vec_;
    inline std::unordered_map<NKey, Eigen::MatrixXd, NKeyHash> ShapeCache::N_mat_;

} // namespace fem_cache
