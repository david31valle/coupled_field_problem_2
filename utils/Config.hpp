#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

struct Config {
    // Keep your fields (with whatever initial values you prefer). They won’t be used
    // until we successfully load+validate the file.
    int problem_dimension;
    std::vector<int> element_order;
    double domain_size;
    int partition;

    std::string density_path;
    std::string initial_density;
    double initial_cell_density;
    double cell_density_perturbation;

    double young_modulus;
    double cell_radius;
    double friction_coefficient;

    double T, dt, time_factor;
    std::string time_increment;
    int max_iter;
    double tol;
    std::string boundary_condition, corners, GP_vals, plot_mesh;

    std::vector<double> parameters() const {
        return {young_modulus, cell_radius, friction_coefficient};
    }

    static std::string trim(std::string s) {
        auto ws = [](int ch){ return ch==' '||ch=='\t'||ch=='\r'||ch=='\n'; };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](int c){return !ws(c);} ));
        s.erase(std::find_if(s.rbegin(), s.rend(), [&](int c){return !ws(c);} ).base(), s.end());
        return s;
    }

    static std::vector<int> parse_int_list(const std::string& v) {
        std::vector<int> out; std::stringstream ss(v); std::string tok;
        while (std::getline(ss, tok, ',')) out.push_back(std::stoi(trim(tok)));
        return out;
    }

    // Parse single key/value into fields, and remember what we saw.
    static void set_kv(Config& c, std::unordered_set<std::string>& seen,
                       const std::string& k, const std::string& v) {
        auto mark = [&](const char* key){ seen.insert(key); };

        if (k=="problem_dimension") { c.problem_dimension = std::stoi(v); mark("problem_dimension"); }
        else if (k=="element_order") { c.element_order = parse_int_list(v); mark("element_order"); }
        else if (k=="domain_size") { c.domain_size = std::stod(v); mark("domain_size"); }
        else if (k=="partition") { c.partition = std::stoi(v); mark("partition"); }
        else if (k=="density_path") { c.density_path = v; mark("density_path"); }
        else if (k=="initial_density") { c.initial_density = v; mark("initial_density"); }
        else if (k=="initial_cell_density") { c.initial_cell_density = std::stod(v); mark("initial_cell_density"); }
        else if (k=="cell_density_perturbation") { c.cell_density_perturbation = std::stod(v); mark("cell_density_perturbation"); }
        else if (k=="young_modulus") { c.young_modulus = std::stod(v); mark("young_modulus"); }
        else if (k=="cell_radius") { c.cell_radius = std::stod(v); mark("cell_radius"); }
        else if (k=="friction_coefficient") { c.friction_coefficient = std::stod(v); mark("friction_coefficient"); }
        else if (k=="T") { c.T = std::stod(v); mark("T"); }
        else if (k=="dt") { c.dt = std::stod(v); mark("dt"); }
        else if (k=="time_increment") { c.time_increment = v; mark("time_increment"); }
        else if (k=="time_factor") { c.time_factor = std::stod(v); mark("time_factor"); }
        else if (k=="max_iter") { c.max_iter = std::stoi(v); mark("max_iter"); }
        else if (k=="tol") { c.tol = std::stod(v); mark("tol"); }
        else if (k=="boundary_condition") { c.boundary_condition = v; mark("boundary_condition"); }
        else if (k=="corners") { c.corners = v; mark("corners"); }
        else if (k=="GP_vals") { c.GP_vals = v; mark("GP_vals"); }
        else if (k=="plot_mesh") { c.plot_mesh = v; mark("plot_mesh"); }
        // unknown keys are ignored (or log a warning if you prefer)
    }

    // ---- MANDATORY loader: file must exist and must contain all required keys ----
    static Config from_ini_required(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error(
                    "No input config file found at '" + path + "'. Provide a valid config file.");
        }

        Config c{};
        std::unordered_set<std::string> seen;
        std::string line;

        while (std::getline(f, line)) {
            auto posHash = line.find('#'); if (posHash!=std::string::npos) line=line.substr(0,posHash);
            auto posEq   = line.find('='); if (posEq==std::string::npos) continue;
            auto key = trim(line.substr(0,posEq));
            auto val = trim(line.substr(posEq+1));
            if (!key.empty() && !val.empty()) {
                try { set_kv(c, seen, key, val); }
                catch (const std::exception& e) {
                    throw std::runtime_error("Invalid value for key '" + key + "': " + val);
                }
            }
        }

        // Validate presence of required keys
        static const char* required[] = {
                "problem_dimension","element_order","domain_size","partition",
                "density_path","initial_density","initial_cell_density","cell_density_perturbation",
                "young_modulus","cell_radius","friction_coefficient",
                "T","dt","time_increment","time_factor","max_iter","tol",
                "boundary_condition","corners","GP_vals","plot_mesh"
        };

        std::vector<std::string> missing;
        for (auto* k : required) if (!seen.count(k)) missing.emplace_back(k);

        if (!missing.empty()) {
            std::ostringstream msg;
            msg << "Config file '" << path << "' is missing required key(s): ";
            for (size_t i=0;i<missing.size();++i) { msg << missing[i] << (i+1<missing.size()? ", ":""); }
            throw std::runtime_error(msg.str());
        }

        return c;
    }

    // CLI overrides stay allowed, but only AFTER a valid file was loaded.
    static Config override_from_argv(int argc, char** argv, Config base) {
        for (int i=1;i<argc;++i) {
            std::string a(argv[i]);
            if (a.rfind("--",0)==0) {
                auto eq = a.find('=');
                if (eq!=std::string::npos) {
                    auto key = a.substr(2, eq-2);
                    auto val = a.substr(eq+1);
                    std::unordered_set<std::string> dummy_seen;
                    set_kv(base, dummy_seen, key, val);
                }
            }
        }
        return base;
    }
};