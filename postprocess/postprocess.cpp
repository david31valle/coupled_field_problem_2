#include "postprocess.hpp"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <numeric>

namespace PostProcess {

// Constructor implementations
PostProcessor::PostProcessor() : config(Config()) {}

PostProcessor::PostProcessor(const Config& cfg) : config(cfg) {}

PostProcessor::PostProcessor(const std::vector<NodeData>& nodeList,
                             const std::vector<ElementData>& elemList,
                             const Config& cfg)
    : nodes(nodeList), elements(elemList), config(cfg) {}

// Set data methods
void PostProcessor::setNodeData(const std::vector<NodeData>& nodeList) {
    nodes = nodeList;
    clearProcessedData();
}

void PostProcessor::setElementData(const std::vector<ElementData>& elemList) {
    elements = elemList;
}

void PostProcessor::setConfig(const Config& cfg) {
    config = cfg;
}

void PostProcessor::setTimeStep(int step) {
    config.time_step = step;
}

void PostProcessor::setOutputDirectory(const std::string& dir) {
    config.output_directory = dir;
}

// Add single node/element
void PostProcessor::addNode(const NodeData& node) {
    nodes.push_back(node);
    clearProcessedData();
}

void PostProcessor::addElement(const ElementData& element) {
    elements.push_back(element);
}

// Clear methods
void PostProcessor::clearData() {
    nodes.clear();
    elements.clear();
    clearProcessedData();
}

void PostProcessor::clearProcessedData() {
    X_c.clear(); Y_c.clear(); Z_c.clear(); C.clear(); GP.clear();
    X_v.clear(); Y_v.clear(); Z_v.clear(); V.clear(); V_x.clear(); V_y.clear(); V_z.clear();
}

// Main processing function
void PostProcessor::process() {
    if (nodes.empty()) {
        std::cerr << "PostProcessor: No node data to process!" << std::endl;
        return;
    }

    extractFieldData();
    ensureOutputDirectory();
    writeAllVTKFiles();
    printStatistics();
}

// UPDATED: Extract field data from nodes for 1D, 2D, or 3D
void PostProcessor::extractFieldData() {
    clearProcessedData();
    const int PD = config.problem_dimension;

    for (const auto& node : nodes) {
        // Process field 1 (cell density) if active
        if (node.field(0) == 1) {
            X_c.push_back(node.x(0));
            Y_c.push_back(node.x(1));
            Z_c.push_back(node.x(2));
            C.push_back(node.u(0));
            GP.push_back(node.GP_vals);
        }

        // Process field 2 (velocity) if active
        if (node.field(1) == 1 && node.u.size() >= 1 + PD) {
            X_v.push_back(node.x(0));
            Y_v.push_back(node.x(1));
            Z_v.push_back(node.x(2));

            double vx = 0, vy = 0, vz = 0;
            if (PD >= 1) vx = node.u(1);
            if (PD >= 2) vy = node.u(2);
            if (PD >= 3) vz = node.u(3);

            V_x.push_back(vx);
            V_y.push_back(vy);
            V_z.push_back(vz);
            V.push_back(std::sqrt(vx*vx + vy*vy + vz*vz));
        }
    }
}

// Ensure output directory exists
void PostProcessor::ensureOutputDirectory() {
    namespace fs = std::filesystem;
    fs::path dir_path(config.output_directory);

    if (config.create_subdirectory) {
        std::stringstream ss;
        ss << "step_" << std::setfill('0') << std::setw(6) << config.time_step;
        dir_path /= ss.str();
    }

    if (!fs::exists(dir_path)) {
        if (!fs::create_directories(dir_path)) {
            std::cerr << "Error: Could not create output directory: " << dir_path << std::endl;
        }
    }
}

// Generate filename
std::string PostProcessor::generateFilename(const std::string& dataType) const {
    namespace fs = std::filesystem;
    std::stringstream ss;
    ss << config.problem_dimension << "D_";
    ss << std::setfill('0') << std::setw(6) << config.time_step;
    ss << "_" << dataType << ".vtk";

    fs::path filepath = fs::path(config.output_directory);
    if (config.create_subdirectory) {
        std::stringstream subdir;
        subdir << "step_" << std::setfill('0') << std::setw(6) << config.time_step;
        filepath /= subdir.str();
    }
    filepath /= ss.str();

    return filepath.string();
}

// Write VTK file based on type
void PostProcessor::writeVTKFile(OutputType type) {
    std::vector<double>* x_data = nullptr, *y_data = nullptr, *z_data = nullptr;
    std::vector<double>* scalar_data = nullptr;
    std::string scalar_name, filename_suffix;

    switch(type) {
        case OutputType::CELL_DENSITY:
            if (!hasCellDensityData()) return;
            x_data = &X_c; y_data = &Y_c; z_data = &Z_c; scalar_data = &C;
            scalar_name = "CellDensity"; filename_suffix = "cell_density";
            break;
        case OutputType::VELOCITY_X:
            if (!hasVelocityData()) return;
            x_data = &X_v; y_data = &Y_v; z_data = &Z_v; scalar_data = &V_x;
            scalar_name = "VelocityX"; filename_suffix = "velocity_x";
            break;
        case OutputType::VELOCITY_Y:
            if (!hasVelocityData()) return;
            x_data = &X_v; y_data = &Y_v; z_data = &Z_v; scalar_data = &V_y;
            scalar_name = "VelocityY"; filename_suffix = "velocity_y";
            break;
        case OutputType::VELOCITY_Z:
            if (!hasVelocityData() || config.problem_dimension < 3) return;
            x_data = &X_v; y_data = &Y_v; z_data = &Z_v; scalar_data = &V_z;
            scalar_name = "VelocityZ"; filename_suffix = "velocity_z";
            break;
        case OutputType::VELOCITY_MAGNITUDE:
            if (!hasVelocityData()) return;
            x_data = &X_v; y_data = &Y_v; z_data = &Z_v; scalar_data = &V;
            scalar_name = "VelocityMagnitude"; filename_suffix = "velocity_magnitude";
            break;
        case OutputType::VELOCITY_VECTORS: writeVelocityVectorFieldVTK(); return;
        case OutputType::ALL: writeAllVTKFiles(); return;
    }

    if (!x_data || x_data->empty()) return;

    std::string filename = generateFilename(filename_suffix);
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }

    int nPoints = x_data->size();
    file << "# vtk DataFile Version 3.0\n";
    file << "Post Process " << config.problem_dimension << "D - " << scalar_name << "\n";
    file << (config.write_binary_vtk ? "BINARY\n" : "ASCII\n");
    file << "DATASET POLYDATA\n";
    file << "POINTS " << nPoints << " float\n";
    for (int i = 0; i < nPoints; ++i) {
        file << (*x_data)[i] << " " << (*y_data)[i] << " " << (*z_data)[i] << "\n";
    }
    file << "VERTICES " << nPoints << " " << 2*nPoints << "\n";
    for (int i = 0; i < nPoints; ++i) {
        file << "1 " << i << "\n";
    }
    file << "POINT_DATA " << nPoints << "\n";
    file << "SCALARS " << scalar_name << " float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nPoints; ++i) {
        file << (*scalar_data)[i] << "\n";
    }
    file.close();
    std::cout << "Written: " << filename << std::endl;
}

void PostProcessor::writeAllVTKFiles() {
    writeCellDensityVTK();
    writeVelocityComponentsVTK();
    writeVelocityVectorFieldVTK();
    writeCombinedVTK();
}

void PostProcessor::writeCellDensityVTK() {
    writeVTKFile(OutputType::CELL_DENSITY);
}

void PostProcessor::writeVelocityComponentsVTK() {
    writeVTKFile(OutputType::VELOCITY_X);
    if (config.problem_dimension >= 2) writeVTKFile(OutputType::VELOCITY_Y);
    if (config.problem_dimension >= 3) writeVTKFile(OutputType::VELOCITY_Z);
    writeVTKFile(OutputType::VELOCITY_MAGNITUDE);
}

void PostProcessor::writeVelocityVectorFieldVTK() {
    if (!hasVelocityData()) return;

    std::string filename = generateFilename("velocity_vectors");
    std::ofstream file(filename);
    if (!file.is_open()) return;

    int nPoints = X_v.size();
    file << "# vtk DataFile Version 3.0\n";
    file << "Velocity Vector Field\n";
    file << "ASCII\nDATASET POLYDATA\n";
    file << "POINTS " << nPoints << " float\n";
    for (int i = 0; i < nPoints; ++i) file << X_v[i] << " " << Y_v[i] << " " << Z_v[i] << "\n";
    file << "VERTICES " << nPoints << " " << 2*nPoints << "\n";
    for (int i = 0; i < nPoints; ++i) file << "1 " << i << "\n";
    file << "POINT_DATA " << nPoints << "\n";
    file << "VECTORS Velocity float\n";
    for (int i = 0; i < nPoints; ++i) file << V_x[i] << " " << V_y[i] << " " << V_z[i] << "\n";
    file << "SCALARS VelocityMagnitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nPoints; ++i) file << V[i] << "\n";
    file.close();
    //std::cout << "Written: " << filename << std::endl;
}

void PostProcessor::writeCombinedVTK() {
    if (nodes.empty()) return;

    std::string filename = generateFilename("combined");
    std::ofstream file(filename);
    if (!file.is_open()) return;

    int nPoints = nodes.size();
    file << "# vtk DataFile Version 3.0\n";
    file << "Combined Fields\n";
    file << "ASCII\nDATASET POLYDATA\n";
    file << "POINTS " << nPoints << " float\n";
    for (const auto& node : nodes) file << node.x(0) << " " << node.x(1) << " " << node.x(2) << "\n";
    file << "VERTICES " << nPoints << " " << 2*nPoints << "\n";
    for (int i = 0; i < nPoints; ++i) file << "1 " << i << "\n";
    file << "POINT_DATA " << nPoints << "\n";
    file << "SCALARS CellDensity float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (const auto& node : nodes) file << (node.field(0) == 1 ? node.u(0) : 0.0) << "\n";
    file << "VECTORS Velocity float\n";
    for (const auto& node : nodes) {
        if (node.field(1) == 1 && node.u.size() >= 1 + config.problem_dimension) {
            double vx = (config.problem_dimension >= 1) ? node.u(1) : 0.0;
            double vy = (config.problem_dimension >= 2) ? node.u(2) : 0.0;
            double vz = (config.problem_dimension >= 3) ? node.u(3) : 0.0;
            file << vx << " " << vy << " " << vz << "\n";
        } else {
            file << "0.0 0.0 0.0\n";
        }
    }
    file.close();
    //std::cout << "Written: " << filename << std::endl;
}

// Statistics, if needed only for debugging
void PostProcessor::printStatistics() const {
    std::cout << "\n=== Post-Processing Statistics (" << config.problem_dimension << "D) ===" << std::endl;
    //std::cout << "Total nodes: " << nodes.size() << std::endl;
    //std::cout << "Nodes with cell density data: " << C.size() << std::endl;
    std::cout << "Nodes with velocity data: " << V.size() << std::endl;
    if (hasCellDensityData()) {
        auto [c_min, c_max] = getCellDensityRange();
        std::cout << "Cell density range: [" << c_min << ", " << c_max << "]" << std::endl;
    }
    if (hasVelocityData()) {
        auto [v_min, v_max] = getVelocityMagnitudeRange();
        std::cout << "Velocity magnitude range: [" << v_min << ", " << v_max << "]" << std::endl;
    }
    std::cout << "======================================\n" << std::endl;
}

std::pair<double, double> PostProcessor::getCellDensityRange() const {
    if (C.empty()) return {0.0, 0.0};
    auto [min_it, max_it] = std::minmax_element(C.begin(), C.end());
    return {*min_it, *max_it};
}

std::pair<double, double> PostProcessor::getVelocityMagnitudeRange() const {
    if (V.empty()) return {0.0, 0.0};
    auto [min_it, max_it] = std::minmax_element(V.begin(), V.end());
    return {*min_it, *max_it};
}

} // namespace PostProcess
