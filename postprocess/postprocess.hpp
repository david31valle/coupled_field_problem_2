#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "../Eigen/Dense"

namespace PostProcess {

// Structure to represent Node List elements, now generalized for 3D.
struct NodeData {
    int id;                     // Node ID
    Eigen::Vector3d x;          // Position (x, y, z) - unused components will be 0
    Eigen::VectorXd u;          // Field unknowns (e.g., c, vx, vy, vz)
    Eigen::Vector2i field;      // Field indicators (e.g., field(0) for c, field(1) for v)
    Eigen::VectorXd GP_vals;    // Gauss point values

    NodeData() : id(0), x(Eigen::Vector3d::Zero()),
                 u(Eigen::VectorXd::Zero(1)),      // Default size, will be resized
                 field(Eigen::Vector2i::Zero()),
                 GP_vals(Eigen::VectorXd::Zero(4)) {}
};

// Structure to represent Element List (remains unchanged)
struct ElementData {
    int id;
    int NPE1;
    int NPE2;
    std::vector<int> connectivity;
    Eigen::MatrixXd GP;

    ElementData() : id(0), NPE1(4), NPE2(4) {}
};

// Enum for different output types (remains unchanged)
enum class OutputType {
    CELL_DENSITY,
    VELOCITY_X,
    VELOCITY_Y,
    VELOCITY_Z, // Added for 3D
    VELOCITY_MAGNITUDE,
    VELOCITY_VECTORS,
    ALL
};

// Configuration structure for post-processing
struct Config {
    std::string output_directory;
    bool create_subdirectory;
    bool write_binary_vtk;
    int time_step;
    int problem_dimension;

    Config() : output_directory("./results"),
               create_subdirectory(true),
               write_binary_vtk(false),
               time_step(0),
               problem_dimension(2) {}
};

// The class is now generalized from PostProcessor2D to PostProcessor
class PostProcessor {
private:
    std::vector<NodeData> nodes;
    std::vector<ElementData> elements;
    Config config;

    // Processed data for field 1 (cell density)
    std::vector<double> X_c, Y_c, Z_c, C;
    std::vector<Eigen::VectorXd> GP;

    // Processed data for field 2 (velocity)
    std::vector<double> X_v, Y_v, Z_v, V, V_x, V_y, V_z;

    // Private methods
    void extractFieldData();
    void ensureOutputDirectory();
    std::string generateFilename(const std::string& dataType) const;

public:
    // Constructors
    PostProcessor();
    PostProcessor(const Config& cfg);
    PostProcessor(const std::vector<NodeData>& nodeList,
                  const std::vector<ElementData>& elemList,
                  const Config& cfg);

    // Destructor
    ~PostProcessor() = default;

    // Set data
    void setNodeData(const std::vector<NodeData>& nodeList);
    void setElementData(const std::vector<ElementData>& elemList);
    void setConfig(const Config& cfg);
    void setTimeStep(int step);
    void setOutputDirectory(const std::string& dir);

    // Add single node/element
    void addNode(const NodeData& node);
    void addElement(const ElementData& element);

    // Clear data
    void clearData();
    void clearProcessedData();

    // Main processing function
    void process();

    // VTK output functions
    void writeVTKFile(OutputType type);
    void writeAllVTKFiles();
    void writeCellDensityVTK();
    void writeVelocityComponentsVTK();
    void writeVelocityVectorFieldVTK();
    void writeCombinedVTK();

    // Utility functions
    int getNumberOfNodes() const { return nodes.size(); }
    int getNumberOfElements() const { return elements.size(); }
    bool hasVelocityData() const { return !X_v.empty(); }
    bool hasCellDensityData() const { return !X_c.empty(); }

    // Statistics
    void printStatistics() const;
    std::pair<double, double> getCellDensityRange() const;
    std::pair<double, double> getVelocityMagnitudeRange() const;
};

} // namespace PostProcess

#endif // POSTPROCESS_HPP
