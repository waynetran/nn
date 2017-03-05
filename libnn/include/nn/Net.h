#pragma once

#include "Common.h"
#include "Utils.h"

class Edge;

class Layer;

class Node {
public:
    static std::shared_ptr<Edge> connect(std::shared_ptr<Node> in,
                                         std::shared_ptr<Node> out,
                                         double weight);

public:
    Node();

    uint64_t getId() { return _id; }

    void setId(uint64_t id) { _id = id; }

    double getValue() { return _value; }

    void setValue(double value) { _value = value; }

    double getError() { return _error; }

    void setError(double error) { _error = error; }

    std::set<std::shared_ptr<Edge> > getOutEdges() { return _outEdges; }

    std::set<std::shared_ptr<Edge> > getInEdges() { return _inEdges; }

    void setOutEdges(
            const std::set<std::shared_ptr<Edge> > &edges) { _outEdges = edges; }

    void setInEdges(
            const std::set<std::shared_ptr<Edge> > &edges) { _inEdges = edges; }

    std::shared_ptr<Edge> getOutEdge(std::shared_ptr<Node> node);

    std::shared_ptr<Edge> getInEdge(std::shared_ptr<Node> node);

    bool hasOut(std::shared_ptr<Node> node) {
        return getOutEdge(node) != nullptr;
    };

    bool hasIn(std::shared_ptr<Node> node) {
        return getInEdge(node) != nullptr;
    };

protected:
    static uint64_t getNewId() {
        _gid++;
        return _gid;
    };
    static uint64_t _gid;
    uint64_t _id;
    double _value;
    double _error;
    std::set<std::shared_ptr<Edge> > _inEdges;
    std::set<std::shared_ptr<Edge> > _outEdges;
};

class Edge {
public:
    Edge(std::shared_ptr<Node> in, std::shared_ptr<Node> out);

    std::shared_ptr<Node> getFirst() { return _first; }

    void setFirst(std::shared_ptr<Node> in) { _first = in; }

    std::shared_ptr<Node> getSecond() { return _second; }

    void setSecond(std::shared_ptr<Node> out) { _second = out; }

    double getWeight() { return _weight; }

    void setWeight(double weight) {
        //weight = MathUtils::clamp(weight, -1000, 1000);
        setWeightPrevious(_weight);
        _weight = weight;
    }

    double getWeightPevious() { return _weightPrevious; }

    void setWeightPrevious(double weight) { _weightPrevious = weight; }

    bool hasNode(std::shared_ptr<Node> node) {
        return getFirst() == node || getSecond() == node;
    }

    std::shared_ptr<Node> getOther(std::shared_ptr<Node> node) {
        return getFirst() == node ? getSecond() : getFirst();
    }

protected:
    std::shared_ptr<Node> _first;
    std::shared_ptr<Node> _second;
    double _weight;
    double _weightPrevious;

};

class Layer {
public:
    //For a given node, sums weight*value of all incoming nodes
    static double weightedSum(std::shared_ptr<Node> node);

    //For a given node, sums weight*error of all out nodes
    static double weightedErrorSum(std::shared_ptr<Node> node);

public:
    Layer(uint64_t id) : _id(id) { }

    uint64_t getId() { return _id; }

    void setId(uint64_t id) { _id = id; }

    std::vector<std::shared_ptr<Node> > getNodes() { return _nodes; }

    void setNodes(
            const std::vector<std::shared_ptr<Node> > &nodes) { _nodes = nodes; }

    void add(std::shared_ptr<Node> node);

    void remove(std::shared_ptr<Node> node);

protected:
    uint64_t _id;
    std::vector<std::shared_ptr<Node> > _nodes;
};

class Net {
public:
    Net(const std::vector<int> &neuronsPerLayer,
        uint64_t maxIterations = 1000, double errorThreshold = 0.0001,
        double learningRate = 0.1, double momentumRate = 0.5
    );

    ~Net();

    void forward(std::vector<double> inputs);

    void updateWeights(std::vector<double> expectedOutputs);

    void train(std::vector<std::vector<double> > inputs,
               std::vector<std::vector<double> > outputs);

    bool isFinished();

    double getMeanOutputError(std::vector<double> expected);

    uint64_t getNumLayers() { return _layers.size(); }

    std::string trace(bool outputOnly = true);

    std::vector<double> getOutputValues();

    std::vector<std::shared_ptr<Node> > getLayerNodes(uint64_t index);

    std::string toString();

protected:
    std::vector<std::shared_ptr<Layer> > _layers;
    //Stop on iterations
    uint64_t _maxIterations;
    //Stop if error is less
    double _errorThreshold;
    //Between 0.0 and 1.0  - speeds up or slows down learning.
    // Higher value is risky and can overshoot.
    // Lower we might not converge before stop condition.
    double _learningRate;
    //Helps our weight updates to limit changes of direction
    double _momentumRate;

    bool _isFinished;

};

