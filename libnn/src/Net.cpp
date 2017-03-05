//
// Created by Wayne Tran on 2016-03-27.
//

#include "nn/Net.h"

using namespace std;
uint64_t Node::_gid = 0;


std::shared_ptr<Edge> Node::connect(std::shared_ptr<Node> first,
                                    std::shared_ptr<Node> second,
                                    double weight) {
    if (first->hasOut(second)) {
        return first->getOutEdge(second);
    }
    std::shared_ptr<Edge> edge = make_shared<Edge>(first, second);
    first->_outEdges.insert(edge);
    second->_inEdges.insert(edge);
    edge->setWeight(weight);
    return edge;
}

Node::Node() : _value(0), _error(0) {
    _id = getNewId();
}

std::shared_ptr<Edge> Node::getOutEdge(std::shared_ptr<Node> node) {
    for (auto edge: getOutEdges()) {
        if (edge->hasNode(node)) {
            return edge;
        }
    }
    return nullptr;
}

std::shared_ptr<Edge> Node::getInEdge(std::shared_ptr<Node> node) {
    for (auto edge: getInEdges()) {
        if (edge->hasNode(node)) {
            return edge;
        }
    }
    return nullptr;
}

Edge::Edge(std::shared_ptr<Node> in, std::shared_ptr<Node> out) :
        _first(in), _second(out), _weight(0), _weightPrevious(0) {

}

void Layer::add(std::shared_ptr<Node> node) {
    _nodes.push_back(node);
}

void Layer::remove(std::shared_ptr<Node> node) {
    _nodes.erase(std::remove(_nodes.begin(), _nodes.end(), node), _nodes.end());
}

double Layer::weightedSum(std::shared_ptr<Node> node) {
    double sum = 0.0;
    for (auto edge: node->getInEdges()) {
        sum += edge->getOther(node)->getValue() * edge->getWeight();
    }
    return sum;
}

double Layer::weightedErrorSum(std::shared_ptr<Node> node) {
    double sum = 0.0;
    for (auto edge: node->getOutEdges()) {
        sum += edge->getOther(node)->getError() * edge->getWeight();
    }
    return sum;
}


Net::Net(const std::vector<int> &neuronsPerLayer, uint64_t maxIterations,
         double errorThreshold, double learningRate, double momentumRate) :
        _maxIterations(maxIterations),
        _errorThreshold(errorThreshold),
        _learningRate(learningRate),
        _momentumRate(momentumRate),
        _isFinished(true) {

    int i = 0;
    for (int num: neuronsPerLayer) {
        shared_ptr<Layer> layer = make_shared<Layer>(i);
        // vector<double> weights;
        for (int j = 0; j < num; j++) {
            shared_ptr<Node> n = make_shared<Node>();
            layer->add(n);
            //Connect previous layer to this layer
            if (i > 0) {
                for (auto n1 : _layers[i - 1]->getNodes()) {
                    Node::connect(n1, n, MathUtils::randomDouble(0.1, 0.5));
                }
            }
        }
        _layers.push_back(layer);
        i++;
    }

}

Net::~Net() {

}

void Net::forward(std::vector<double> inputs) {
    if (inputs.size() != _layers[0]->getNodes().size()) {
        return;
    }

    //Set input neuron values
    for (int i = 0; i < inputs.size(); ++i) {
        _layers[0]->getNodes()[i]->setValue(inputs[i]);
    }

    //Calculate activation values starting with first hidden layer
    for (int i = 1; i < _layers.size(); ++i) {
        for (auto n: _layers[i]->getNodes()) {
            double sum = _layers[i]->weightedSum(n);
            n->setValue(MathUtils::sigmoid(sum));
        }
    }

    return;
}

void Net::updateWeights(std::vector<double> expectedOutputs) {

    shared_ptr<Layer> outputLayer = _layers[_layers.size() - 1];
    shared_ptr<Layer> inputLayer = _layers[0];
    if (!outputLayer ||
        expectedOutputs.size() != outputLayer->getNodes().size()) {
        return;
    }


    //Start from outputLayer all the way back to input, calculating the
    //error gradients
    for (uint64_t i = _layers.size() - 1; i != (uint64_t) -1; --i) {
        shared_ptr<Layer> layer = _layers[i];
        int j = 0;
        for (auto node: layer->getNodes()) {
            double error = 0;
            if (i == _layers.size() - 1) {
                //Error gradient for output layer
                error = MathUtils::errorGradOutput(node->getValue(),
                                                   expectedOutputs[j]);
            } else {
                //Hidden Layer
                error = MathUtils::errorGrad(node->getValue(),
                                             Layer::weightedErrorSum(node));

            }
            node->setError(error);
            j++;
        }
    }



    //Update Weights
    for (uint64_t i = _layers.size() - 2; i != (uint64_t) -1; --i) {
        shared_ptr<Layer> layer = _layers[i];
        for (auto node: layer->getNodes()) {
            //Hidden Layer
            double deltaWeight = 0;
            double momentumTerm = 0;
            //Calculate new weights for all outgoing edges

            for (auto edge: node->getOutEdges()) {
                double outErr = edge->getOther(node)->getError();
                deltaWeight = _learningRate * outErr * node->getValue();
                momentumTerm = 0; //_momentumRate * edge->getWeightPevious();
                double newWeight =
                        edge->getWeight() + deltaWeight + momentumTerm;
                edge->setWeight(newWeight);
            }


        }
    }


}

void Net::train(std::vector<std::vector<double> > inputs,
                std::vector<std::vector<double> > outputs) {

    double error = HUGE_VALF;
    _isFinished = false;
    bool stop = false;
    std::vector<double> errors;
    errors.resize(inputs.size());

    for (uint64_t j = 0; j < _maxIterations && !stop; j++) {
        uint64_t i = j % inputs.size();
        std::vector<double> input = inputs[i];
        std::vector<double> output = outputs[i];
        //cout << " Inputs: " << Utils::toString(input) << endl;
        //cout << " Outputs: " << Utils::toString(output) << endl;
        error = HUGE_VALF;
        //cout << "global iteration :" << j << endl;
        for (int k = 0; k < _maxIterations && error > _errorThreshold; k++) {
            forward(input);
            updateWeights(output);
            error = getMeanOutputError(output);
            //cout << "Pattern iteration :" << k << endl;
            //cout << "MSE: " << error << " THRESHOLD: " << _errorThreshold <<
            //endl;
            //cout << trace() << endl;
        }


        if (i == 3) {
            errors.clear();
            double largestError = 0.0;
            for (auto in: inputs) {
                forward(in);
                error = getMeanOutputError(output);
                if (error > largestError)
                    largestError = error;
            }

            if (largestError < _errorThreshold) {
                stop = true;
            }
        }

        cout << Utils::toString(inputs[i]) << " -> " <<
        Utils::toString(getOutputValues()) << endl;

    }
    _isFinished = true;
}

bool Net::isFinished() {
    return _isFinished;
}

double Net::getMeanOutputError(vector<double> expected) {
    return MathUtils::meanSquaredError(getOutputValues(), expected);
}

std::vector<double> Net::getOutputValues() {
    std::vector<double> out;
    for (auto node: getLayerNodes(getNumLayers() - 1)) {
        out.push_back(node->getValue());
    }
    return out;
}

vector<shared_ptr<Node> > Net::getLayerNodes(uint64_t index) {
    if (index > _layers.size() - 1) {
        vector<shared_ptr<Node> > nodes;
        return nodes;
    }
    return _layers[index]->getNodes();
}

std::string Net::trace(bool outputOnly) {
    stringstream ss;
    if (outputOnly) {
        for (auto value: getOutputValues()) {
            ss << value << " ";
        }
        ss << endl;
    } else {
        for (auto layer: _layers) {
            ss << "layer: " << layer->getId() << endl;
            for (auto neuron: layer->getNodes()) {
                ss << "node: " << neuron->getId() << "  value: " <<
                neuron->getValue() << " error: " << neuron->getError() << endl;
                for (auto edge: neuron->getOutEdges()) {
                    ss << "     out node: " <<
                    edge->getOther(neuron)->getId() <<
                    "    weight: " << edge->getWeight() << endl;
                }
            }
        }
    }
    return ss.str();
}

std::string Net::toString() {
    stringstream ss;
    int i = 0;
    for (auto val: getOutputValues()) {
        ss << val << " ";
        i++;
    }
    ss << endl;
    return ss.str();
}
