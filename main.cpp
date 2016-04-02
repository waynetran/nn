#include <iostream>
#include "Net.h"

using namespace std;

int main() {
    vector<int> layers{2, 5, 1};

    Net net(layers, 2000);
    cout << "INIT: \n";
    cout << net.trace() << endl;

    vector<vector<double> > inputs;
    vector<double> input1{0.0, 0.0};
    vector<double> input2{1.0, 0.0};
    vector<double> input3{0.0, 1.0};
    vector<double> input4{1.0, 1.0};
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    inputs.push_back(input4);

    vector<vector<double> > outputs;
    vector<double> output1{0.0};
    vector<double> output2{1.0};
    vector<double> output3{1.0};
    vector<double> output4{0.0};
    outputs.push_back(output1);
    outputs.push_back(output2);
    outputs.push_back(output3);
    outputs.push_back(output4);

    net.train(inputs, outputs);

    cout << "TRAINED: \n";
    cout << net.trace(false) << endl;

    net.forward(input1);
    cout << net.toString();
    net.forward(input2);
    cout << net.toString();
    net.forward(input3);
    cout << net.toString();
    net.forward(input4);
    cout << net.toString();

    return 0;
}