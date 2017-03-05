#include <iostream>
#include "nn/Net.h"

using namespace std;

bool testXOR(){
    vector<int> layers{2, 5, 1};

    Net net(layers, 2000);
    cout << "INIT: \n";
    //cout << net.trace() << endl;

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

    //cout << "TRAINED: \n";
    //cout << net.trace(false) << endl;

    net.forward(input1);
    auto out = net.getOutputValues();
    if(fabs(out[0] - output1[0]) > 0.1)
        return false;

    net.forward(input2);
    auto out2 = net.getOutputValues();
    if(fabs(out2[0] - output2[0]) > 0.1)
        return false;

    net.forward(input3);
    auto out3 = net.getOutputValues();
    if(fabs(out3[0] - output3[0]) > 0.1)
        return false;

    net.forward(input4);
    auto out4 = net.getOutputValues();
    if(fabs(out4[0] - output4[0]) > 0.1)
        return false;

    cout << "XOR SUCCESS!";

    return true;
}

bool testAND(){
    vector<int> layers{2, 5, 1};

    Net net(layers, 2000);
    cout << "INIT: \n";
    //cout << net.trace() << endl;

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
    vector<double> output2{0.0};
    vector<double> output3{0.0};
    vector<double> output4{1.0};
    outputs.push_back(output1);
    outputs.push_back(output2);
    outputs.push_back(output3);
    outputs.push_back(output4);

    net.train(inputs, outputs);

    //cout << "TRAINED: \n";
    //cout << net.trace(false) << endl;

    net.forward(input1);
    auto out = net.getOutputValues();
    if(fabs(out[0] - output1[0]) > 0.1)
        return false;

    net.forward(input2);
    auto out2 = net.getOutputValues();
    if(fabs(out2[0] - output2[0]) > 0.1)
        return false;

    net.forward(input3);
    auto out3 = net.getOutputValues();
    if(fabs(out3[0] - output3[0]) > 0.1)
        return false;

    net.forward(input4);
    auto out4 = net.getOutputValues();
    if(fabs(out4[0] - output4[0]) > 0.1)
        return false;

    cout << "AND SUCCESS!";

    return true;
}


int main() {
    bool ret = testXOR();

    ret |= testAND();

    return ret ? 0:1;
}
