//
// Created by Wayne Tran on 2016-03-27.
//
#include "Common.h"

#pragma once

class Utils {
public:
    template<typename T>
    static std::string toString(std::vector<T> vec) {
        std::stringstream ss;
        for (auto e: vec) {
            ss << e << " ";
        }
        return ss.str();
    }
};

class MathUtils {
public:
    static double sigmoid(double x) {
        return 1.0 / (1 + exp(-x));
    }

    static double sigmoidPrime(double x) {
        //return sigmoid(x)*(1-sigmoid(x));
        return exp(-x) / pow(1 + exp(-x), 2);  //Simplified
    }

    static double sigmoidPrimePreCalc(double sigmoidResult) {
        return sigmoidResult * (1 - sigmoidResult);
    }

    //Calculate the errorGradient of the value of an output neuron
    static double errorGradOutput(double actual, double expected) {
        return sigmoidPrimePreCalc(actual) * (expected - actual);
    }

    //Calculate the errorGradient of an internal neuron's value
    static double errorGrad(double actual, double weightedErrorSum) {
        return sigmoidPrimePreCalc(actual) * weightedErrorSum;
    }

    static double mean(std::vector<double> results) {
        if (results.empty())
            return 0.0;

        double sum = 0.0;
        for (auto n: results) {
            sum += n;
        }
        sum /= results.size();
        return sum;
    }

    static double meanSquaredError(std::vector<double> result,
                                   std::vector<double> expected) {
        if (result.empty() || expected.empty() ||
            result.size() != expected.size()) {
            return 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i < result.size(); ++i) {
            sum += pow(expected[i] - result[i], 2);
        }
        sum /= result.size();
        return sum;
    }

    static double meanError(std::vector<double> result,
                            std::vector<double> expected) {
        if (result.empty() || expected.empty() ||
            result.size() != expected.size()) {
            return 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i < result.size(); ++i) {
            sum += abs(expected[i] - result[i]);
        }
        sum /= result.size();
        return sum;
    }

    static double clamp(double val, double min = -1, double max = 1) {
        return val > max ? max : (val < min ? min : val);
    }


    static int randomInt(int Low, int High) {
        return rand() % (High - Low + 1) + Low;
    }

    static double randomDouble(double Low, double High) {
        return ((double) rand() / RAND_MAX) * (High - Low) + Low;
    }
};