//
// Created by Amory Hoste on 13/04/2020.
//

#ifndef XGBOOST_PSEUDORANDOM_H
#define XGBOOST_PSEUDORANDOM_H

#include <random>

namespace pseudorandom {

    std::mt19937& get_generator();

    double get_random_double(std::mt19937& generator);
}

#endif //XGBOOST_PSEUDORANDOM_H
