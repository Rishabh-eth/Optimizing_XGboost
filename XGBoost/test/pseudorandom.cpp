//
// Created by Amory Hoste on 13/04/2020.
//

#include "pseudorandom.h"

# define SEED 0
# define MIN_RAND -10000
# define MAX_RAND 100000

std::mt19937& pseudorandom::get_generator() {
    static std::mt19937 generator(SEED);
    return generator;
}

double  pseudorandom::get_random_double(std::mt19937& generator) {
    // Initialized upon first call to the function.
    static std::uniform_real_distribution<> random_double(MIN_RAND, MAX_RAND);
    return random_double(generator);
}
