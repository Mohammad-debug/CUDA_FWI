#include <iostream>
#include <fstream>
#include "globvar.cuh"
#include "fd_wave_model.cuh"

int main(int argc, char** argv) {
    std::cout << "Maximum amount of OpenMP threads:" << std::endl << std::endl;

    auto configuration_file = argv[1];

    auto* model = new FDWaveModel(configuration_file);

    if (model->fwinv) {
        model->fwi_simulate();
    }
    else {
        model->forward_simulate();
    }


  // delete model;//This was giving error after kernel execution, while deallocation
    std::cout << argc << std::endl << "PROGRAM COMPLETE." << std::endl;

    exit(0);
}
