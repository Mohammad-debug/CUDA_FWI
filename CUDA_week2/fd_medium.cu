//fd_grid.cpp

/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/
#include <iostream>
#include <cmath>

#include "fd_medium.cuh"
#include "INIReader.cuh"
//#include "../ext/inih/INIReader.cuh"
#include "util.cu"


void FDMedium::parse_configuration(const char* configuration_file_relative_path) {
    //
    std::cout << "parsing FD MEDIUM from file '"
        << configuration_file_relative_path << "'." << std::endl;

    INIReader reader(configuration_file_relative_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load .ini file\n";
        exit(1);
    }
    // Default medium Static Material Parameters
    scalar_rho = reader.GetReal("medium", "scalar_rho", 0.0);
    scalar_vp = reader.GetReal("medium", "scalar_vp", 0.0);
    scalar_vs = reader.GetReal("medium", "scalar_vs", 0.0);

    velocity_to_lame(); // compute scalar Lame paramaters

    std::cout << std::endl << "Medium:" << std::endl;
    std::cout << "density = " << scalar_rho << std::endl;
    std::cout << "Vp = " << scalar_vp << ", Vs = " << scalar_vs << std::endl;
    std::cout << "Lamé's parameters: lamda = " << scalar_lam << ", mu = " << scalar_mu << std::endl;
    std::cout << std::endl;

}

void FDMedium::velocity_to_lame() {
    // Produces material grid of rho, mu and lamda
    scalar_mu = real_sim(pow(scalar_vs, 2)) * scalar_rho;
    scalar_lam = real_sim(pow(scalar_vp, 2)) * scalar_rho - 2.0 * scalar_mu;

}


void FDMedium::stagger_over_grid(int dimz, int dimx) {
    // Prepares Lame parameters and density over staggered grid
    // dimx and dimz should include grids required for pml and fd order

    for (int ix = 0; ix < dimx; ix++) {
        for (int iz = 0; iz < dimz; iz++) {

            rho[iz][ix] = scalar_rho;
            lam[iz][ix] = scalar_lam;
            mu[iz][ix] = scalar_mu;
        }
    }
}

void FDMedium::average_parameters(int dimz, int dimx) {
    // Harmonic 2d average of mu and
    // Arithmatic 1d average of rho

    for (int ix = 0; ix < dimx; ix++) {
        for (int iz = 0; iz < dimz; iz++) {

            // Harmonic average for mu
            mu_zx[iz][ix] = 4.0 / ((1.0 / mu[iz][ix]) +
                (1.0 / mu[iz][ix + 1]) + (1.0 / mu[iz + 1][ix]) + (1.0 / mu[iz + 1][ix + 1]));

            if ((mu[iz][ix] == 0.0) || (mu[iz][ix + 1] == 0.0) || (mu[iz + 1][ix] == 0.0) || (mu[iz + 1][ix + 1] == 0.0)) {
                mu_zx[iz][ix] = 0.0;
            }

            // Arithmatic average of rho
            // the averages are inversed for computational efficiency       
            rho_zp[iz][ix] = 1.0 / (0.5 * (rho[iz][ix] + rho[iz + 1][ix]));
            rho_xp[iz][ix] = 1.0 / (0.5 * (rho[iz][ix] + rho[iz][ix + 1]));

            if ((rho[iz][ix] < 1e-4) && (rho[iz + 1][ix] < 1e-4)) {
                rho_zp[iz][ix] = 0.0;
            }

            if ((rho[iz][ix] < 1e-4) && (rho[iz][ix + 1] < 1e-4)) {
                rho_zp[iz][ix] = 0.0;
            }

        }
    }

}

void FDMedium::allocate_medium(int dimz, int dimx) {
    // Allocates the basic material parameters (Lame parameters)
    allocate_array_2d(rho, dimz, dimx);
    allocate_array_2d(lam, dimz, dimx);
    allocate_array_2d(mu, dimz, dimx);
}


void FDMedium::allocate_medium_av(int dimz, int dimx) {
    // Allocates the averaged material parameters 
    allocate_array_2d(rho_zp, dimz, dimx);
    allocate_array_2d(rho_xp, dimz, dimx);
    allocate_array_2d(mu_zx, dimz, dimx);
}

void FDMedium::deallocate_medium(int dimz) {
    // Allocates the basic material parameters (Lame parameters)
    deallocate_array_2d(rho, dimz);
    deallocate_array_2d(lam, dimz);
    deallocate_array_2d(mu, dimz);
}

void FDMedium::deallocate_medium_av(int dimz) {
    // Allocates the averaged material parameters 
    deallocate_array_2d(rho_zp, dimz);
    deallocate_array_2d(rho_xp, dimz);
    deallocate_array_2d(mu_zx, dimz);
}