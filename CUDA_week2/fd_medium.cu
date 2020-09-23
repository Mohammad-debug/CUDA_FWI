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

//fd_grid.cpp

/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/
#include <iostream>
#include <fstream>
#include <cmath>



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

    std::ofstream outFile; // file to print vz arrays

    for (int ix = 0; ix < dimx; ix++) {
        for (int iz = 0; iz < dimz; iz++) {

            rho[iz][ix] = scalar_rho;
            lam[iz][ix] = scalar_lam;
            mu[iz][ix] = scalar_mu;
        }
    }
}

void FDMedium::stagger_over_grid_modify(int dimz, int dimx) {
    // Prepares Lame parameters and density over staggered grid
    // dimx and dimz should include grids required for pml and fd order

    std::cout << "WARNING: The medium properties are modified." << std::endl;

    std::ofstream outFile; // file to print vz arrays

    for (int ix = 0; ix < dimx; ix++) {
        for (int iz = 0; iz < dimz; iz++) {

            rho[iz][ix] = scalar_rho;
            lam[iz][ix] = scalar_lam;
            mu[iz][ix] = scalar_mu;

            // Overiding maerial for certain area  to make a circle of harder material
            if (((iz - dimz / 2) * (iz - dimz / 2) + (ix - dimx / 2) * (ix - dimx / 2)) < dimz * dimz / 49) {
                //if ((iz >0.4*dimz && iz <= 0.6*dimz) && (ix >0.4*dimx && ix <= 0.6*dimx)) {
                //if (iz>0.4*dimz && iz<0.6*dimz){
                lam[iz][ix] = 1.3 * scalar_lam;
                //rho[iz][ix] = 1.3 * scalar_rho;
                //lam[iz][ix] = 332.0 * 332.0 * 1.25;
               // mu[iz][ix] = 0.0;


              // overwrite material with properties of air for a crack
             // if ((ix >0.4*dimz && ix <= 0.6*dimz) && (iz == 400 && iz == 400)) {
              //  rho[iz][ix] = 1.25;
               // lam[iz][ix] = 332.0 * 332.0 * 1.25;
               // mu[iz][ix] = 0.0;

            }
        }
    }

    // Printing AASCI data to plot lamda 

    outFile.open("./io/snap_data/lamda_snap.csv");
    for (int j = 0; j < dimz; j++) {
        for (int i = 0; i < dimx; i++) {
            outFile << rho[j][i] << ", ";
        }
        outFile << std::endl;
    }
    outFile.close();


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

void FDMedium::fwi_grad_update(real_sim** mat, real_sim** mat_new, real_sim** grad_mat,
    real_sim step_length, int dimz, int dimx,
    int z1, int x1, int fwi_dz, int fwi_dx, bool override, int matType) {

    // Finding the maximum of the parameter and gradient
    real_sim mat_max = 0.0;
    real_sim grad_max = 0.0;

    real_sim grad_sum = 0.0;
    real_sim mat_sum = 0.0;
    real_sim new_mat_sum = 0.0;

    real_sim mat_upper_limit;
    real_sim mat_lower_limit;

    if (matType == 1) {
        mat_upper_limit = 3000;
        mat_lower_limit = 1.5;
    }
    else if (matType == 2) {
        mat_upper_limit = 48.0e+9;
        mat_lower_limit = 0;
    }
    else if (matType == 3) {
        mat_upper_limit = 27.0e+9;
        mat_lower_limit = 0;
    }


    // finding the maximum of the gradient
    for (int iz = z1; iz < dimz; iz += fwi_dz) {
        for (int ix = x1; ix < dimx; ix += fwi_dx) {
            if (mat[iz][ix] > mat_max) mat_max = mat[iz][ix];
            if (fabs(grad_mat[iz][ix]) > grad_max) grad_max = grad_mat[iz][ix];
        }
    }

    // step length normalized locally
    real_sim step_length_norm = step_length * (mat_max / grad_max);

    // Material update after fwi gradient calculation
    for (int iz = z1; iz < dimz; iz += fwi_dz) {
        for (int ix = x1; ix < dimx; ix += fwi_dx) {
            mat_new[iz][ix] = mat[iz][ix] + step_length_norm * grad_mat[iz][ix];


            if (mat_new[iz][ix] < mat_lower_limit) mat_new[iz][ix] = mat_lower_limit;
            if (mat_new[iz][ix] > mat_upper_limit) mat_new[iz][ix] = mat_upper_limit;


            // for check of the values
            grad_sum += fabs(grad_mat[iz][ix]);
            mat_sum += fabs(mat[iz][ix]);
            new_mat_sum += fabs(mat_new[iz][ix]);

            // Avoid negative values
            if (mat_new[iz][ix] < 0.0) mat_new[iz][ix] = mat[iz][ix];
        }
    }

    // Upper and lower limit to be bracketed later ** (to be added)

    if (override == true) {
        std::cout << "Warning: Old material overwrite" << std::endl;
        // Old material to be renewed as new material now
        for (int iz = z1; iz < dimz; iz += fwi_dz) {
            for (int ix = x1; ix < dimx; ix += fwi_dx) {
                mat[iz][ix] = mat_new[iz][ix];

            }
        }
    }
    std::cout << std::endl << "MATERIAL UPDATE: " << std::endl;
    std::cout << "Step length = " << step_length
        << ", normalized step length = " << step_length_norm
        << ", mat_max = " << mat_max << ", grad_max = " << grad_max << std::endl;
    //std::cout << "MAT change: mat = " << mat_sum << ", grad = " << grad_sum << ", new mat = "
    //<< new_mat_sum << std::endl;

}


void FDMedium::allocate_medium(int dimz, int dimx) {
    // Allocates the basic material parameters (Lame parameters)
    allocate_array_2d(rho, dimz, dimx);
    allocate_array_2d(lam, dimz, dimx);
    allocate_array_2d(mu, dimz, dimx);
    allocate_array_2d(rho_old, dimz, dimx);
    allocate_array_2d(lam_old, dimz, dimx);
    allocate_array_2d(mu_old, dimz, dimx);
}

void FDMedium::medium_average(real_sim& C_lam, real_sim& C_mu, real_sim& C_rho, int dimz, int dimx) {

    // Calculates average for medium parameters

    C_lam = 0.0;
    C_mu = 0.0;
    C_rho = 0.0;

    for (int iz = 0; iz < dimz; iz++) {
        for (int ix = 0; ix < dimx; ix++) {

            C_lam += lam[iz][ix];
            C_mu += mu[iz][ix];
            C_rho += rho[iz][ix];

        }
    }

    C_lam = C_lam / (dimz * dimx);
    C_mu = C_mu / (dimz * dimx);
    C_rho = C_rho / (dimz * dimx);

    std::cout << "The value of Density inside: " << C_rho << std::endl;

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
    deallocate_array_2d(rho_old, dimz);
    deallocate_array_2d(lam_old, dimz);
    deallocate_array_2d(mu_old, dimz);

}

void FDMedium::deallocate_medium_av(int dimz) {
    // Allocates the averaged material parameters 
    deallocate_array_2d(rho_zp, dimz);
    deallocate_array_2d(rho_xp, dimz);
    deallocate_array_2d(mu_zx, dimz);
}