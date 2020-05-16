//fd_cpml.cpp

/*
* Created by: Min Basnet
* 2020.April.14
* Kathmandu, Nepal
*/
#include <iostream>

#include "fd_cpml.cuh"
#include "INIReader.cuh"
//#include "../ext/inih/INIReader.cuh"
#include "util.cu"


int FDCpml::npml; // number of PML grids in each side
real_sim FDCpml::dh; // Spacing of the grid in PML layers
real_sim FDCpml::fpml; // Frequency of PML layer (Hz)
real_sim FDCpml::npower, FDCpml::k_max_PML;
real_sim FDCpml::rcoef; // Reflection coefficient
real_sim FDCpml::damp_v_PML; // Damping value

int FDCpml::tpml;
real_sim* FDCpml::a, * FDCpml::b, * FDCpml::K;// CPML parameters
real_sim* FDCpml::a_half, * FDCpml::b_half, * FDCpml::K_half; // CPML interpolated parameters  

void FDCpml::parse_configuration(const char* configuration_file_relative_path) {
    //
    std::cout << "parsing FD CPML from file '"
        << configuration_file_relative_path << "'." << std::endl;

    INIReader reader(configuration_file_relative_path);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load .ini file\n";
        exit(1);
    }

    // PML boundary (static members)
    npml = reader.GetInteger("PML_boundary", "npml", 10);
    dh = reader.GetReal("PML_boundary", "dh", 0.1);
    fpml = reader.GetReal("PML_boundary", "fpml", 20.0);

    npower = reader.GetReal("PML_boundary", "npower", 2.0);
    k_max_PML = reader.GetReal("PML_boundary", "k_max_PML", 1.0);
    rcoef = reader.GetReal("PML_boundary", "rcoef", 0.001);
    damp_v_PML = reader.GetReal("PML_boundary", "damp_v_PML", 0.0);

    tpml = 2 * npml + 2;
    std::cout << std::endl << "CPML:" << std::endl;
    std::cout << "npml = " << npml << ", dh = " << dh << ", fpml = " << fpml << ", npower = " << npower << std::endl;
    std::cout << "k_max_PML = " << k_max_PML << ", rcoef = " << rcoef << ", damp_v_PML = " << damp_v_PML << std::endl;
    std::cout << "tpml = " << tpml << std::endl;
    std::cout << std::endl;

}


void FDCpml::cpml_pro(real_sim dt) {
    // Computes the factors for application of PML boundary
    // Assumes that all sides have same reflection coefficients and number of grids and spacing

    // Local CPML variable
    const real_sim alpha_max_PML = 2.0 * PI * (fpml / 2.0); // from festa and Vilotte 
    real_sim thickness_PML, xorigin_PML, xval;
    real_sim abscissa_in_PML, abscissa_normalized;
    real_sim d0, d, d_half, alpha_prime, alpha_prime_half;


    thickness_PML = npml * dh; // Cartesian thickness of CPML layer in one side

    // compute d0 from INRIA report section 6.1 
    d0 = -(npower + 1) * damp_v_PML * log(rcoef) / (2.0 * thickness_PML);

    // Initialize the arrays to respective default values
    for (int i = 0; i <= (2 * npml + 2); i++) {
        a[i] = 0.0; a_half[i] = 0.0;
        b[i] = 1.0; b_half[i] = 1.0;
        K[i] = 1.0; K_half[i] = 1.0;
    }

    // Negative or left side of the boundary
    for (int i = 0; i <= npml; i++) {

        // Initialize and reset the variables
        abscissa_normalized = 0.0;
        d = 0; d_half = 0;
        alpha_prime = 0.0; alpha_prime_half = 0.0;


        // zero index in the start of PML Boundary and advancing towards the start of absorption
        xorigin_PML = thickness_PML;
        xval = dh * real_sim(i); // value of the absissa

        // define damping profile at the grid points 
        abscissa_in_PML = xorigin_PML - xval;
        if (abscissa_in_PML >= 0) {
            abscissa_normalized = abscissa_in_PML / thickness_PML;
            d = d0 * pow(abscissa_normalized, npower);

            // this taken from Gedney page 8.2 
            K[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized, npower);
            alpha_prime = alpha_max_PML * (1.0 - abscissa_normalized);
        }

        // define damping profile at half the grid points 
        abscissa_in_PML = xorigin_PML - (xval + dh / 2.0);
        if (abscissa_in_PML >= 0) {
            abscissa_normalized = abscissa_in_PML / thickness_PML;
            d_half = d0 * pow(abscissa_normalized, npower);

            // this taken from Gedney page 8.2 
            K_half[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized, npower);
            alpha_prime_half = alpha_max_PML * (1.0 - abscissa_normalized);
        }

        // just in case, for -0.5 at the end 
        if (alpha_prime < 0.0) alpha_prime = 0.0;
        if (alpha_prime_half < 0.0) alpha_prime_half = 0.0;

        b[i] = exp(-(d / K[i] + alpha_prime) * dt);
        b_half[i] = exp(-(d_half / K_half[i] + alpha_prime_half) * dt);

        // avoid division by zero outside the PML 
        if (fabs(d) > 1.0e-6) {
            a[i] = d * (b[i] - 1.0) / (K[i] * (d + K[i] * alpha_prime));
        }

        if (fabs(d_half) > 1.0e-6) {
            a_half[i] = d_half * (b_half[i] - 1.0) /
                (K_half[i] * (d_half + K_half[i] * alpha_prime_half));
        }

    } // Negative side of the boundary completed

    // ---------------------------------------------------
    // ---------------------------------------------------

    // Positive or right side of the boundary
    for (int i = npml + 1; i < npml * 2 + 2; i++) {

        // Initialize and reset the variables
        abscissa_normalized = 0.0;
        d = 0; d_half = 0;
        alpha_prime = 0.0; alpha_prime_half = 0.0;

        // zero index in the start of PML decay and advancing towards the end
        xorigin_PML = dh * (npml + 1);
        xval = dh * real_sim(i); // value of the absissa

        // define damping profile at the grid points 
        abscissa_in_PML = xval - xorigin_PML;
        if (abscissa_in_PML >= 0) {
            abscissa_normalized = abscissa_in_PML / thickness_PML;
            d = d0 * pow(abscissa_normalized, npower);

            // this taken from Gedney page 8.2 
            K[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized, npower);
            alpha_prime = alpha_max_PML * (1.0 - abscissa_normalized);
        }

        // define damping profile at half the grid points 
        abscissa_in_PML = (xval + dh / 2.0) - xorigin_PML;
        if (abscissa_in_PML >= 0) {
            abscissa_normalized = abscissa_in_PML / thickness_PML;
            d_half = d0 * pow(abscissa_normalized, npower);

            // this taken from Gedney page 8.2 
            K_half[i] = 1.0 + (k_max_PML - 1.0) * pow(abscissa_normalized, npower);
            alpha_prime_half = alpha_max_PML * (1.0 - abscissa_normalized);
        }

        // just in case, for -0.5 at the end 
        if (alpha_prime < 0.0) alpha_prime = 0.0;
        if (alpha_prime_half < 0.0) alpha_prime_half = 0.0;

        b[i] = exp(-(d / K[i] + alpha_prime) * dt);
        b_half[i] = exp(-(d_half / K_half[i] + alpha_prime_half) * dt);

        // avoid division by zero outside the PML 
        if (fabs(d) > 1.0e-6) {
            a[i] = d * (b[i] - 1.0) / (K[i] * (d + K[i] * alpha_prime));
        }

        if (fabs(d_half) > 1.0e-6) {
            a_half[i] = d_half * (b_half[i] - 1.0) /
                (K_half[i] * (d_half + K_half[i] * alpha_prime_half));
        }

    } // Positive side of the boundary completed

}


void FDCpml::allocate_pml() {
    // Allocates the arrays that are needed for later computations

    std::cout << "Allocating PML factors." << std::endl;

    allocate_array_1d(a, 2 * (npml + 1));
    allocate_array_1d(b, 2 * (npml + 1));
    allocate_array_1d(K, 2 * (npml + 1));

    allocate_array_1d(a_half, 2 * (npml + 1));
    allocate_array_1d(b_half, 2 * (npml + 1));
    allocate_array_1d(K_half, 2 * (npml + 1));

    std::cout << "PML factors allocated." << std::endl;
}

void FDCpml::allocate_memory_arrays(int dimz, int dimx) {
    // Allocates memory arrays for velocity and stress derivatives

    std::cout << "Allocating PML memory arrays." << std::endl;
    // Allocate velocity derivatives memory
    allocate_array_2d(mem_vx_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_vz_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_vx_z, 2 * (npml + 1), dimx);
    allocate_array_2d(mem_vz_z, 2 * (npml + 1), dimx);

    // Allocate stress derivatives memory
    allocate_array_2d(mem_sxx_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_szx_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_szx_z, 2 * (npml + 1), dimx);
    allocate_array_2d(mem_szz_z, 2 * (npml + 1), dimx);

    std::cout << "PML memory arrays allocated." << std::endl;

}


void FDCpml::deallocate_pml() {
    // Deallocates the static PML arrays

    std::cout << "Deallocating PML factors." << std::endl;
    deallocate_array_1d(a);
    deallocate_array_1d(b);
    deallocate_array_1d(K);

    deallocate_array_1d(a_half);
    deallocate_array_1d(b_half);
    deallocate_array_1d(K_half);

    std::cout << "PML factors deallocated." << std::endl;
}


void FDCpml::deallocate_memory_arrays(int dimz) {
    // Deallocates memory arrays for velocity and stress derivatives

    std::cout << "Deallocating PML memory arrays." << std::endl;
    // Deallocate velocity derivatives memory
    deallocate_array_2d(mem_vx_x, dimz);
    deallocate_array_2d(mem_vz_x, dimz);
    deallocate_array_2d(mem_vx_z, tpml);
    deallocate_array_2d(mem_vz_z, tpml);

    // Deallocate stress derivatives memory
    deallocate_array_2d(mem_sxx_x, dimz);
    deallocate_array_2d(mem_szx_x, dimz);
    deallocate_array_2d(mem_szx_z, tpml);
    deallocate_array_2d(mem_szz_z, tpml);

    std::cout << "PML memory arrays deallocated." << std::endl;

}

void allocate_pml_memory_arrays(int npml, int dimz, int dimx,
    real_sim** mem_vx_x, real_sim** mem_vz_x,
    real_sim** mem_vx_z, real_sim** mem_vz_z,
    real_sim** mem_sxx_x, real_sim** mem_szx_x,
    real_sim** mem_szx_z, real_sim** mem_szz_z) {
    // Allocates memory arrays for velocity and stress derivatives

    std::cout << "Allocating PML memory arrays." << std::endl;
    // Allocate velocity derivatives memory
    allocate_array_2d(mem_vx_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_vz_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_vx_z, 2 * (npml + 1), dimx);
    allocate_array_2d(mem_vz_z, 2 * (npml + 1), dimx);

    // Allocate stress derivatives memory
    allocate_array_2d(mem_sxx_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_szx_x, dimz, 2 * (npml + 1));
    allocate_array_2d(mem_szx_z, 2 * (npml + 1), dimx);
    allocate_array_2d(mem_szz_z, 2 * (npml + 1), dimx);

    std::cout << "PML memory arrays allocated." << std::endl;

}

void deallocate_pml_memory_arrays(int npml, int dimz,
    real_sim** mem_vx_x, real_sim** mem_vz_x,
    real_sim** mem_vx_z, real_sim** mem_vz_z,
    real_sim** mem_sxx_x, real_sim** mem_szx_x,
    real_sim** mem_szx_z, real_sim** mem_szz_z) {
    // Allocates memory arrays for velocity and stress derivatives

    std::cout << "Deallocating PML memory arrays." << std::endl;
    // Allocate velocity derivatives memory
    deallocate_array_2d(mem_vx_x, dimz);
    deallocate_array_2d(mem_vz_x, dimz);
    deallocate_array_2d(mem_vx_z, 2 * (npml + 1));
    deallocate_array_2d(mem_vz_z, 2 * (npml + 1));

    // Allocate stress derivatives memory
    deallocate_array_2d(mem_sxx_x, dimz);
    deallocate_array_2d(mem_szx_x, dimz);
    deallocate_array_2d(mem_szx_z, 2 * (npml + 1));
    deallocate_array_2d(mem_szz_z, 2 * (npml + 1));

    std::cout << "PML memory arrays deallocated." << std::endl;

}