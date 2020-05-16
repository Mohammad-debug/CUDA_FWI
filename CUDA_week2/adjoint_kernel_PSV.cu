//forward_kernel_PSV.cpp

/*
* Created by: Min Basnet
* 2020.April.16
* Kathmandu, Nepal
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "globvar.cuh"
#include "util.cu"
#include "fd_cpml.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__global__ void kernel_I(int tf, int fwi_dt, int dt, int nzt, int fwi_z1, int fwi_z2, int fwi_x1, int fwi_x2,
    int fwi_dz, int fwi_dx, int nft, int nfz, int nfx,
    real_sim* fwi_sxx, real_sim* fwi_szx, real_sim* fwi_szz, real_sim* fwi_vx, real_sim*
    fwi_vz, real_sim* sxx, real_sim* szx, real_sim* szz, real_sim* vx, real_sim* vz, real_sim* mu,
    real_sim* lam, real_sim* grad_lam, real_sim* grad_mu, real_sim* grad_rho) {

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    real_sim s1, s2, s3, s4;
    if (iz >= fwi_z1 && iz < fwi_z2 && iz % fwi_dz == fwi_z1) { // storing only a portion and with grid inteval
        int zf = (iz - fwi_z1) / fwi_dz; // z index for fwi gradient storage

        if (ix >= fwi_x1 && ix < fwi_x2 && ix % fwi_dx == fwi_x1) {
            int xf = (ix - fwi_x1) / fwi_dx; // x index for fwi gradient storage
            int offset = tf * nft * nfz + zf * nfz + xf;


            s1 = (fwi_sxx[offset] + fwi_szz[offset]) * (sxx[iz * nzt + ix] + szz[iz * nzt + ix])
                * 0.25 / ((lam[iz * nzt + ix] + mu[iz * nzt + ix]) * (lam[iz * nzt + ix] + mu[iz * nzt + ix]));

            s2 = (fwi_sxx[offset] - fwi_szz[offset]) * (sxx[iz * nzt + ix] - szz[iz * nzt + ix])
                / (mu[iz * nzt + ix] * mu[iz * nzt + ix]);

            s3 = (fwi_szx[offset] * szx[iz * nzt + ix]) / (mu[iz * nzt + ix] * mu[iz * nzt + ix]);

            // The time derivatives of the velocity may have to be computed differently
            s4 = vx[iz * nzt + ix] * fwi_vx[offset] + vz[iz * nzt + ix] * fwi_vz[offset];

            grad_lam[zf * nfz + xf] += fwi_dt * dt * s1;
            grad_mu[zf * nfz + xf] += fwi_dt * dt * (s3 + s1 + s2);
            grad_rho[zf * nfz + xf] += fwi_dt * dt * s4;


        }
    }
    else {
        return;
    }
}

__global__ void kernel_II(int ishot, int nt, int nzt, int nxt, int fpad, int ppad, real_sim dt, real_sim dx, real_sim dz,
    int fdorder, real_sim* vx, real_sim* vz, real_sim* sxx,
    real_sim* szx, real_sim* szz, real_sim* lam, real_sim* mu,
    real_sim* mu_zx, real_sim* rho_zp, real_sim* rho_xp, int npml,
    real_sim* a, real_sim* b, real_sim* K, real_sim* a_half, real_sim* b_half, real_sim* K_half,
    real_sim* mem_vx_x, real_sim* mem_vx_z, real_sim* mem_vz_x, real_sim* mem_vz_z,
    real_sim* mem_sxx_x, real_sim* mem_szx_x, real_sim* mem_szz_z, real_sim* mem_szx_z, bool fsurf) {

    //**********************************************************************************
    real_sim sxx_x, szx_x, szx_z, szz_z; // spatial stress derivatives
    real_sim vx_x, vx_z, vz_x, vz_z; // spatial velocity derivatives
    int nz1, nz2, nx1, nx2; // The computational grid boundaries
    int px, pz; // index for PML arrys
    int isnap; // to take snapshots for data storage

    int tf, zf, xf; // Index parameters for fwi data storage

    real_sim hc[2] = { 1.0, 1.0 };

    // Initial calculation of indices
    //---------------------------------------------
    nz1 = fpad; nz2 = nzt - fpad; nx1 = fpad; nx2 = nxt - fpad; // index variables

    // index to map PML at the right or positive end
    int pnx, pnz;
    pnx = nxt - 2 * ppad + fpad - 1; // nx + ppad + npml + 1 and nx = nxt - 2*ppad

    if (fsurf) {
        pnz = nzt - 2 * ppad - 1; // nz + ppad + npml + 1 and nz = nzt - ppad - fpad
    }
    else {
        pnz = nzt - 2 * ppad + fpad - 1; // nz + ppad + npml + 1 and nz = nzt - 2*ppad
    }

    real_sim dxi = 1.0 / dx; real_sim dzi = 1.0 / dz; // inverse of dx and dz
//**********************************************************************************************************


    //int k = blockIdx.x * blockDim.x + threadIdx.x;
    //int ix = k % nx2;
    //int iz = k % nz2;
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;

    //if (threadIdx.x == 0 && blockIdx.x == 0)
    //    printf("Debug Params : . ..");

    ////***********************************************
    //switch (fdorder) {
    //case(2): {

    //    // Calculate spatial velocity derivatives


    switch (fdorder) {
    case(2):
        if (ix < nx2 && ix >= nx1 && iz >= nz1 && iz < nz2)
        {

            // Calculate spatial velocity derivatives
            vx_x = dxi * hc[1] * (vx[iz * nzt + ix] - vx[iz * nzt + (ix - 1)]);
            vz_x = dxi * hc[1] * (vz[iz * nzt + (ix + 1)] - vz[iz * nzt + ix]);
            vx_z = dzi * hc[1] * (vx[(iz + 1) * nzt + ix] - vx[iz * nzt + ix]);
            vz_z = dzi * hc[1] * (vz[iz * nzt + ix] - vz[(iz - 1) * nzt + ix]);
           
    // ---------------------------------------------------
    // CPML layers for stress tensor kernel
    // ---------------------------------------------------



            if (npml > 0) {

                if (ix >= fpad && ix <= ppad)
                { // left CPML
                  // Mapping the static CPML and memory variables to
                    px = ix - fpad; // the memory array index
                    //std::cout << std::endl << "Fault1 " << ix << std::endl;


                    mem_vx_x[iz * nzt + px] = b[px] * mem_vx_x[iz * nzt + px] + a[px] * vx_x;
                    mem_vz_x[iz * nzt + px] = b_half[px] * mem_vz_x[iz * nzt + px] + a_half[px] * vz_x;

                    vx_x = vx_x / K[px] + mem_vx_x[iz * nzt + px];
                    vz_x = vz_x / K_half[px] + mem_vz_x[iz * nzt + px];

                } // cpml left



                if (ix >= (nxt - ppad - 1) && ix < nxt - fpad) { // right CPML
                  // Mapping the static CPML and memory variables to

                    px = ix - pnx; // The PML factors index

                    mem_vx_x[iz * nzt + px] = b[px] * mem_vx_x[iz * nzt + px] + a[px] * vx_x;
                    mem_vz_x[iz * nzt + px] = b_half[px] * mem_vz_x[iz * nzt + px] + a_half[px] * vz_x;

                    vx_x = vx_x / K[px] + mem_vx_x[iz * nzt + px];
                    vz_x = vz_x / K_half[px] + mem_vz_x[iz * nzt + px];


                } // cpml right


                if (iz >= fpad && iz <= ppad && !fsurf) { // top CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - fpad; // the memory array index

                    mem_vz_z[pz * 2 * (npml + 1) + ix] = b[pz] * mem_vz_z[pz * 2 * (npml + 1) + ix] + a[pz] * vz_z;
                    mem_vx_z[pz * 2 * (npml + 1) + ix] = b_half[pz] * mem_vx_z[pz * 2 * (npml + 1) + ix] + a_half[pz] * vx_z;

                    vz_z = vz_z / K[pz] + mem_vz_z[pz * 2 * (npml + 1) + ix];
                    vx_z = vx_z / K_half[pz] + mem_vx_z[pz * 2 * (npml + 1) + ix];

                    //std::cout << pz<< ", ";

                } // cpml top



                if (iz >= (nzt - ppad - 1) && iz < nzt - fpad) { // bottom CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - pnz; // The PML factors index

                    mem_vz_z[pz * 2 * (npml + 1) + ix] = b[pz] * mem_vz_z[pz * 2 * (npml + 1) + ix] + a[pz] * vz_z;
                    mem_vx_z[pz * 2 * (npml + 1) + ix] = b_half[pz] * mem_vx_z[pz * 2 * (npml + 1) + ix] + a_half[pz] * vx_z;

                    vz_z = vz_z / K[pz] + mem_vz_z[pz * 2 * (npml + 1) + ix];
                    vx_z = vx_z / K_half[pz] + mem_vx_z[pz * 2 * (npml + 1) + ix];

                    //std::cout << pz<< ", ";

                } // cpml bottom

            }    // npml>0
            __syncthreads();
            //// --------------------------------------------------------------------------
            //// --------------------------------------------------------------------------

            ////// updating stresses

            szx[iz * nzt + ix] += dt * mu_zx[iz * (nzt - 1) + ix] * (vz_x + vx_z);
            sxx[iz * nzt + ix] += dt * (lam[iz * nzt + ix] * (vx_x + vz_z) + (2.0f * mu[iz * nzt + ix] * vx_x));
            szz[iz * nzt + ix] += dt * (lam[iz * nzt + ix] * (vx_x + vz_z) + (2.0f * mu[iz * nzt + ix] * vz_z));


            // -----------------------------------------
   // ----------------------------------------- 
   // Override stress for free surface implementation

            if (fsurf && iz == fpad) {
                // Free surface at z = 0 or nz = fpad

                // Denise manual  page 13
                szz[fpad * nzt + ix] = 0.0;
                szx[fpad * nzt + ix] = 0.0;
                sxx[fpad * nzt + ix] = 4.0 * dt * vx_x * (lam[fpad * nzt + ix] * mu[fpad * nzt + ix] + mu[fpad * nzt + ix] * mu[fpad * nzt + ix])
                    / (lam[fpad * nzt + ix] + 2.0 * mu[fpad * nzt + ix]);

            }

        }
        else {
            return;
        }

    default:
        //  std::cout << "FDORDER = " << fdorder << npml << std::endl;
        //  std::cout << "FD order mismatch. <SIMULATION ABORTED>" << std::endl;
          //exit(0);
       // printf("FD order mismatch. <SIMULATION ABORTED>\n");
    } // end of switch


}

__global__ void kernel_III(int ishot, int nt, int nzt, int nxt, int fpad, int ppad, real_sim dt, real_sim dx, real_sim dz,
    int fdorder, real_sim* vx, real_sim* vz, real_sim* sxx,
    real_sim* szx, real_sim* szz, real_sim* lam, real_sim* mu,
    real_sim* mu_zx, real_sim* rho_zp, real_sim* rho_xp, int npml,
    real_sim* a, real_sim* b, real_sim* K, real_sim* a_half, real_sim* b_half, real_sim* K_half,
    real_sim* mem_vx_x, real_sim* mem_vx_z, real_sim* mem_vz_x, real_sim* mem_vz_z,
    real_sim* mem_sxx_x, real_sim* mem_szx_x, real_sim* mem_szz_z, real_sim* mem_szx_z, bool fsurf) {


    real_sim sxx_x, szx_x, szx_z, szz_z; // spatial stress derivatives
    real_sim vx_x, vx_z, vz_x, vz_z; // spatial velocity derivatives
    int nz1, nz2, nx1, nx2; // The computational grid boundaries
    int px, pz; // index for PML arrys
    int isnap; // to take snapshots for data storage

    int tf, zf, xf; // Index parameters for fwi data storage


    real_sim hc[2] = { 1.0, 1.0 };

    // Initial calculation of indices
    //---------------------------------------------
    nz1 = fpad; nz2 = nzt - fpad; nx1 = fpad; nx2 = nxt - fpad; // index variables

    // index to map PML at the right or positive end
    int pnx, pnz;
    pnx = nxt - 2 * ppad + fpad - 1; // nx + ppad + npml + 1 and nx = nxt - 2*ppad

    if (fsurf) {
        pnz = nzt - 2 * ppad - 1; // nz + ppad + npml + 1 and nz = nzt - ppad - fpad
    }
    else {
        pnz = nzt - 2 * ppad + fpad - 1; // nz + ppad + npml + 1 and nz = nzt - 2*ppad
    }

    real_sim dxi = 1.0 / dx; real_sim dzi = 1.0 / dz; // inverse of dx and dz


    //*************************************************

    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;


    switch (fdorder) {
    case(2):
        if (ix < nx2 && ix >= nx1 && iz >= nz1 && iz < nz2)
        {
            // printf("abc");
            // compute spatial stress derivatives
            sxx_x = dxi * hc[1] * (sxx[iz * nzt + ix + 1] - sxx[iz * nzt + ix]);
            szx_z = dxi * hc[1] * (szx[iz * nzt + ix] - szx[(iz - 1) * nzt + ix]);

            szx_x = dzi * hc[1] * (szx[iz * nzt + ix] - szx[iz * nzt + ix - 1]);
            szz_z = dzi * hc[1] * (szz[(iz + 1) * nzt + ix] - szz[iz * nzt + ix]);


            // ---------------------------------------------------
            // CPML layers for particle velocity kernel
            // ---------------------------------------------------

            if (npml > 0) {

                if (ix >= fpad && ix < ppad) { // left CPML
                  // Mapping the static CPML and memory variables to
                    px = ix - fpad; // the memory array index

                    mem_sxx_x[iz * nzt + px] = b[px] * mem_sxx_x[iz * nzt + px] + a[px] * sxx_x;
                    mem_szx_x[iz * nzt + px] = b_half[px] * mem_szx_x[iz * nzt + px] + a_half[px] * szx_x;

                    sxx_x = sxx_x / K[px] + mem_sxx_x[iz * nzt + px];
                    szx_x = szx_x / K_half[px] + mem_szx_x[iz * nzt + px];

                } // cpml left

                if (ix >= (nxt - ppad - 1) && ix < (nxt - fpad)) { // right CPML
                  // Mapping the static CPML and memory variables to
                    px = ix - pnx; // The PML factors index

                    mem_sxx_x[iz * nzt + px] = b[px] * mem_sxx_x[iz * nzt + px] + a[px] * sxx_x;
                    mem_szx_x[iz * nzt + px] = b_half[px] * mem_szx_x[iz * nzt + px] + a_half[px] * szx_x;

                    sxx_x = sxx_x / K[px] + mem_sxx_x[iz * nzt + px];
                    szx_x = szx_x / K_half[px] + mem_szx_x[iz * nzt + px];

                } // cpml right


                if (iz >= fpad && iz < ppad && !fsurf) { // top CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - fpad; // the memory array index

                    mem_szz_z[pz * 2 * (npml + 1) + ix] = b[pz] * mem_szz_z[pz * 2 * (npml + 1) + ix] + a[pz] * szz_z;
                    mem_szx_z[pz * 2 * (npml + 1) + ix] = b_half[pz] * mem_szx_z[pz * 2 * (npml + 1) + ix] + a_half[pz] * szx_z;

                    szz_z = szz_z / K[pz] + mem_szz_z[pz * 2 * (npml + 1) + ix];
                    szx_z = szx_z / K_half[pz] + mem_szx_z[pz * 2 * (npml + 1) + ix];

                } // cpml top

                if (iz >= (nzt - ppad - 1) && iz < nzt - fpad) { // bottom CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - pnz; // The PML factors index

                    mem_szz_z[pz * 2 * (npml + 1) + ix] = b[pz] * mem_szz_z[pz * 2 * (npml + 1) + ix] + a[pz] * szz_z;
                    mem_szx_z[pz * 2 * (npml + 1) + ix] = b_half[pz] * mem_szx_z[pz * 2 * (npml + 1) + ix] + a_half[pz] * szx_z;

                    szz_z = szz_z / K[pz] + mem_szz_z[pz * 2 * (npml + 1) + ix];
                    szx_z = szx_z / K_half[pz] + mem_szx_z[pz * 2 * (npml + 1) + ix];

                } // cpml bottom
                __syncthreads();
            } // npml>0

            // update particle velocities
            vx[iz * nzt + ix] += dt * rho_xp[iz * (nzt - 1) + ix] * (sxx_x + szx_z);
            vz[iz * nzt + ix] += dt * rho_zp[iz * (nzt - 1) + ix] * (szx_x + szz_z);


        }
        else { return; }

        //  break;

    default:
        //  std::cout << "FDORDER = " << fdorder << npml << std::endl;
        //  std::cout << "FD order mismatch. <SIMULATION ABORTED>" << std::endl;
          //exit(0);
       // printf("FD order mismatch. <SIMULATION ABORTED>\n");
    } // end of switch


}

__global__ void kernel_IV(int nx1, int nx2, int fpad, int nzt, real_sim* szx, real_sim* szz) {
    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    int sz = 1;
    if (ix >= nx1 && ix < nx2 && sz >= 1 && sz <= fpad) {
        // mirroring 
        szx[(fpad - sz) * nzt + ix] = -szx[(fpad + sz) * nzt + ix];
        szz[(fpad - sz) * nzt + ix] = -szz[(fpad + sz) * nzt + ix];

    }

}




void adjoint_kernel_PSV(int ishot, // shot index
                        // Time and space grid arguments
    int nt, int nzt, int nxt, int fpad, int ppad,
    real_sim dt, real_sim dx, real_sim dz, int snap_interval, bool fsurf,
    // computationsl arguments
    real_sim* hc, int fdorder,
    // Wave arguments
    real_sim** vx, real_sim** vz,
    real_sim** sxx, real_sim** szx, real_sim** szz,
    // Medium arguments
    real_sim** lam, real_sim** mu,
    real_sim** mu_zx, real_sim** rho_zp, real_sim** rho_xp,
    //PML arguments
    int npml, real_sim* a, real_sim* b, real_sim* K,
    real_sim* a_half, real_sim* b_half, real_sim* K_half,
    // PML memory arrays
    real_sim** mem_vx_x, real_sim** mem_vx_z, real_sim** mem_vz_x, real_sim** mem_vz_z,
    real_sim** mem_sxx_x, real_sim** mem_szx_x, real_sim** mem_szz_z, real_sim** mem_szx_z,
    // Source arguments
    int nsrc, ivec src_x, ivec src_z, ivec src_comp,
    real_sim** src_signal, ivec source_to_fire_shot,
    // FWI arguments
    bool fwinv, int fwi_dt, int fwi_dx, int fwi_dz,
    int fwi_x1, int fwi_x2, int fwi_z1, int fwi_z2,
    real_sim*** fwi_vx, real_sim*** fwi_vz, real_sim*** fwi_sxx,
    real_sim*** fwi_szx, real_sim*** fwi_szz,
    // Gradient of the materials
    real_sim** grad_lam, real_sim** grad_mu, real_sim** grad_rho) {

    //const bool fwi = 1;
    // int nt = number of timesteps
    // int nz1, nz2, nx1, nx2 = start and end grids along z and x directions
    // int dt, dx, dz = grid spacing in time and space
    // int* hc = holberg coefficients
    // real_sim **&vx, **&vz, **&sxx, **&szx, **&szz, // wave parameters (particle velocity and stresses)
    // real_sim **&lam, **&mu, **&mu_zx, **&rho_zp, **&rho_xp // medium parameters (lamé's parameters')

    // real_sim *a, *b, *K;// CPML parameters
    // real_sim *a_half, *b_half, *K_half // CPML interpolated parameters
    // real_sim ** mem_vx_x, ** mem_vx_z, ** mem_vz_x, ** mem_vz_z; // PML velocity derivative memory
    // real_sim **&mem_sxx_x, **&mem_szx_x, **&mem_szz_z, real_sim **&mem_szx_z // PML stress derivative memory
    // bool fsurf :: free surface on the top


    // Source arguments
    // int nsrc = number of sources
    // int **src_loc = grid location of source + source parameter type for eg exploxive, vz only etc
    // real_sim ** src_signal = signal values for the sources

    real_sim sxx_x, szx_x, szx_z, szz_z; // spatial stress derivatives
    real_sim vx_x, vx_z, vz_x, vz_z; // spatial velocity derivatives
    int nz1, nz2, nx1, nx2; // The computational grid boundaries
    int px, pz; // index for PML arrys
    int isnap; // to take snapshots for data storage

    int tf, zf, xf; // Index parameters for fwi data storage
    real_sim s1, s2, s3, s4; // Intermediate variables for gradient calculation

    std::ofstream outFile; // file to print vz arrays


    // Initial calculation of indices
    //---------------------------------------------
    nz1 = fpad; nz2 = nzt - fpad; nx1 = fpad; nx2 = nxt - fpad; // index variables

    // index to map PML at the right or positive end
    int pnx, pnz;
    pnx = nxt - 2 * ppad + fpad - 1; // nx + ppad + npml + 1 and nx = nxt - 2*ppad

    if (fsurf) {
        pnz = nzt - 2 * ppad - 1; // nz + ppad + npml + 1 and nz = nzt - ppad - fpad
    }
    else {
        pnz = nzt - 2 * ppad + fpad - 1; // nz + ppad + npml + 1 and nz = nzt - 2*ppad
    }

    real_sim dxi = 1.0 / dx; real_sim dzi = 1.0 / dz; // inverse of dx and dz

    // -----------------------------------------------------------


    // Reset kernels
    // -----------------------------------------------------
    // Stress and velocity kernels
    for (int iz = 0; iz < nzt; iz++) {
        for (int ix = 0; ix < nxt; ix++) {
            // Wave velocity and stress tensor arrays
            vx[iz][ix] = 0.0;
            vz[iz][ix] = 0.0;
            sxx[iz][ix] = 0.0;
            szx[iz][ix] = 0.0;
            szz[iz][ix] = 0.0;
        }
    }

    // Gradient kernels
    //-----------------------------

    const int nft = 1 + (nt - 1) / fwi_dt;
    const int nfz = 1 + (fwi_z2 - fwi_z1) / fwi_dz;
    const int nfx = 1 + (fwi_x2 - fwi_x1) / fwi_dx;

    for (int iz = 0; iz < nfz; iz++) {
        for (int ix = 0; ix < nfx; ix++) {

            // Gradients of the material
            grad_lam[iz][ix] = 0.0;
            grad_mu[iz][ix] = 0.0;
            grad_rho[iz][ix] = 0.0;

        }
    }
    //*******************************************GPU***********************


    real_sim* d_a; real_sim* d_b; real_sim* d_K; real_sim* d_a_half; real_sim* d_b_half; real_sim* d_K_half;

    int size_1d = 2 * (npml + 1);

    gpuErrchk(cudaMalloc((void**)&d_a, size_1d * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_b, size_1d * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_K, size_1d * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_a_half, size_1d * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_b_half, size_1d * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_K_half, size_1d * sizeof(real_sim)));


    real_sim* d_vx; real_sim* d_vz; real_sim* d_sxx;
    real_sim* d_szx; real_sim* d_szz; real_sim* d_lam;  real_sim* d_mu;
    real_sim* d_mu_zx; real_sim* d_rho_zp; real_sim* d_rho_xp;
    real_sim* d_mem_vx_x; real_sim* d_mem_vx_z; real_sim* d_mem_vz_x; real_sim* d_mem_vz_z;
    real_sim* d_mem_sxx_x; real_sim* d_mem_szx_x; real_sim* d_mem_szz_z; real_sim* d_mem_szx_z;

    real_sim size = nzt * nxt;

    gpuErrchk(cudaMalloc((void**)&d_vx, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_vz, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_sxx, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_szx, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_szz, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_lam, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_mu, size * sizeof(real_sim)));
    size = (nzt - 1) * (nxt - 1);
    gpuErrchk(cudaMalloc((void**)&d_mu_zx, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_rho_zp, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_rho_xp, size * sizeof(real_sim)));
    size = nzt * 2 * (npml + 1);
    gpuErrchk(cudaMalloc((void**)&d_mem_vx_x, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_mem_vx_z, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_mem_vz_x, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_mem_vz_z, size * sizeof(real_sim)));

    gpuErrchk(cudaMalloc((void**)&d_mem_sxx_x, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_mem_szx_x, size * sizeof(real_sim)));
    size = nxt * 2 * (npml + 1);
    gpuErrchk(cudaMalloc((void**)&d_mem_szz_z, size * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_mem_szx_z, size * sizeof(real_sim)));

    size = nzt * nxt;
    

     //*****************
          // updating velocity tensors

    gpuErrchk(cudaMemcpy(d_mem_vx_x, mem_vx_x[0], nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mem_vx_z, mem_vx_z[0], nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mem_vz_x, mem_vz_x[0], nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mem_vz_z, mem_vz_z[0], nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));

     gpuErrchk(cudaMemcpy(d_sxx, sxx[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));
     gpuErrchk(cudaMemcpy(d_szx, szx[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));
     gpuErrchk(cudaMemcpy(d_szz, szz[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lam, lam[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mu, mu[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_mu_zx, mu_zx[0], (nzt - 1) * (nxt - 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rho_zp, rho_zp[0], (nzt - 1) * (nxt - 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_rho_xp, rho_xp[0], (nzt - 1) * (nxt - 1) * sizeof(real_sim), cudaMemcpyHostToDevice));


    gpuErrchk(cudaMemcpy(d_mem_sxx_x, mem_sxx_x[0], nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mem_szx_x, mem_szx_x[0], nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_mem_szz_z, mem_szz_z[0], nxt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mem_szx_z, mem_szx_z[0], nxt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_a, a, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_K, K, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a_half, a_half, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b_half, b_half, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_K_half, K_half, size_1d * sizeof(real_sim), cudaMemcpyHostToDevice));


    gpuErrchk(cudaMemcpy(d_vx, vx[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));


    real_sim* d_fwi_vx;
    real_sim* d_fwi_vz;
    real_sim* d_fwi_sxx;
    real_sim* d_fwi_szx;
    real_sim* d_fwi_szz;

    real_sim* d_grad_lam;
    real_sim* d_grad_mu;
    real_sim* d_grad_rho;

    int size_grad = nfz * nfx;
    gpuErrchk(cudaMalloc((void**)&d_grad_lam, size_grad * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_grad_mu, size_grad * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_grad_rho, size_grad * sizeof(real_sim)));

    gpuErrchk(cudaMemcpy(d_grad_lam, grad_lam[0], size_grad * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_grad_mu, grad_mu[0], size_grad * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_grad_rho, grad_rho[0], size_grad * sizeof(real_sim), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void**)&d_fwi_vx, (nft * nfz * nfx) * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_fwi_vz, (nft * nfz * nfx) * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_fwi_sxx, (nft * nfz * nfx) * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_fwi_szx, (nft * nfz * nfx) * sizeof(real_sim)));
    gpuErrchk(cudaMalloc((void**)&d_fwi_szz, (nft * nfz * nfx) * sizeof(real_sim)));

    gpuErrchk(cudaMemcpy(d_fwi_vx, fwi_vx[0][0], (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fwi_vz, fwi_vz[0][0], (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fwi_sxx, fwi_sxx[0][0], (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fwi_szx, fwi_szx[0][0], (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fwi_szz, fwi_szz[0][0], (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    int box1 = 16, box2 = 16;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz2 + box1 - 1) / box1, (nx2 + box2 - 1) / box2);




    //************************************************************

    // Start of time loop
    isnap = 0;
    for (int it = nt - 1; it >= 0; it--) {

        // ---------------------------------------------------------
        // Computation of gradient kernels
        gpuErrchk(cudaMemcpy(d_vz, vz[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));

        if (fwinv && !(it % fwi_dt)) {


            tf = it / fwi_dt; // t index for fwi gradient storage
            //std::cout<<"fwi time: " << it << ", adjoint simulation" << std::endl;
            kernel_I << < blocksPerGrid, threadsPerBlock >> > (tf, fwi_dt, dt, nzt, fwi_z1, fwi_z2, fwi_x1, fwi_x2,
                fwi_dz, fwi_dx, nft, nfz, nfx,
                d_fwi_sxx, d_fwi_szx, d_fwi_szz, d_fwi_vx,
                d_fwi_vz, d_sxx, d_szx, d_szz, d_vx, d_vz, d_mu,
                d_lam, d_grad_lam, d_grad_mu, d_grad_rho);

        }

        //****************************************************Kernrl calls GPU*************************************




        // Calculate spatial velocity derivatives
        kernel_II << < blocksPerGrid, threadsPerBlock >> > (ishot, nt, nzt, nxt, fpad, ppad, dt, dx, dz,
            fdorder, d_vx, d_vz, d_sxx,
            d_szx, d_szz, d_lam, d_mu,
            d_mu_zx, d_rho_zp, d_rho_xp, npml,
            d_a, d_b, d_K, d_a_half, d_b_half, d_K_half,
            d_mem_vx_x, d_mem_vx_z, d_mem_vz_x, d_mem_vz_z,
            d_mem_sxx_x, d_mem_szx_x, d_mem_szz_z, d_mem_szx_z, fsurf);
       
        gpuErrchk(cudaPeekAtLastError());

        // compute spatial stress derivatives
        kernel_III << < blocksPerGrid, threadsPerBlock >> > (ishot, nt, nzt, nxt, fpad, ppad, dt, dx, dz,
            fdorder, d_vx, d_vz, d_sxx,
            d_szx, d_szz, d_lam, d_mu,
            d_mu_zx, d_rho_zp, d_rho_xp, npml,
            d_a, d_b, d_K, d_a_half, d_b_half, d_K_half,
            d_mem_vx_x, d_mem_vx_z, d_mem_vz_x, d_mem_vz_z,
            d_mem_sxx_x, d_mem_szx_x, d_mem_szz_z, d_mem_szx_z, fsurf);
        gpuErrchk(cudaPeekAtLastError());

      


        if (fsurf) { // Mirroring stresses for free surface condition

            kernel_IV << < blocksPerGrid, threadsPerBlock >> > (nx1, nx2, fpad, nzt, d_szx, d_szz);
            gpuErrchk(cudaPeekAtLastError());

        }
        gpuErrchk(cudaDeviceSynchronize());
       

        gpuErrchk(cudaMemcpy(vz[0], d_vz, size * sizeof(real_sim), cudaMemcpyDeviceToHost));

        //****************************************************************************

                // Adding Velocity update related sources
                //----------------------------------------
        for (int is = 0; is <= 0 /*nsrc*/; is++) {

            if (source_to_fire_shot[is] == ishot) {
                switch (src_comp[is]) {// defines the signal type
                case(2): // vz component only
                    vz[src_z[is]][src_x[is]] += src_signal[is][it];
                }
            }
        }

        // ---------------------------------------

        // Printing out AASCII data for snap intervals
        if (!(it % snap_interval || it == 0)) {
            std::cout << "Time step " << it << " of " << nt << " in adjoint kernel." << std::endl;
            isnap++;
        }

    } // end of time loop

    gpuErrchk(cudaMemcpy(grad_lam[0], d_grad_lam, size_grad * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(grad_mu[0], d_grad_mu, size_grad * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(grad_rho[0], d_grad_rho, size_grad * sizeof(real_sim), cudaMemcpyDeviceToHost));

    //********************************************************************Memcpy back to host
    gpuErrchk(cudaMemcpy(fwi_vx[0][0], d_fwi_vx, (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(fwi_vz[0][0], d_fwi_vz, (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(fwi_sxx[0][0], d_fwi_sxx, (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(fwi_szx[0][0], d_fwi_szx, (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(fwi_szz[0][0], d_fwi_szz, (nft * nfz * nfx) * sizeof(real_sim), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(vx[0], d_vx, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(sxx[0], d_sxx, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(szx[0], d_szx, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(szz[0], d_szz, size * sizeof(real_sim), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(lam[0], d_lam, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mu[0], d_mu, size * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mu_zx[0], d_mu_zx, (nzt - 1) * (nxt - 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(rho_zp[0], d_rho_zp, (nzt - 1) * (nxt - 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(rho_xp[0], d_rho_xp, (nzt - 1) * (nxt - 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(mem_vx_x[0], d_mem_vx_x, nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mem_vx_z[0], d_mem_vx_z, nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mem_vz_x[0], d_mem_vz_x, nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mem_vz_z[0], d_mem_vz_z, nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));


    gpuErrchk(cudaMemcpy(mem_sxx_x[0], d_mem_sxx_x, nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mem_szx_x[0], d_mem_szx_x, nzt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mem_szz_z[0], d_mem_szz_z, nxt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(mem_szx_z[0], d_mem_szx_z, nxt * 2 * (npml + 1) * sizeof(real_sim), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());


}
