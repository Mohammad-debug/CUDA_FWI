
/*
* Created by: Min Basnet
* 2020.April.16
* Kathmandu, Nepal
*/
#include <time.h>
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



__global__ void kernel_zero(int tf, int nt, int fwi_z1, int fwi_z2, int fwi_x1, int fwi_x2, int fwi_dz, int fwi_dx, int nft, int nzt,int nxt, int nfx, real_sim* fwi_sxx, real_sim* fwi_szx, real_sim* fwi_szz, real_sim* fwi_vx, real_sim*
    fwi_vz, real_sim* sxx, real_sim* szx, real_sim* szz, real_sim* vx, real_sim* vz) {


    //(int tf, int nt, int fwi_z1, int fwi_z2, int fwi_x1, int fwi_x2, int fwi_dz, int fwi_dx, int nft, int nzt, int nxt,nfx real_sim* fwi_sxx, real_sim* fwi_szx, real_sim* fwi_szz, real_sim* fwi_vx, real_sim*
   // fwi_vz, real_sim* sxx, real_sim* szx, real_sim* szz, real_sim* vx, real_sim* vz)


    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (iz >= fwi_z1 && iz < fwi_z2 && ((iz % fwi_dz) == fwi_z1 % fwi_dz) && ix >= fwi_x1 && ix < fwi_x2 && ((ix % fwi_dx) == fwi_x1 % fwi_dx)) { // storing only a portion and with grid inteval
        int zf = (iz - fwi_z1) / fwi_dz; // z index for fwi gradient storage
    
                                         // printf("Hello Executed below pz \n");
        
            int xf = (ix - fwi_x1) / fwi_dx; // x index for fwi gradient storage
            
            int offset = tf * nzt * nxt + zf * nxt + xf;
            fwi_sxx[offset] = sxx[iz * nxt + ix];

            fwi_szx[offset] = szx[iz * nxt + ix];
            fwi_szz[offset] = szz[iz * nxt + ix];

            fwi_vx[offset] = vx[iz * nxt + ix];
            fwi_vz[offset] = vz[iz * nxt + ix];

        
    }
    else {
        return;
    }
}

__global__ void kernel_one(int ishot, int nt, int nzt, int nxt, int fpad, int ppad, real_sim dt, real_sim dx, real_sim dz,
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


    int iz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;


    switch (fdorder) {
    case(2):
        if (ix < nx2 && ix >= nx1 && iz >= nz1 && iz < nz2)

        {

            // Calculate spatial velocity derivatives
            vx_x = dxi * hc[1] * (vx[iz * nxt + ix] - vx[iz * nxt + (ix - 1)]);
            vz_x = dxi * hc[1] * (vz[iz * nxt + (ix + 1)] - vz[iz * nxt + ix]);
            vx_z = dzi * hc[1] * (vx[(iz + 1) * nxt + ix] - vx[iz * nxt + ix]);
            vz_z = dzi * hc[1] * (vz[iz * nxt + ix] - vz[(iz - 1) * nxt + ix]);
            //************************************************************************************************
    // ---------------------------------------------------
    // CPML layers for stress tensor kernel
    // ---------------------------------------------------




        // ---------------------------------------------------
        // CPML layers for stress tensor kernel
         //---------------------------------------------------

            if (npml > 0) {

                if (ix >= fpad && ix <= ppad)
                { // left CPML
                  // Mapping the static CPML and memory variables to
                    px = ix - fpad; // the memory array index


                    mem_vx_x[iz * 2 * (npml + 1) + px] = b[px] * mem_vx_x[iz * 2 * (npml + 1) + px] + a[px] * vx_x;
                    mem_vz_x[iz * 2 * (npml + 1) + px] = b_half[px] * mem_vz_x[iz * 2 * (npml + 1) + px] + a_half[px] * vz_x;

                    vx_x = vx_x / K[px] + mem_vx_x[iz * 2 * (npml + 1) + px];
                    vz_x = vz_x / K_half[px] + mem_vz_x[iz * 2 * (npml + 1) + px];

                } // cpml left



                if (ix >= (nxt - ppad - 1) && ix < nxt - fpad) { // right CPML
                  // Mapping the static CPML and memory variables to

                    px = ix - pnx; // The PML factors index

           //         
                    mem_vx_x[iz * 2 * (npml + 1) + px] = b[px] * mem_vx_x[iz * 2 * (npml + 1) + px] + a[px] * vx_x;
                    mem_vz_x[iz * 2 * (npml + 1) + px] = b_half[px] * mem_vz_x[iz * 2 * (npml + 1) + px] + a_half[px] * vz_x;

                    vx_x = vx_x / K[px] + mem_vx_x[iz * 2 * (npml + 1) + px];
                    vz_x = vz_x / K_half[px] + mem_vz_x[iz * 2 * (npml + 1) + px];



                } // cpml right


                if (iz >= fpad && iz <= ppad && !fsurf) { // top CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - fpad; // the memory array index


                    mem_vz_z[pz * nxt + ix] = b[pz] * mem_vz_z[pz * nxt + ix] + a[pz] * vz_z;
                    mem_vx_z[pz * nxt + ix] = b_half[pz] * mem_vx_z[pz * nxt + ix] + a_half[pz] * vx_z;

                    vz_z = vz_z / K[pz] + mem_vz_z[pz * nxt + ix];
                    vx_z = vx_z / K_half[pz] + mem_vx_z[pz * nxt + ix];

                    //std::cout << pz<< ", ";

                } // cpml top



                if (iz >= (nzt - ppad - 1) && iz < nzt - fpad) { // bottom CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - pnz; // The PML factors index
                    mem_vz_z[pz * nxt + ix] = b[pz] * mem_vz_z[pz * nxt + ix] + a[pz] * vz_z;
                    mem_vx_z[pz * nxt + ix] = b_half[pz] * mem_vx_z[pz * nxt + ix] + a_half[pz] * vx_z;

                    vz_z = vz_z / K[pz] + mem_vz_z[pz * nxt + ix];
                    vx_z = vx_z / K_half[pz] + mem_vx_z[pz * nxt + ix];
                    //std::cout << pz<< ", ";

                } // cpml bottom

            }    // npml>0
            __syncthreads();

            ////// updating stresses

            szx[iz * nxt + ix] += dt * mu_zx[iz * (nxt - 1) + ix] * (vz_x + vx_z);
            sxx[iz * nxt + ix] += dt * (lam[iz * nxt + ix] * (vx_x + vz_z) + (2.0f * mu[iz * nxt + ix] * vx_x));
            szz[iz * nxt + ix] += dt * (lam[iz * nxt + ix] * (vx_x + vz_z) + (2.0f * mu[iz * nxt + ix] * vz_z));

            // Override stress for free surface implementation

            if (fsurf && iz == fpad) {
                // Free surface at z = 0 or nz = fpad

                // Denise manual  page 13
                szz[fpad * nxt + ix] = 0.0;
                szx[fpad * nxt + ix] = 0.0;
                sxx[fpad * nxt + ix] = 4.0 * dt * vx_x * (lam[fpad * nxt + ix] * mu[fpad * nxt + ix] + mu[fpad * nxt + ix] * mu[fpad * nxt + ix])
                    / (lam[fpad * nxt + ix] + 2.0 * mu[fpad * nxt + ix]);



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

__global__ void kernel_two(int ishot, int nt, int nzt, int nxt, int fpad, int ppad, real_sim dt, real_sim dx, real_sim dz,
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
            sxx_x = dxi * hc[1] * (sxx[iz * nxt + ix + 1] - sxx[iz * nxt + ix]);
            szx_z = dxi * hc[1] * (szx[iz * nxt + ix] - szx[(iz - 1) * nxt + ix]);

            szx_x = dzi * hc[1] * (szx[iz * nxt + ix] - szx[iz * nxt + ix - 1]);
            szz_z = dzi * hc[1] * (szz[(iz + 1) * nxt + ix] - szz[iz * nxt + ix]);
            // ---------------------------------------------------
            // CPML layers for particle velocity kernel
            // ---------------------------------------------------

            if (npml > 0) {

                if (ix >= fpad && ix < ppad) { // left CPML
                  // Mapping the static CPML and memory variables to
                    px = ix - fpad; // the memory array index

                    mem_sxx_x[iz * 2 * (npml + 1) + px] = b[px] * mem_sxx_x[iz * 2 * (npml + 1) + px] + a[px] * sxx_x;
                    mem_szx_x[iz * 2 * (npml + 1) + px] = b_half[px] * mem_szx_x[iz * 2 * (npml + 1) + px] + a_half[px] * szx_x;

                    sxx_x = sxx_x / K[px] + mem_sxx_x[iz * 2 * (npml + 1) + px];
                    szx_x = szx_x / K_half[px] + mem_szx_x[iz * 2 * (npml + 1) + px];

                } // cpml left

                if (ix >= (nxt - ppad - 1) && ix < (nxt - fpad)) { // right CPML
                  // Mapping the static CPML and memory variables to
                    px = ix - pnx; // The PML factors index

                    mem_sxx_x[iz * 2 * (npml + 1) + px] = b[px] * mem_sxx_x[iz * 2 * (npml + 1) + px] + a[px] * sxx_x;
                    mem_szx_x[iz * 2 * (npml + 1) + px] = b_half[px] * mem_szx_x[iz * 2 * (npml + 1) + px] + a_half[px] * szx_x;

                    sxx_x = sxx_x / K[px] + mem_sxx_x[iz * 2 * (npml + 1) + px];
                    szx_x = szx_x / K_half[px] + mem_szx_x[iz * 2 * (npml + 1) + px];

                } // cpml right


                if (iz >= fpad && iz < ppad && !fsurf) { // top CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - fpad; // the memory array index

                    mem_szz_z[pz * nxt + ix] = b[pz] * mem_szz_z[pz * nxt + ix] + a[pz] * szz_z;
                    mem_szx_z[pz * nxt + ix] = b_half[pz] * mem_szx_z[pz * nxt + ix] + a_half[pz] * szx_z;

                    szz_z = szz_z / K[pz] + mem_szz_z[pz * nxt + ix];
                    szx_z = szx_z / K_half[pz] + mem_szx_z[pz * nxt + ix];

                } // cpml top

                if (iz >= (nzt - ppad - 1) && iz < nzt - fpad) { // bottom CPML
                  // Mapping the static CPML and memory variables to
                    pz = iz - pnz; // The PML factors index
                    mem_szz_z[pz * nxt + ix] = b[pz] * mem_szz_z[pz * nxt + ix] + a[pz] * szz_z;
                    mem_szx_z[pz * nxt + ix] = b_half[pz] * mem_szx_z[pz * nxt + ix] + a_half[pz] * szx_z;

                    szz_z = szz_z / K[pz] + mem_szz_z[pz * nxt + ix];
                    szx_z = szx_z / K_half[pz] + mem_szx_z[pz * nxt + ix];

                } // cpml bottom
                __syncthreads();
            } // npml>0

            // update particle velocities
            vx[iz * nxt + ix] += dt * rho_xp[iz * (nxt - 1) + ix] * (sxx_x + szx_z);
            vz[iz * nxt + ix] += dt * rho_zp[iz * (nxt - 1) + ix] * (szx_x + szz_z);


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



__global__ void kernel_Thri(int nx1, int nx2, int fpad, int nxt, real_sim* szx, real_sim* szz) {
    int sz = blockIdx.x * blockDim.x + threadIdx.x;
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= nx1 && ix < nx2 && sz >= 1 && sz <= fpad) {
       // printf("Hello Executed below thri \n");
        // mirroring 
        szx[(fpad - sz) * nxt + ix] = -szx[(fpad + sz) * nxt + ix];
        szz[(fpad - sz) * nxt + ix] = -szz[(fpad + sz) * nxt + ix];

    }

}









void forward_kernel_PSV_GPU(int ishot, // shot number
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
    // Receiver arguments
    int nrec, ivec rec_x, ivec rec_z, ivec rec_comp,
    real_sim** rec_signal, ivec receiver_to_record_shot,//shot_to_fire
    // FWI arguments
    bool fwinv, int fwi_dt, int fwi_dx, int fwi_dz,
    int fwi_x1, int fwi_x2, int fwi_z1, int fwi_z2,
    real_sim*** fwi_vx, real_sim*** fwi_vz, real_sim*** fwi_sxx,
    real_sim*** fwi_szx, real_sim*** fwi_szz,
    //*****************CPU PARAMS***************
    real_sim* d_a, real_sim* d_b, real_sim* d_K, real_sim* d_a_half, real_sim* d_b_half, real_sim* d_K_half,
    //
    real_sim* d_vx, real_sim* d_vz, real_sim* d_sxx, real_sim* d_szx, real_sim* d_szz,
    //
    real_sim* d_fwi_vx,
    real_sim* d_fwi_vz,
    real_sim* d_fwi_sxx,
    real_sim* d_fwi_szx,
    real_sim* d_fwi_szz,
    //
    real_sim* d_mem_vx_x, real_sim* d_mem_vx_z, real_sim* d_mem_vz_x, real_sim* d_mem_vz_z,
    real_sim* d_mem_sxx_x, real_sim* d_mem_szx_x, real_sim* d_mem_szz_z, real_sim* d_mem_szx_z,
    //
    real_sim* d_lam, real_sim* d_mu,
    real_sim* d_mu_zx, real_sim* d_rho_zp, real_sim* d_rho_xp

) {
    // std::cout << "Reached Here1" << "\n";
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

    real_sim size = nzt * nxt;
    gpuErrchk(cudaMemset(d_vz, 0, size * sizeof(real_sim)));
    gpuErrchk(cudaMemset(d_vx, 0, size * sizeof(real_sim)));

    gpuErrchk(cudaMemset(d_sxx, 0, size * sizeof(real_sim)));
    gpuErrchk(cudaMemset(d_szx, 0, size * sizeof(real_sim)));
    gpuErrchk(cudaMemset(d_szz, 0, size * sizeof(real_sim)));


    
    if (fwinv) {

        const int nft = 1 + (nt - 1) / fwi_dt;
        const int nfz = 1 + (fwi_z2 - fwi_z1) / fwi_dz;
        const int nfx = 1 + (fwi_x2 - fwi_x1) / fwi_dx;
        size = nft * nfz * nfx;
 
        gpuErrchk(cudaMemset(d_fwi_vx, 0, size * sizeof(real_sim)));
        gpuErrchk(cudaMemset(d_fwi_vz, 0, size * sizeof(real_sim)));
        gpuErrchk(cudaMemset(d_fwi_sxx, 0, size * sizeof(real_sim)));
        gpuErrchk(cudaMemset(d_fwi_szx, 0, size * sizeof(real_sim)));
        gpuErrchk(cudaMemset(d_fwi_szz, 0, size * sizeof(real_sim)));



    }

    // Start of time loop
    isnap = 0;
    //**********************************Memory Copy**************************************
    clock_t start = 0, end = 0;
    double start1 = 0;
    double cpu_time_used;

    size = nzt * nxt;

    int box1 = 16, box2 = 16;
    dim3 threadsPerBlock(box1, box2);
    dim3 blocksPerGrid((nz2-nz1 + box1 - 1) / box1, (nx2-nx1 + box2 - 1) / box2);

   box1 = 16, box2 = 16;
    dim3 threadsPerBlock0(box1, box2);
    dim3 blocksPerGrid0((fwi_z2- fwi_z1 + box1 - 1) / box1, (fwi_x2- fwi_x1 + box2 - 1) / box2);

 
    const int nft = 1 + (nt - 1) / fwi_dt;
    const int nfz = 1 + (fwi_z2 - fwi_z1) / fwi_dz;
    const int nfx = 1 + (fwi_x2 - fwi_x1) / fwi_dx;

    gpuErrchk(cudaPeekAtLastError());
    //**************************************************************************************************


    for (int it = 0; it < nt; it++) {
        start = clock();

        // ---------------------------------------------------------
        // Storing velocity  & stress tensors for gradient calculations for full waveform inversion
        if (fwinv && !(it % fwi_dt)) {
            tf = it / fwi_dt; // t index for fwi gradient storage
           // std::cout<<"fwi time: " << it << ", forward simulation  "<<fwi_dt << std::endl;

            kernel_zero << < blocksPerGrid0, threadsPerBlock >> > (tf, nt, fwi_z1, fwi_z2, fwi_x1, fwi_x2,
                fwi_dz, fwi_dx, nft, nzt, nxt,nfx, d_fwi_sxx, d_fwi_szx, d_fwi_szz, d_fwi_vx,
                d_fwi_vz, d_sxx, d_szx, d_szz, d_vx, d_vz);
            gpuErrchk(cudaPeekAtLastError());
        }


        //(int tf, int nt, int fwi_z1, int fwi_z2, int fwi_x1, int fwi_x2, int fwi_dz, int fwi_dx, int nft, int nzt, int nxt,nfx real_sim* fwi_sxx, real_sim* fwi_szx, real_sim* fwi_szz, real_sim* fwi_vx, real_sim*
       // fwi_vz, real_sim* sxx, real_sim* szx, real_sim* szz, real_sim* vx, real_sim* vz)
        gpuErrchk(cudaDeviceSynchronize());
        //******************************************GPU****************************************


       gpuErrchk(cudaMemcpy(d_vz, vz[0], size * sizeof(real_sim), cudaMemcpyHostToDevice));

        // Calculate spatial velocity derivatives
        kernel_one << < blocksPerGrid, threadsPerBlock >> > (ishot, nt, nzt, nxt, fpad, ppad, dt, dx, dz,
            fdorder, d_vx, d_vz, d_sxx,
            d_szx, d_szz, d_lam, d_mu,
            d_mu_zx, d_rho_zp, d_rho_xp, npml,
            d_a, d_b, d_K, d_a_half, d_b_half, d_K_half,
            d_mem_vx_x, d_mem_vx_z, d_mem_vz_x, d_mem_vz_z,
            d_mem_sxx_x, d_mem_szx_x, d_mem_szz_z, d_mem_szx_z, fsurf);
      //  gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        // compute spatial stress derivatives
        kernel_two << < blocksPerGrid, threadsPerBlock >> > (ishot, nt, nzt, nxt, fpad, ppad, dt, dx, dz,
            fdorder, d_vx, d_vz, d_sxx,
            d_szx, d_szz, d_lam, d_mu,
            d_mu_zx, d_rho_zp, d_rho_xp, npml,
            d_a, d_b, d_K, d_a_half, d_b_half, d_K_half,
            d_mem_vx_x, d_mem_vx_z, d_mem_vz_x, d_mem_vz_z,
            d_mem_sxx_x, d_mem_szx_x, d_mem_szz_z, d_mem_szx_z, fsurf);
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());

        gpuErrchk(cudaMemcpy(vz[0], d_vz, size * sizeof(real_sim), cudaMemcpyDeviceToHost));

        if (fsurf) { // Mirroring stresses for free surface condition

            kernel_Thri << < blocksPerGrid, threadsPerBlock >> > (nx1, nx2, fpad, nxt, d_szx, d_szz);

        }

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaPeekAtLastError());
        //****************************************************************************************************************




                // Adding Velocity update related sources

                //--------------------------
        for (int is = 0; is < nsrc; is++) {

            if (source_to_fire_shot[is] == ishot) {
                switch (src_comp[is]) {// defines the signal type
                case(2): // vz component only
                    vz[src_z[is]][src_x[is]] += src_signal[is][it];
                }
            }
        }

 

        // Recording the signals to the receivers
        for (int ir = 0; ir <= 0 /*nrec*/; ir++) {

            if (receiver_to_record_shot[ir] == ishot) {

                switch (rec_comp[ir]) {// defines the signal type
                case(2): // vz component only
                    rec_signal[ir][it] = vz[rec_z[ir]][rec_x[ir]];
                }

            }
        }


        // Printing out AASCII data for snap intervals
      /*  if (!(it % snap_interval || it == 0)) {
            std::cout << "Time step " << it << " of " << nt << " in forward kernel." << std::endl;
              outFile.open("./io/snap_data/vz_snap" + std::to_string(isnap) + ".csv");

              for (int j = 0; j < nzt; j++) {
                  for (int i = 0; i < nxt; i++) {
                      outFile << vz[j][i] << ", ";
                  }
                  outFile << std::endl;
              }
              outFile.close();
            isnap++;
        }*/
  
    } // end of time loop

    


}



// ****************               CPU    ****************************
//forward_kernel_PSV.cpp


void forward_kernel_PSV(int ishot, // shot number
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
    // Receiver arguments
    int nrec, ivec rec_x, ivec rec_z, ivec rec_comp,
    real_sim** rec_signal, ivec receiver_to_record_shot,//shot_to_fire
    // FWI arguments
    bool fwinv, int fwi_dt, int fwi_dx, int fwi_dz,
    int fwi_x1, int fwi_x2, int fwi_z1, int fwi_z2,
    real_sim*** fwi_vx, real_sim*** fwi_vz, real_sim*** fwi_sxx,
    real_sim*** fwi_szx, real_sim*** fwi_szz) {
    


     // Source arguments
     // int nsrc = number of sources
     // int **src_loc = grid location of source + source parameter type for eg exploxive, vz only etc
     // real_sim ** src_signal = signal values for the sources
    double start1 = clock();
    real_sim sxx_x, szx_x, szx_z, szz_z; // spatial stress derivatives
    real_sim vx_x, vx_z, vz_x, vz_z; // spatial velocity derivatives
    int nz1, nz2, nx1, nx2; // The computational grid boundaries
    int px, pz; // index for PML arrys
    int isnap; // to take snapshots for data storage

    int tf, zf, xf; // Index parameters for fwi data storage

    std::ofstream outFile; // file to print vz arrays
    clock_t start = 0, end = 0;
    double cpu_time_used;

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
   //****************************************************************************************


    //**************************************************************************************************


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
    // std::cout << "Reached Here1.5" << "\n";
    if (fwinv) {
        const int nft = 1 + (nt - 1) / fwi_dt;
        const int nfz = 1 + (fwi_z2 - fwi_z1) / fwi_dz;
        const int nfx = 1 + (fwi_x2 - fwi_x1) / fwi_dx;

        for (int it = 0; it < nft; it++) {
            for (int iz = 0; iz < nfz; iz++) {
                for (int ix = 0; ix < nfx; ix++) {

                    // FWI storage arrays
                    fwi_vx[it][iz][ix] = 0.0;
                    fwi_vz[it][iz][ix] = 0.0;
                    fwi_sxx[it][iz][ix] = 0.0;
                    fwi_szx[it][iz][ix] = 0.0;
                    fwi_szz[it][iz][ix] = 0.0;

                }
            }
        }

    }

    // Start of time loop
    isnap = 0;

    for (int it = 0; it < nt; it++) {

        // ---------------------------------------------------------
        // Storing velocity  & stress tensors for gradient calculations for full waveform inversion
        if (fwinv && !(it % fwi_dt)) {
            tf = it / fwi_dt; // t index for fwi gradient storage

           std::cout<<"fwi time: " << it << ", forward simulation" << std::endl;
            for (int iz = fwi_z1; iz < fwi_z2; iz += fwi_dz) { // storing only a portion and with grid inteval
                zf = (iz - fwi_z1) / fwi_dz; // z index for fwi gradient storage

                for (int ix = fwi_x1; ix < fwi_x2; ix += fwi_dx) {
                    xf = (ix - fwi_x1) / fwi_dx; // x index for fwi gradient storage

                    fwi_sxx[tf][zf][xf] = sxx[iz][ix];
                    fwi_szx[tf][zf][xf] = szx[iz][ix];
                    fwi_szz[tf][zf][xf] = szz[iz][ix];
                    fwi_vx[tf][zf][xf] = vx[iz][ix];
                    fwi_vz[tf][zf][xf] = vz[iz][ix];

                }
            }
        }

        // --------------------------------------------------------
      //  std::cout << "Reached Here2" << "\n";
        start = clock();
        // Time integration of dynamic stress fields
        switch (fdorder) {
        case(2):
            // updating stress tensors
            for (int iz = nz1; iz < nz2; iz++) {
                //std::cout << std::endl << "PML indices: " << std::endl;
                for (int ix = nx1; ix < nx2; ix++) {
                    //std::cout << ix << ", " << iz << std::endl;

                    // Calculate spatial velocity derivatives
                    vx_x = dxi * hc[1] * (vx[iz][ix] - vx[iz][ix - 1]);
                    vz_x = dxi * hc[1] * (vz[iz][ix + 1] - vz[iz][ix]);
                    vx_z = dzi * hc[1] * (vx[iz + 1][ix] - vx[iz][ix]);
                    vz_z = dzi * hc[1] * (vz[iz][ix] - vz[iz - 1][ix]);

                    // ---------------------------------------------------
                    // CPML layers for stress tensor kernel
                    // ---------------------------------------------------

                    if (npml > 0) {

                        if (ix >= fpad && ix <= ppad) { // left CPML
                          // Mapping the static CPML and memory variables to
                            px = ix - fpad; // the memory array index
                            //std::cout << std::endl << "Fault1 " << ix << std::endl;

                            mem_vx_x[iz][px] = b[px] * mem_vx_x[iz][px] + a[px] * vx_x;
                            mem_vz_x[iz][px] = b_half[px] * mem_vz_x[iz][px] + a_half[px] * vz_x;

                            vx_x = vx_x / K[px] + mem_vx_x[iz][px];
                            vz_x = vz_x / K_half[px] + mem_vz_x[iz][px];

                        } // cpml left

                        if (ix >= (nxt - ppad - 1) && ix < nxt - fpad) { // right CPML
                          // Mapping the static CPML and memory variables to

                            px = ix - pnx; // The PML factors index

                            mem_vx_x[iz][px] = b[px] * mem_vx_x[iz][px] + a[px] * vx_x;
                            mem_vz_x[iz][px] = b_half[px] * mem_vz_x[iz][px] + a_half[px] * vz_x;

                            vx_x = vx_x / K[px] + mem_vx_x[iz][px];
                            vz_x = vz_x / K_half[px] + mem_vz_x[iz][px];


                        } // cpml right


                        if (iz >= fpad && iz <= ppad && !fsurf) { // top CPML
                          // Mapping the static CPML and memory variables to
                            pz = iz - fpad; // the memory array index


                            mem_vz_z[pz][ix] = b[pz] * mem_vz_z[pz][ix] + a[pz] * vz_z;
                            mem_vx_z[pz][ix] = b_half[pz] * mem_vx_z[pz][ix] + a_half[pz] * vx_z;

                            vz_z = vz_z / K[pz] + mem_vz_z[pz][ix];
                            vx_z = vx_z / K_half[pz] + mem_vx_z[pz][ix];

                            //std::cout << pz<< ", ";

                        } // cpml top

                        if (iz >= (nzt - ppad - 1) && iz < nzt - fpad) { // bottom CPML
                          // Mapping the static CPML and memory variables to
                            pz = iz - pnz; // The PML factors index

                            mem_vz_z[pz][ix] = b[pz] * mem_vz_z[pz][ix] + a[pz] * vz_z;
                            mem_vx_z[pz][ix] = b_half[pz] * mem_vx_z[pz][ix] + a_half[pz] * vx_z;

                            vz_z = vz_z / K[pz] + mem_vz_z[pz][ix];
                            vx_z = vx_z / K_half[pz] + mem_vx_z[pz][ix];

                            //std::cout << pz<< ", ";

                        } // cpml bottom

                    } // npml>0
                    // --------------------------------------------------------------------------
                    // --------------------------------------------------------------------------

                    // updating stresses
                    szx[iz][ix] += dt * mu_zx[iz][ix] * (vz_x + vx_z);
                    sxx[iz][ix] += dt * (lam[iz][ix] * (vx_x + vz_z) + (2.0 * mu[iz][ix] * vx_x));
                    szz[iz][ix] += dt * (lam[iz][ix] * (vx_x + vz_z) + (2.0 * mu[iz][ix] * vz_z));

                    // -----------------------------------------
                    // ----------------------------------------- 
                    // Override stress for free surface implementation

                    if (fsurf && iz == fpad) {
                        // Free surface at z = 0 or nz = fpad

                        // Denise manual  page 13
                        szz[fpad][ix] = 0.0;
                        szx[fpad][ix] = 0.0;
                        sxx[fpad][ix] = 4.0 * dt * vx_x * (lam[fpad][ix] * mu[fpad][ix] + mu[fpad][ix] * mu[fpad][ix])
                            / (lam[fpad][ix] + 2.0 * mu[fpad][ix]);

                        //}
                    }


                }
            }


            // STRESS MIRRORING TECHNIQUE FOR FREE SURFACE CONDITION
            if (fsurf) { // Mirroring stresses for free surface condition
                for (int ix = nx1; ix < nx2; ix++) {
                    for (int sz = 1; sz <= fpad; sz++) { // mirroring 
                        szx[fpad - sz][ix] = -szx[fpad + sz][ix];
                        szz[fpad - sz][ix] = -szz[fpad + sz][ix];
                    }
                }
            }

            // std::cout << "Reached Here3" << "\n";

             // updating velocity tensors
            for (int iz = nz1; iz < nz2; iz++) {
                for (int ix = nx1; ix < nx2; ix++) {

                    // compute spatial stress derivatives
                    sxx_x = dxi * hc[1] * (sxx[iz][ix + 1] - sxx[iz][ix]);
                    szx_x = dxi * hc[1] * (szx[iz][ix] - szx[iz][ix - 1]);
                    szx_z = dzi * hc[1] * (szx[iz][ix] - szx[iz - 1][ix]);
                    szz_z = dzi * hc[1] * (szz[iz + 1][ix] - szz[iz][ix]);


                    // ---------------------------------------------------
                    // CPML layers for particle velocity kernel
                    // ---------------------------------------------------

                    if (npml > 0) {

                        if (ix >= fpad && ix < ppad) { // left CPML
                          // Mapping the static CPML and memory variables to
                            px = ix - fpad; // the memory array index

                            mem_sxx_x[iz][px] = b[px] * mem_sxx_x[iz][px] + a[px] * sxx_x;
                            mem_szx_x[iz][px] = b_half[px] * mem_szx_x[iz][px] + a_half[px] * szx_x;

                            sxx_x = sxx_x / K[px] + mem_sxx_x[iz][px];
                            szx_x = szx_x / K_half[px] + mem_szx_x[iz][px];

                        } // cpml left

                        if (ix >= (nxt - ppad - 1) && ix < nxt - fpad) { // right CPML
                          // Mapping the static CPML and memory variables to
                            px = ix - pnx; // The PML factors index

                            mem_sxx_x[iz][px] = b[px] * mem_sxx_x[iz][px] + a[px] * sxx_x;
                            mem_szx_x[iz][px] = b_half[px] * mem_szx_x[iz][px] + a_half[px] * szx_x;

                            sxx_x = sxx_x / K[px] + mem_sxx_x[iz][px];
                            szx_x = szx_x / K_half[px] + mem_szx_x[iz][px];

                        } // cpml right

                        if (iz >= fpad && iz < ppad && !fsurf) { // top CPML
                          // Mapping the static CPML and memory variables to
                            pz = iz - fpad; // the memory array index

                            mem_szz_z[pz][ix] = b[pz] * mem_szz_z[pz][ix] + a[pz] * szz_z;
                            mem_szx_z[pz][ix] = b_half[pz] * mem_szx_z[pz][ix] + a_half[pz] * szx_z;

                            szz_z = szz_z / K[pz] + mem_szz_z[pz][ix];
                            szx_z = szx_z / K_half[pz] + mem_szx_z[pz][ix];

                        } // cpml top

                        if (iz >= (nzt - ppad - 1) && iz < nzt - fpad) { // bottom CPML
                          // Mapping the static CPML and memory variables to
                            pz = iz - pnz; // The PML factors index

                            mem_szz_z[pz][ix] = b[pz] * mem_szz_z[pz][ix] + a[pz] * szz_z;
                            mem_szx_z[pz][ix] = b_half[pz] * mem_szx_z[pz][ix] + a_half[pz] * szx_z;

                            szz_z = szz_z / K[pz] + mem_szz_z[pz][ix];
                            szx_z = szx_z / K_half[pz] + mem_szx_z[pz][ix];

                        } // cpml bottom

                    } // npml>0
                    // --------------------------------------------------------------------------
                    // --------------------------------------------------------------------------

                    // update particle velocities
                    vx[iz][ix] += dt * rho_xp[iz][ix] * (sxx_x + szx_z);
                    vz[iz][ix] += dt * rho_zp[iz][ix] * (szx_x + szz_z);

                }
            }

            break;

        default:
            std::cout << "FDORDER = " << fdorder << npml << std::endl;
            std::cout << "FD order mismatch. <SIMULATION ABORTED>" << std::endl;
            exit(0);
        } // end of switch


        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

        //  printf("\nExecution time Gpu = %f\n", cpu_time_used);


        start = clock();

        // Adding Velocity update related sources
     //   std::cout << "Reached Here4" << "\n";
        //--------------------------
        for (int is = 0; is < nsrc; is++) {

            if (source_to_fire_shot[is] == ishot) {
                switch (src_comp[is]) {// defines the signal type
                case(2): // vz component only
                    vz[src_z[is]][src_x[is]] += src_signal[is][it];
                }
            }
        }

        // ------------------------------------

        // Recording the signals to the receivers
        for (int ir = 0; ir <=0 /*nrec*/; ir++) {
           
            if (receiver_to_record_shot[ir] == ishot) {
               
                switch (rec_comp[ir]) {// defines the signal type
                case(2): // vz component only
                    rec_signal[ir][it] = vz[rec_z[ir]][rec_x[ir]];
                }

            }
        }


        // Printing out AASCII data for snap intervals
        if (!(it % snap_interval || it == 0)) {
            std::cout << "Time step " << it << " of " << nt << " in forward kernel." << std::endl;
            //outFile.open("./io/snap_data/vz_snap" + std::to_string(isnap) + ".csv");
          /*  for (int j = 0; j < nzt; j++) {
                for (int i = 0; i < nxt; i++) {
                    outFile << vz[j][i] << ", ";
                }
                outFile << std::endl;
            }
            outFile.close();*/
            isnap++;
        }
        /*end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;*/

        //  printf("\nExecution time after Gpu = %f\n", cpu_time_used);



          // std::cout << "Reached Here6" << "\n";
    } // end of time loop
    end = clock();
    cpu_time_used = ((double)(end - start1)) / CLOCKS_PER_SEC;

   // printf("\nKERNEL TIME = %f\n", cpu_time_used);


}