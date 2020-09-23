//fd_kernel.cuh

/*
* Created by: Min Basnet
* 2020.April.19
* Kathmandu, Nepal
*/

#include "globvar.cuh"


void forward_kernel_PSV(int ishot, // shot number
                        // Time and space grid arguments
    int nt, int nzt, int nxt, int fpad, int ppad,
    real_sim dt, real_sim dx, real_sim dz, int snap_interval, bool fsurf,
    // computationsl arguments
    real_sim* hc, int fdorder,
    // Wave arguments
    real_sim** vx, real_sim** vz,
    real_sim** sxx, real_sim** szx, real_sim** szz, real_sim** We,
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
    real_sim** rec_signal, ivec receiver_to_record_shot,
    // FWI arguments
    bool fwinv, int fwi_dt, int fwi_dx, int fwi_dz,
    int fwi_x1, int fwi_x2, int fwi_z1, int fwi_z2,
    real_sim*** fwi_vx, real_sim*** fwi_vz,
    real_sim*** fwi_sxx, real_sim*** fwi_szx, real_sim*** fwi_szz);


void adjoint_kernel_PSV(int ishot, // shot index
                        // Time and space grid arguments
    int nt, int nzt, int nxt, int fpad, int ppad,
    real_sim dt, real_sim dx, real_sim dz, int snap_interval, bool fsurf,
    // computationsl arguments
    real_sim* hc, int fdorder,
    // Wave arguments
    real_sim** vx, real_sim** vz,
    real_sim** sxx, real_sim** szx, real_sim** szz, real_sim** We,
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
    real_sim*** fwi_vx, real_sim*** fwi_vz,
    real_sim*** fwi_sxx, real_sim*** fwi_szx, real_sim*** fwi_szz,
    // Gradient of the materials
    real_sim** grad_lam, real_sim** grad_mu, real_sim** grad_rho);

real_sim calc_opt_step(real_sim* L2_test, real_sim* step_length);

//*********************************
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
    real_sim** rec_signal, ivec receiver_to_record_shot,
    // FWI arguments
    bool fwinv, int fwi_dt, int fwi_dx, int fwi_dz,
    int fwi_x1, int fwi_x2, int fwi_z1, int fwi_z2,
    real_sim*** fwi_vx, real_sim*** fwi_vz, real_sim*** fwi_sxx,
    real_sim*** fwi_szx, real_sim*** fwi_szz,
    //*****************GPU PARAMS***************
    real_sim* d_a, real_sim* d_b, real_sim* d_K, real_sim* d_a_half, real_sim* d_b_half, real_sim* d_K_half,
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
);


void adjoint_kernel_PSV_GPU(int ishot, // shot index
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
    real_sim** grad_lam, real_sim** grad_mu, real_sim** grad_rho,
    //*****************GPU PARAMS***************
    real_sim* d_a, real_sim* d_b, real_sim* d_K, real_sim* d_a_half, real_sim* d_b_half, real_sim* d_K_half,
    real_sim* d_vx, real_sim* d_vz, real_sim* d_sxx,  real_sim* d_szx, real_sim* d_szz,
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

real_sim* d_grad_lam,
real_sim* d_grad_mu,
real_sim* d_grad_rho,
//
real_sim* d_lam, real_sim* d_mu,
real_sim* d_mu_zx, real_sim* d_rho_zp, real_sim* d_rho_xp


);