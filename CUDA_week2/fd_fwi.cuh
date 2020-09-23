
/*
* Created by: Min Basnet
* 2020.May.04
* Kathmandu, Nepal
*/

#ifndef FD_FWI_H				
#define FD_FWI_H	

#include "globvar.cuh"
#include "fd_grid.cuh"

class FDFWI {
public:

	int dt, dx, dz; // grid intervals to store forward data
	int z1, z2, x1, x2; // Grid boundaries to store forward data

	// 2D wave parameters (PSV and SH wave)
	real_sim*** vx, *** vz; // Particle velocity
	real_sim*** sxx, *** szz; // Stress tensors (Translational)
	real_sim*** szx; // Stress tensors (Rotational)

	real_sim** grad_lam, ** grad_mu, ** grad_rho; // Gradients of the medium parameters

	void parse_configuration(const char* configuration_file_relative_path, FDGrid grid);

	void allocate_fwi(int dimt, int dimz, int dimx); // Allocate velocity and stress tensors

	void reset_fwi_kernel(int dimt, int dimz, int dimx, bool grad_reset); // Reset the kernels after each iteration

	void deallocate_fwi(int dimt, int dimz); // Deallocates velocity and stress tensors

};

#endif