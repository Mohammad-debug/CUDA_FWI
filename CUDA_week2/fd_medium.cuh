
/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

#ifndef FD_MEDIUM_H				
#define FD_MEDIUM_H	

#include "globvar.cuh"

class FDMedium {
public:
	// Scalar elastic material parameters
	real_sim scalar_rho, scalar_vp, scalar_vs;
	real_sim scalar_mu, scalar_lam;
	real_sim** rho_old, ** mu_old, ** lam_old;

	// 2D model space parameters
	real_sim** rho, ** mu, ** lam; // Lami's parameters
	real_sim** vp, ** vs; // Wave velocitis
	real_sim** rho_xp, ** rho_zp, ** mu_zx; // Interpolated materials for staggered grids

	//FDMedium(); // Default constructor
	//explicit FDMedium(const char *configuration_file_relative_path); // constructor
	//~FDMedium(); // Destructor

	void parse_configuration(const char* configuration_file_relative_path); // parse input

	void velocity_to_lame(); // create scalar lame parameters

	void stagger_over_grid(int dimz, int dimx); // stagger scalar lami parameter over grid

	void average_parameters(int dimz, int dimx); // Calculate required material averages

	void allocate_medium(int dimz, int dimx); // Allocate lame's parameters and rho

	void allocate_medium_av(int dimz, int dimx); // Allocate averaged medium parameters

	void deallocate_medium(int dimz); // Deallocate lame's parameters and rho

	void deallocate_medium_av(int dimz); // Deallocate average medium parameters 
	// temporary ones
	void stagger_over_grid_modify(int dimz, int dimx); // stagger scalar lami parameter over grid

	void fwi_grad_update(real_sim** mat, real_sim** mat_new, real_sim** grad_mat,
		real_sim step_length, int dimz, int dimx,
		int z1, int x1, int fwi_dz, int fwi_dx, bool override, int matType);

	void medium_average(real_sim& C_lam, real_sim& C_mu, real_sim& C_rho, int dimz, int dimx);

};

#endif