
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

};

#endif