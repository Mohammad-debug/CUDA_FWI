//fd_grid.cuh

/*
* Created by: Min Basnet
* 2020.April.13
* Kathmandu, Nepal
*/

#ifndef FD_GRID_H				
#define FD_GRID_H	

#include "globvar.cuh"

class FDGrid {
public:
	int nt, nx, ny, nz; // Number of grids in time and space
	real_sim dt, dx, dy, dz; // Grid intervals in time and space

	int fpad, ppad; // Additional grids to accommodate fdorder and PML boundary layer
	bool fsurf = 0; // Free surface on the top
	int nxt, nzt; // total grid size after padding

	int snap_interval; // time_step intervals for data output and print on the screen

	//explicit FDGrid(const char *configuration_file_relative_path);FDCpml();
	//FDGrid();
	//~FDGrid();

	void parse_configuration(const char* configuration_file_relative_path);

	void total_grid(int fdorder, int npml); // Add pml and grids due to fdorder

};

#endif