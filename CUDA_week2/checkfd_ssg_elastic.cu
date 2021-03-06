#include <cmath>
#include <iostream>
#include "ext.cuh"
#include "globvar.cuh"

void checkfd_ssg_elastic(int NX, int NY, real_sim DH, real_sim DT, real_sim TS, int FW,
	real_sim** prho, real_sim** ppi, real_sim** pu, real_sim* hc) {


	const int MYID = 0;

	/* local variables */

	real_sim  c, cmax_p = 0.0, cmin_p = 1e9, cmax_s = 0.0, cmin_s = 1e9, fmax, gamma;
	real_sim  cmax = 0.0, cmin = 1e9, dtstab, dhstab;
	int nfw = round(FW / DH);
	int i, j, ny1 = 1, nx, ny, nx_min, ny_min;

	int neglect_zero_cs = 0;


	nx = NX; ny = NY;

	/* low Q frame not yet applied as a absorbing boundary */
	/* if (!FREE_SURF) ny1=1+nfw;*/
	nfw = 0;


	/* find maximum model phase velocity of shear waves at infinite
		  frequency within the whole model */
	for (i = 1 + nfw; i <= (nx - nfw); i++) {
		for (j = ny1; j <= (ny - nfw); j++) {

			//if(INVMAT1==3){
			if (fabs(pu[j][i]) < 1.0e-9) {
				neglect_zero_cs = 1;
			}
			else c = sqrt(pu[j][i] / prho[j][i]);
			//}
			//if(INVMAT1==1){
			//c=pu[j][i];}

			if (cmax_s < c) cmax_s = c;
			if (cmin_s > c) cmin_s = c;
		}
	}



	/* find maximum model phase velocity of P-waves at infinite
		 frequency within the whole model */
	for (i = 1 + nfw; i <= (nx - nfw); i++) {
		for (j = ny1; j <= (ny - nfw); j++) {

			//if(INVMAT1==3){
			c = sqrt((ppi[j][i] + 2.0 * pu[j][i]) / prho[j][i]);//}

			//if(INVMAT1==1){
			//c=ppi[j][i];}

			if (cmax_p < c) cmax_p = c;
			if (cmin_p > c) cmin_p = c;
		}
	}


	if (cmax_s > cmax_p) cmax = cmax_s;
	else cmax = cmax_p;
	if (cmin_s < cmin_p) cmin = cmin_s;
	else cmin = cmin_p;


	fmax = 2.0 * TS; // expression modifiec
	dhstab = (cmin / (12.0 * fmax));
	gamma = fabs(hc[1]) + fabs(hc[2]) + fabs(hc[3]) + fabs(hc[4]) + fabs(hc[5]) + fabs(hc[6]);
	dtstab = DH / (sqrt(2) * gamma * cmax);
	/*dtstab=DH/(sqrt(2.0)*cmax);*/

	nx_min = nx - FW;
	ny_min = ny - FW;

	if (MYID == 0) {

		if (neglect_zero_cs == 1) std::cout << "WARNING: zero shear waves neglected" << std::endl;

		if (DH > dhstab) {
			std::cout << "WARNING:: Grid dispersion will influence wave propagation, choose smaller grid spacing (DH)." << std::endl;
			std::cout << "DHSTAB = " << dhstab << ", DH = " << DH << std::endl;
		}
		else {
			std::cout << "DHSTAB = " << dhstab << ", DH = " << DH << std::endl;
		}

		if (DT > dtstab) {
			std::cout << " The simulation will get unstable, choose smaller DT. DTSTAB = " << dtstab << " DT = " << DT << std::endl;
		}
		else {
			std::cout << " The simulation will be stable. DTSTAB = " << dtstab << " DT = " << DT << std::endl;
		}

		std::cout << std::endl << "----------------------- ABSORBING BOUNDARY ------------------------" << std::endl;
		if ((FW > nx_min) || (FW > ny_min)) {
			std::cout << "ERROR:: The width of the absorbing boundary is larger than one computational domain. Choose smaller FW or use less CPUs." << std::endl;
			exit(0);
		}

		std::cout << " Width (FW) of absorbing frame should be at least 10 gridpoints." << std::endl;
		std::cout << " You have specified a width of " << FW << " gridpoints.\n" << std::endl;
		if (FW < 10)
			std::cout << "WARNING:: Be aware of artificial reflections from grid boundaries !" << std::endl;

	}

}