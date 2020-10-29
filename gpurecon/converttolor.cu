#include "headerfiles.h"
__global__ void convertolor(short *dev_lor_data_array, float *dx_array,float *dy_array,float *dz_array, int nlines)
{
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nlines; i += blockDim.x * gridDim.x) { 
	

	short rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2;
	rsectorID1 = *(dev_lor_data_array+0+6*i);
	moduleID1 = *(dev_lor_data_array+1+6*i);
	crystalID1 = *(dev_lor_data_array+2+6*i);
	rsectorID2 = *(dev_lor_data_array+3+6*i);
	moduleID2 = *(dev_lor_data_array+4+6*i);
	crystalID2 = *(dev_lor_data_array+5+6*i);
		
	Lorposition lor = CalcLorPositionFull(rsectorID1, rsectorID2, moduleID1, moduleID2, crystalID1, crystalID2);

	*(dx_array+i) = (lor.x1-lor.x0)/sqrt((lor.x1-lor.x0)*(lor.x1-lor.x0)+(lor.y1-lor.y0)*(lor.y1-lor.y0)+(lor.z1-lor.z0)*(lor.z1-lor.z0));//dx/dl
	*(dy_array+i) = (lor.y1-lor.y0)/sqrt((lor.x1-lor.x0)*(lor.x1-lor.x0)+(lor.y1-lor.y0)*(lor.y1-lor.y0)+(lor.z1-lor.z0)*(lor.z1-lor.z0));//dy/dl
	*(dz_array+i) = (lor.z1-lor.z0)/sqrt((lor.x1-lor.x0)*(lor.x1-lor.x0)+(lor.y1-lor.y0)*(lor.y1-lor.y0)+(lor.z1-lor.z0)*(lor.z1-lor.z0));//dz/dl
	}




__syncthreads();
}

