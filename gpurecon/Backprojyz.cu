#include"headerfiles.h"
__global__ void Backprojyz(float *dev_image, float *back_image, CUDAlor *lines,  int linesN, int backProjOnly )
{
	//current slice
	__shared__ float slice[Ny * Nz];
	int img_slice = Ny * Nz;
	float max_distance = TOR_WIDTH * TOR_WIDTH / 2.0f;
	// loop over slices
	for (int current_slice = 0; current_slice < Nx; ++current_slice){
		
		int offset = current_slice * img_slice; 
		for (int vi = threadIdx.x; vi < img_slice; vi += blockDim.x){
			
			slice[vi] = 0; // back_image[vi+offset] = 0; 
		}
		__syncthreads();
		for (int line = threadIdx.x + blockIdx.x * blockDim.x; line < linesN; line += blockDim.x * gridDim.x) { 

		// convert pointer type to struct
		CUDAlor *the_line = (CUDAlor*)lines + line;  

		// line direction
		float l1 = the_line->dy, l2 = the_line->dz;
		float t = (current_slice - the_line->x0) / the_line->dx;
		// point O
		float y = the_line->y0 + t * l1, z = the_line->z0 + t * l2;
		int centerY = floor(y+0.5), centerZ = floor(z+0.5); 
		// search pixels within the tube (eclipse)
  		for (int yy = centerY - TOR_WIDTH; yy <= centerY + TOR_WIDTH; ++yy){
    		for (int zz = centerZ - TOR_WIDTH; zz <= centerZ + TOR_WIDTH; ++zz){
      			if ( yy >= 0 && yy < Ny && zz >= 0 && zz < Nz) {
         			float dy = yy - centerY, dz = zz - centerZ;
         			float inner = dy * l1 + dz * l2; 

         			// Distance to the line, squared 
         			float d2 = dy * dy + dz * dz - inner * inner; 
         			float kern = (d2 < max_distance) ? exp(-d2 * ISIGMA) : 0;
				float divvalue = backProjOnly>0 ? 1 : the_line->value;
				float thevalue = ((the_line->value)>0) ? (kern/divvalue) : 0; 
				atomicAdd(&slice[yy + Ny * zz], thevalue);
      			} //endif
   		} // endfor Write the value back to global memory 
		} // endfor

	}__syncthreads();	// endfor line
		for (int vi = threadIdx.x; vi < img_slice; vi += blockDim.x){	
			back_image[vi+offset]+=slice[vi];	  
   		}__syncthreads();
  } // endfor slice
} //end function
