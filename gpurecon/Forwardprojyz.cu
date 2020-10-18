#include"headerfiles.h"
__global__ void Forwardprojyz( float *dev_image, float *lines, int linesN )
{
	//current slice
	//��x������Ƭ
	__shared__ float slice[Ny * Nz];
	int img_slice = Ny * Nz;
	float max_distance = TOR_WIDTH * TOR_WIDTH / 2.0f;
	// loop over slices
	for (int current_slice = 0; current_slice < Nx; ++current_slice){
		
		int offset = current_slice * img_slice; 
		for (int vi = threadIdx.x; vi < img_slice; vi += blockDim.x){
			
			slice[vi] = dev_image[vi + offset]; 
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
		//centerY Ϊ���ڽ��洦���ĵ�
		float sum = 0;
		// search pixels within the tube (eclipse)
  		for (int yy = centerY - TOR_WIDTH; yy <= centerY + TOR_WIDTH; ++yy){
    		for (int zz = centerZ - TOR_WIDTH; zz <= centerZ + TOR_WIDTH; ++zz){
      			if ( yy >= 0 && yy < Ny && zz >= 0 && zz < Nz) {
         			float dy = yy - y, dz = zz - z;
         			float inner = dy * l1 + dz * l2; 	// OQ
					//YY ZZ �����ĵ�TOR��������R
					//dy dzΪ����R��һ��P�������ĵ�λ��
					//Q ΪP������ͶӰ
					//�����ΪOP^2-OQ^2
         			// Distance to the line, squared 
         			float d2 = dy * dy + dz * dz - inner * inner;	// OP^2 - OQ^2 
         			float kern = (d2 < max_distance) ? exp(-d2 * ISIGMA) : 0;
         			sum += slice[yy + Ny * zz] * kern; 
					//�õ���SLICE���
      			} //endif
   			} // endfor Write the value back to global memory 
		} //endfor
		  the_line->value += sum;
	} // endfor line	  
   __syncthreads(); 
  }  // endfor slice
__syncthreads();
}   // endfunction
