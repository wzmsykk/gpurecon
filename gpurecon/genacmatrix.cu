#include"headerfiles.h"
void genctdim(CTdims* host_ctdim, const char* ctheader);
void genctdim(CTdims* host_ctdim, char* ctheader) {
	FILE* fp;
	char* linebuffer;
	char* p, * vp;
	int totalnumoflines;
	int cmpres = 0;
	if (ctheader != "") {
		fp = fopen(ctheader, "r");
		if (fp == nullptr) {
			printf("(ERROR) CT headerfile not found.\n");
			exit(1);
		}
		totalnumoflines = GetLines_c(ctheader);
		printf("totallines=%d\n", totalnumoflines);
		linebuffer = (char*)malloc(sizeof(char) * 512);
		for (int i = 0; i < totalnumoflines; i++) {
			fgets(linebuffer, 512, fp);
			p = strtok(linebuffer, " ");
			if (!strcmp(p, "DimSize")) {
				p = strtok(NULL, "=");
				vp = strtok(p, " ");
				printf("vp=%s", vp);
				host_ctdim->xdim = atoi(vp);
				vp = strtok(NULL, " ");
				host_ctdim->ydim = atoi(vp);
				vp = strtok(NULL, " ");
				host_ctdim->zdim = atoi(vp);
			}
			else if (!strcmp(p, "Offset")) {
				p = strtok(NULL, "=");
				vp = strtok(p, " ");
				host_ctdim->x0 = atof(vp);
				vp = strtok(NULL, " ");
				host_ctdim->y0 = atof(vp);
				vp = strtok(NULL, " ");
				host_ctdim->z0 = atof(vp);
			}
			else if (!strcmp(p, "ElementSpacing")) {
				p = strtok(NULL, "=");
				vp = strtok(p, " ");
				host_ctdim->xspacing = atof(vp);
				vp = strtok(NULL, " ");
				host_ctdim->yspacing = atof(vp);
				vp = strtok(NULL, " ");
				host_ctdim->zspacing = atof(vp);
			}
		}
		fclose(fp);
	}
	else {
		host_ctdim->x0 = -Nx * pixel_size / 2.0f;
		host_ctdim->y0 = -Ny * pixel_size / 2.0f;
		host_ctdim->z0 = -Nz * pixel_size / 2.0f;
		host_ctdim->xspacing = pixel_size;
		host_ctdim->yspacing = pixel_size;
		host_ctdim->zspacing = pixel_size;
		host_ctdim->xdim = Nx;
		host_ctdim->ydim = Ny;
		host_ctdim->zdim = Nz;
	}
	printf("CT header read OK.\n");
	printf("(DEBUG) xoffset=%f,yoffset=%f,zoffset=%f\n", host_ctdim->x0, host_ctdim->y0, host_ctdim->z0);
	printf("(DEBUG) xspace=%f,yspace=%f,zspace=%f\n", host_ctdim->xspacing, host_ctdim->yspacing, host_ctdim->zspacing);
	printf("(DEBUG) xdim=%d,ydim=%d,zdim=%d\n", host_ctdim->xdim, host_ctdim->ydim, host_ctdim->zdim);
}
int genacmatrix(float* dev_attenuation_matrix, CTdims* dev_ctdim, short* dev_ct_matrix, char* ct_header_filename) {
	CTdims* host_ctdim;
	host_ctdim = (CTdims*)malloc(sizeof(CTdims));

	genctdim(host_ctdim, ct_header_filename);
	cudaMemcpy(dev_ctdim, host_ctdim, sizeof(CTdims), cudaMemcpyHostToDevice);
	free(host_ctdim);

	cudaDeviceSynchronize();

	genacvalue << <256, 512 >> > (dev_attenuation_matrix, dev_ctdim, dev_ct_matrix);
	cudaDeviceSynchronize();
	return 0;
}
__global__ void genacvalue(float* attenuation_matrix, CTdims* dev_ctdim, short* dev_ct_matrix) {
	//do nothing
	////miu=1e-8
	//AM[X][Y][Z]
	//CT[X][Y][Z]
	//请用SIMPLEITK等工具将CT缩放+平移与PET图像同XYZ轴
	int xdim = dev_ctdim->xdim;
	int ydim = dev_ctdim->ydim;
	int zdim = dev_ctdim->zdim;
	int totalvoxels = xdim * ydim * zdim;
	for (int line_index = threadIdx.x + blockIdx.x * blockDim.x; line_index < totalvoxels; line_index += blockDim.x * gridDim.x) {
		int mz = line_index / (xdim * ydim);
		int temp = line_index - mz * (xdim * ydim);
		int my = temp / xdim;
		int mx = temp - my * xdim;
		short ctvalue = dev_ct_matrix[line_index];
		//short value = dev_ct_matrix[mx*Ny*Nz+my*Nz+mz];
		//CT HU TO ATTENUATION LIST
		//H20 9.598E-02 cm-1=9.598E-3 mm-1
		//if (value<A && ){
		//atv=CONST01
		//}else if ..
		//TO DO 
		//根据CT 的HU值查找衰减
		//现在水模高2cm
		//半径5cm
		/*float rx = ((float)mx + 0.5) * dev_ctdim->xspacing + dev_ctdim->x0;
		float ry = ((float)my + 0.5) * dev_ctdim->yspacing + dev_ctdim->y0;
		float rz = ((float)mz + 0.5) * dev_ctdim->zspacing + dev_ctdim->z0;
		if ((rx * rx + ry * ry) < 2500.0f && rz <10.0f && rz>-10.0f) {
			attenuation_matrix[mz* dev_ctdim->ydim * dev_ctdim->xdim +my* dev_ctdim->xdim +mx] = 9.598E-03;
		}
		else {
			attenuation_matrix[mz * dev_ctdim->ydim * dev_ctdim->xdim + my * dev_ctdim->xdim + mx] = 0;
		}//YZX		*/
		if (dev_ct_matrix == nullptr) {
			attenuation_matrix[mz * dev_ctdim->ydim * dev_ctdim->xdim + my * dev_ctdim->xdim + mx] = 0;
		}
		else {
			if (ctvalue >= 0) {
				attenuation_matrix[mz * dev_ctdim->ydim * dev_ctdim->xdim + my * dev_ctdim->xdim + mx] = 0.1 * (0.096 + ctvalue * (0.172 - 0.096) / 1400.0);// unit per mm
			}
			else if (ctvalue < 0) {
				attenuation_matrix[mz * dev_ctdim->ydim * dev_ctdim->xdim + my * dev_ctdim->xdim + mx] = 0.1 * (ctvalue + 1000) * (0.096) / 1000.0;// unit per mm
			}
		}



	}__syncthreads();
}