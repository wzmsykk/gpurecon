#include "headerfiles.h"
int GetLines(char* filename)
{
	printf("getting number of lines for file: %s\n",filename);
    FILE *fp;
    int n=0;
    int ch;
    if((fp = fopen(filename,"r+")) == NULL)  
    {  
        fprintf(stderr,"can not open file %s\n",filename);  
        return 0;
    }

    while((ch = fgetc(fp)) != EOF) 
    {  
        if(ch == '\n')  
        {  
            n++;  
        } 
    }  

    fclose(fp); 
    return n;
}


void PrintConfig()
{
    printf(" Nx = %d\n Ny = %d\n Nz = %d\n",Nx, Ny, Nz);
    int sharedmemorySizeInt = Nx*Nz;
    double sharedMemorySize = sharedmemorySizeInt*4*4/1024.0;
    printf(" need to allocate shared memory with size: Nx x Nz = %d (%lf KB)\n",Nx*Nz, sharedMemorySize);
    printf(" refer to the max shared memory limit on the device, otherwise may cause blank image\n");
    //TEST CODE
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf(" Total amount of shared memory per block = %zu\n", deviceProp.sharedMemPerBlock);

    if (deviceProp.sharedMemPerBlock < sharedMemorySize) {
        printf("(WARNING) shared memory needed is larger than the device could provide.\n");
    }

    printf(" TOR_WIDTH = %d\n pixel_size = %lf\n ISIGMA = %lf\n",TOR_WIDTH, (double)pixel_size, ISIGMA);
}




int SaveImageToFile(float * dev_image, char* filename, int size)
{
    printf("saving image to %s \n", filename);
	float * host_image = (float *)malloc(sizeof(float)*size);
	cudaMemcpy(host_image, dev_image, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE * save_image = fopen (filename , "wb");//wb!! 
	if (save_image == NULL) 
	{
		printf("can not write to image file!\n");
		exit(1);
	}
    fwrite(host_image, sizeof(float), size, save_image);
    printf("image %s saved!\n",filename);
	fclose(save_image);
	free(host_image);    
    return 0;
    //MATRIX[Y][Z][X]
}

int SaveImageToFile_EX(float* dev_image, char* filename, int imagesize, int offset, int savesize)
{

    float* host_image = (float*)malloc(sizeof(float) * imagesize);
    cudaMemcpy(host_image, dev_image, imagesize * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* save_image = fopen(filename, "wb");//wb!! 
    if (save_image == NULL)
    {
        printf("can not write to image file!\n");
        exit(1);
    }
    float* head_ptr = host_image + offset;
    fwrite(head_ptr, sizeof(float), savesize, save_image);
    printf("saved image to %s \n", filename);
    fclose(save_image);
    free(host_image);
    return 0;
    //MATRIX[Y][Z][X]
}


int dumpAcValueAndLOR(char* filename, CUDAlor* lines, int dumpcount) {
    int i;
    CUDAlor* host_lor = (CUDAlor*)malloc(sizeof(CUDAlor) * dumpcount);
    cudaMemcpy(host_lor,lines, sizeof(CUDAlor) * dumpcount, cudaMemcpyDeviceToHost);
    CUDAlor* currline;
    FILE* fp = fopen(filename, "w");
    float x0, x1, y0, y1, z0, z1, acvalue,value;
    for (i = 0; i < dumpcount; i++) {
        currline = host_lor +i;
        x0 = currline->rx0;
        x1 = currline->rx1;
        y0 = currline->ry0;
        y1 = currline->ry1;
        z0 = currline->rz0;
        z1 = currline->rz1;
        acvalue = currline->attcorrvalue;
        value = currline->value;
        fprintf(fp, "LOR COUNT:%d x0:%f y0:%f z0:%f x1:%f y1:%f z1:%f acvalue:%f value:%f\n", i, x0, y0, z0, x1, y1, z1, acvalue, value);

    }
    free(host_lor);
    fclose(fp);
    return 0;
}