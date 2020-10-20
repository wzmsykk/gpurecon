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
    printf(" TOR_WIDTH = %d\n pixel_size = %lf\n ISIGMA = %lf\n",TOR_WIDTH, (double)pixel_size, ISIGMA);
}


void SaveImageToFile(float * dev_image, char* filename, int size)
{

	float * host_image = (float *)malloc(sizeof(float)*size);
	cudaMemcpy(host_image, dev_image, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE * save_image = fopen (filename , "wb");//wb!! 
	if (save_image == NULL) 
	{
		printf("can not write to image file!\n");
		exit(1);
	}
    fwrite(host_image, sizeof(float), size, save_image);
    printf("saved image to %s \n",filename);
	fclose(save_image);
	free(host_image);    

    //MATRIX[Y][Z][X]
}