//// GATE NEC CALCULATION (WITHOUT DOS BATCH)

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define		FLOAT_FORMAT float
#define		LENGTH_FILENAME_MAX 256
const int	N_FIELDS = 10;
const float 	FIVE_ELEVEN = 0.511; //MeV
const float 	E_THRESH = 0.01;//MeV		
const float	E_BLUR_511 = 0*0.12; // (%12)  energy resolution at 511 for CZT
const double	T_BLUR_511 = 0*0.000000001; //time resolution at 511 for CZT unit (ns)
const double	T_DEAD = 0*20.0*0.000000001; // non-paralyzable dead time

/** gaussian random number generator */				
/* mean 0, standard dev. 1 */
float gasrand()
{
  static int iset = 0;
  static float gset;
  float v1, v2, fac, rsq;
  if (iset == 0) {
    do {
      v1 = 2.0 * rand()/(float)RAND_MAX - 1.0;
      v2 = 2.0 * rand()/(float)RAND_MAX - 1.0;
      rsq = v1*v1 + v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    
    fac = sqrt(-2.0*log(rsq)/rsq);
    gset = v1 * fac;
    iset=1;
    return(v2*fac);
  }
  else {
    iset = 0;
    return(gset);
  }
}

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

void energyblur( FLOAT_FORMAT * e, FLOAT_FORMAT blur_pc )		
{
	if (*e<E_THRESH)
		return;
	FLOAT_FORMAT efwhm = blur_pc/100.0 * sqrt(FIVE_ELEVEN/(*e)) * (*e);	
	FLOAT_FORMAT estdev = efwhm / 2.35;			//// ENERGY STANDARD DEVIATION    FWHM = 2.35 * estdev
	FLOAT_FORMAT eblur = (float)gasrand() * estdev;
	*e += eblur;
}


void timeblur( double * t,double blur_t )				
{
	double tfwhm = blur_t;
	double tstdev = tfwhm / (2.35*1.414);
	double tblur = (double)gasrand() * tstdev;
	*t += tblur;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//**************************************************************************************************************//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)	
{

	if(argc <=1)
	{
		printf("usage: [./a.out] [dataSinglesfile]\n");
		exit(1);
	}
	
	int totalnumoflines = GetLines(argv[1]);

	
    char filename_w[LENGTH_FILENAME_MAX] = "";	////SET UP OUTPUT NAME
	strcpy(filename_w, argv[1]);			////SAME AS INPUT FILE BUT .NEC
	strcat(filename_w, ".cid");

	FILE* streamw = fopen(filename_w,"w");		////WRITE THE OUTPUT

	if ( !streamw )
		{
			fprintf(stderr,"Cannot open output file for writing.\n");
			return 1;   
		}


////      DATA INPUT AND STORAGE
	
	float sensitivity[1][1];		
	int CoincidenceTotal=0;
	int TrueTotal[1][1]={0};
	int ScatterTotal[1][1]={0};
	int RandomTotal[1][1]={0};
	int MultipleTotal[1][1]={0};
	float NEC[1][1]={0.0};
	float total_time[1][1]={0.0};

        int  E_index,T_index;		

	float nrg_window[1];
 	float time_window[1];
	int iiii;
	for (iiii=0;iiii<1;iiii++)
		{ nrg_window[iiii] = (iiii+1)*24.0;	// energy w: 3, 6, 9... 60
		  time_window[iiii] = (iiii+1)*2;     // time w  : 0.5, 1, ...10
		}
			
	for (E_index=0;E_index<1;E_index++)
		
	{
		for (T_index=0;T_index<1;T_index++)
		{
	
		// input Singles file
		FILE* streamrc = fopen(argv[1],"rt");

		if( streamrc == 0 )
		{
			fprintf(stderr,"File %s was not found.\n",argv[1]);
			return 1;		
		}

		int r=N_FIELDS; 		
		int iteration=totalnumoflines;		// LOOP OVER SINGLES FILE
		int *eventID = (int *)malloc(sizeof(int)*iteration); 
		int *rsectorID = (int *)malloc(sizeof(int)*iteration);
		int *moduleID = (int *)malloc(sizeof(int)*iteration);
		int *crystalID = (int *)malloc(sizeof(int)*iteration);
		//int *sourceID = (int *)malloc(sizeof(int)*iteration);  // need to change N_FIELD
		double *time = (double *)malloc(sizeof(double)*iteration); 
		float *energy = (float *)malloc(sizeof(float)*iteration); 
		float *x = (float *)malloc(sizeof(float)*iteration);
		float *y = (float *)malloc(sizeof(float)*iteration);
		float *z = (float *)malloc(sizeof(float)*iteration);
		int *tissueSca = (int *)malloc(sizeof(int)*iteration);
		
		// THESE MIGHT BE USELESS		
//		int last_hit=0;
//		FLOAT_FORMAT total_events=0.0;
		double last_time=0.0;
		double trigger1 =0.0;
		double trigger2 =0.0;
		//////////////////////////////	
		float Emin = FIVE_ELEVEN * (200.0 - nrg_window[E_index]) / 200.0;	
		float Emax = FIVE_ELEVEN * (200.0 + nrg_window[E_index]) / 200.0;
		double Tmax = 0.000000001*time_window[T_index];			

		int k=0;
		while ( r == N_FIELDS && k < iteration)					////LOOP FOR CALCULATION
		{
		r=fscanf(streamrc, "%*d %d %*d %f %f %f %*d %d \
		%d %*d %d %*d %lf %f %d", \
			&eventID[k], &x[k],&y[k],&z[k],&rsectorID[k],&moduleID[k],&crystalID[k],&time[k],&energy[k],&tissueSca[k]);
       		        k=k+1;
		} 			
		printf("%d",k);
		fclose(streamrc);

                // loop until EOF

	int index=0;
	iteration=k-2;					//// BECAUSE OF index + 1
	CoincidenceTotal = 0;
	double temptime;


	while (index<iteration)
	{

	timeblur( &time[index], T_BLUR_511 );		
	timeblur( &time[index+1],T_BLUR_511);
//////////////////////////////////////////   DEAD TIME   ver 1 /////////////////////////////////////////////////
//	timeblur( &time[index+1],T_BLUR_511);
//	while (index<iteration && (time[index]<trigger1 && time[index+1]<trigger2) ){
//	index = index + 2;
//	timeblur( &time[index], T_BLUR_511 );		
//	timeblur( &time[index+1],T_BLUR_511);
//	}
//	trigger1 = time[index] + T_DEAD;	trigger2 = time[index+1] + T_DEAD;
//		
//	fprintf(stdout," %f percent done\n",(double)index/(double)iteration*100.0);
//	last_time=time[index];
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////   DEAD TIME   ver 2 /////////////////////////////////////////////////
//	while (index<iteration && time[index]<trigger1 ){
//	index = index + 1;
//	timeblur( &time[index], T_BLUR_511 );		
//	}
//	trigger1 = time[index] + T_DEAD;		
////	fprintf(stdout," %f percent done\n",(double)index/(double)iteration*100.0);
//	last_time=time[index];
//	timeblur( &time[index+1],T_BLUR_511);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	if (	fabs(time[index]-time[index+1])<Tmax   ){
						
			last_time=time[index];		
			double temptime = time[index+2];
			double temptime1 = temptime;
			timeblur( &temptime,T_BLUR_511);
			if (	fabs(time[index]-temptime)<Tmax   )
			{				
					// multiple
					int tmptid=2;
					MultipleTotal[E_index][T_index]=MultipleTotal[E_index][T_index]+1;
					while (index<iteration && fabs(time[index]-temptime)<Tmax ){
					tmptid = tmptid+1;
					temptime = time[index+tmptid];
					timeblur( &temptime,T_BLUR_511);}
					if (tmptid !=2){tmptid = tmptid-1;}
					//// tmptid (>3) EVENTS ARE GONE
					index = index + tmptid;
		        }
				////////////////////////////////////////////////////////
			else if (	fabs(time[index]-temptime1)>=Tmax ) {
					// double
					// check rsectorID			
				     if (rsectorID[index] != rsectorID[index+1]) 							 				{				
				
					// measure and blur energy		
					energyblur( &energy[index], E_BLUR_511);	
					energyblur( &energy[index+1], E_BLUR_511);

					if(energy[index]>Emin && energy[index+1]>Emin && energy[index]<Emax && energy[index+1]<Emax)
				{
				
					CoincidenceTotal=CoincidenceTotal+1;	// COINCIDENCE CONDITION IS SATISFIED		
					if(eventID[index]==eventID[index+1] && (tissueSca[index]+tissueSca[index+1])==0)
						{
						TrueTotal[E_index][T_index]=TrueTotal[E_index][T_index]+1;		//TRUE
						if(rsectorID[index]%40<=rsectorID[index+1]%40){
						fprintf(streamw,"%d\t%d\t%d\t%d\t%d\t%d\n",rsectorID[index],moduleID[index],crystalID[index],rsectorID[index+1],moduleID[index+1],crystalID[index+1]);}
						else{
						fprintf(streamw,"%d\t%d\t%d\t%d\t%d\t%d\n",rsectorID[index+1],moduleID[index+1],crystalID[index+1],rsectorID[index],moduleID[index],crystalID[index]);}
						}
					else if (eventID[index]==eventID[index+1] && (tissueSca[index]+tissueSca[index+1])>0)
						{ 
						ScatterTotal[E_index][T_index]=(ScatterTotal[E_index][T_index])+1;	//SCATTER
						}
					else if(eventID[index]!=eventID[index+1])
						{ 
						RandomTotal[E_index][T_index]=(RandomTotal[E_index][T_index])+1;	//RANDOM
						}
				}
					//// TWO EVENTS ARE GONE
			}
				index = index + 2;						
								  }
				
	     						}
			
	else	
		{
			index=index+1;		//
		}

}		////ITERATION STOPS HERE


		free(eventID);
		free(rsectorID);
		free(moduleID);
		free(crystalID);
		free(time);
		free(energy); 
		free(x);
		free(y);
		free(z);
		free(tissueSca);	

	NEC[E_index][T_index]=pow(TrueTotal[E_index][T_index],2)/((TrueTotal[E_index][T_index]+ScatterTotal[E_index][T_index]+RandomTotal[E_index][T_index])*(float)last_time);
	
	total_time[E_index][T_index]=(float)(last_time);

    fprintf(stdout,"Coincidnce: %d \n",CoincidenceTotal);
	fprintf(stdout,"True : %d \n",TrueTotal[E_index][T_index]);
	fprintf(stdout,"Scatter: %d \n",ScatterTotal[E_index][T_index]);
	fprintf(stdout,"Random: %d \n",RandomTotal[E_index][T_index]);
	fprintf(stdout,"Multiple: %d \n",MultipleTotal[E_index][T_index]);
	fprintf(stdout,"Total Time: %e \n",total_time[E_index][T_index]);
	fprintf(stdout,"NEC: %f \n",NEC[E_index][T_index]);

				
	}	//// BIG LOOP INCLUDING T_INDEX STOPS HERE
}	//// BIG LOOP INCLUDING E_INDEX STOPS HERE

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//**************************************************************************************************************//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////	DATA OUTPUT
	

	
	fclose(streamw);

	return 0;
}

