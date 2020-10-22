#include"headerfiles.h"
__device__ Lorposition CalcLorPositionFull(short rsectorID1, short rsectorID2, short moduleID1, short moduleID2, short crystalID1, short crystalID2)
{
	//从晶体ID得到XYZ
    float ring_radius = 315.5;
	int module_num = 5;
	int rsector_num = 48;
	int crystalx_num = 20;
	float crystal_depth=20;
	float crystal_pitch =2;
	float PI = 3.1416;
	int crystalx_num_square=crystalx_num*crystalx_num;
	int half_z_crystals=50;// (5 rings in z axis * 20 crystals per ring) / 2 

    Lorposition lor;
	float angle1 = -(float)(rsectorID1%rsector_num)*2.0f*PI/(float)(rsector_num);
	float angle2 = -(float)(rsectorID2%rsector_num)*2.0f*PI/(float)(rsector_num);

	lor.x0 = sin(angle1)*crystal_pitch*(-(crystalx_num-1)/2.0f+((crystalID1%crystalx_num_square)%crystalx_num))+(((float)crystalID1/crystalx_num_square)*crystal_depth+ring_radius)*cos(angle1);
	lor.x1 = sin(angle2)*crystal_pitch*(-(crystalx_num-1)/2.0f+((crystalID2%crystalx_num_square)%crystalx_num))+(((float)crystalID2/crystalx_num_square)*crystal_depth+ring_radius)*cos(angle2);
	lor.y0 = cos(angle1)*crystal_pitch*(-(crystalx_num-1)/2.0f+((crystalID1%crystalx_num_square)%crystalx_num))-(((float)crystalID1/crystalx_num_square)*crystal_depth+ring_radius)*sin(angle1);
	lor.y1 = cos(angle2)*crystal_pitch*(-(crystalx_num-1)/2.0f+((crystalID2%crystalx_num_square)%crystalx_num))-(((float)crystalID2/crystalx_num_square)*crystal_depth+ring_radius)*sin(angle2);
	
	lor.z0 = crystal_pitch*((float)moduleID1*crystalx_num+(crystalID1%crystalx_num_square)/crystalx_num - half_z_crystals);
	lor.z1 = crystal_pitch*((float)moduleID2*crystalx_num+(crystalID2%crystalx_num_square)/crystalx_num - half_z_crystals); 
    
    //printf("x0=%f,x1=%f,y0=%f,y1=%f,z0=%f,z1=%f\n",lor.x0,lor.x1,lor.y0,lor.y1,lor.z0,lor.z1);

    return lor;
}
