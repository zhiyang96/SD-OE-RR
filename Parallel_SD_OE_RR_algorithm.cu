
/*  Parallel SD-OE-RR algorithm for Compton camera imaging
    Zhiyang Yao,Department of Engineering Physics,Tsinghua University
    2021/7/4, 2D reconstruction version
    FOV: projected coordinate system (θ_xz，θ_yz，z）based on spherical coordinates on the plane z from the detector.
    Voxels based BP with resolution recovery & subset-driven OE iteration
    Ref: Yao Z, Yuan Y, Wu J, et al. Rapid compton camera imaging for source terms investigation in the nuclear decommissioning with a subset-driven origin ensemble algorithm. 
         Radiation Physics and Chemistry, 2022: 110133. 
*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <cmath>
#include "math.h"
#include <stdlib.h>
#include <list>
#include <vector>
#include <time.h>
#include <assert.h>
#include <curand_kernel.h>
#include <curand.h>
#define SOE_itr 1e7

using namespace std;

// some function used in CPU
__host__ float rand_FloatRange(float a, float b)   // Generate the random number, used in OE iteration
{
  return ((b - a) * (rand() / (float)RAND_MAX) + a);
}

// some functions used in GPU
__device__ float GetLength(float x11, float y11, float z11, float x21, float y21, float z21)
{  // Obtain the spatial distance between two point in GPU
   float len = sqrt(pow((x11 - x21), 2) + pow((y11 - y21), 2) + pow((z11 - z21), 2));
   return len;
}
__device__ float GetComptonAngle(float e_scatter,float e_absorb)
{  // Calculate the cosine of the scattering angle in GPU
   // m0*C^2 = 510.99 keV
   float mcomptonAngle = 1 - 510.99 * ((1.0 / e_absorb) - (1.0 / (e_absorb + e_scatter)));
   //float mcomptonAngle = 1 - 510.99 * ((1.0 / e_absorb) - (1.0 / (662.0)));
   return mcomptonAngle;
}

// GPU kernel function for voxels based BP with resolution recovery
__global__ void Recon_BP_2D(float x1, float y1,float z1,float x2,float y2,float z2,float e1,float e2,int *pDensity,int Voxels_Num,float z_plane)
{
   int tid = threadIdx.x + blockIdx.x* blockDim.x;
   int tid_y = threadIdx.x;
   int tid_x = blockIdx.x;
   float d_theta = 0.01;
   int XYbins = 128;
   float M_Pi = 3.141592657;
   float theta_min = -M_Pi*0.5;
   float theta_max = M_Pi*0.5;
   float voxel_theta = (theta_max - theta_min)/(float)XYbins;
   float theta_x0, theta_y0, x0, y0, z0;
   float len1,len2,Prod,r_comptonAngle,mcomptonAngle,discT;
   // obtain the coordinates in Cartesian coordinate system
   z0 = z_plane;
   theta_x0 = (tid_x - XYbins/2 + 0.5)*voxel_theta;
   theta_y0 = (tid_y - XYbins/2 + 0.5)*voxel_theta;
   // transfer coordinates to projected coordinate system
   x0 = z0 * tan(theta_x0);
   y0 = z0 * tan(theta_y0);
   if(tid < Voxels_Num)
   {
      if(e1!=0)
      {
         len1 = GetLength(x0, y0, z0, x1, y1, z1);
         len2 = GetLength(x1, y1, z1, x2, y2, z2);
         Prod = (x0-x1) * (x1-x2) + (y0 - y1) * (y1 - y2) + (z0 - z1) * (z1-z2);
         r_comptonAngle = (float)Prod / (len1 * len2);
         mcomptonAngle = GetComptonAngle(e1,e2);
         discT = abs(r_comptonAngle - mcomptonAngle);
         if (discT < d_theta)
         {  
            atomicAdd(&(pDensity[tid_y+XYbins*tid_x]),1);           
         }
      }
   }
}

// GPU main cuda function for 2D imaging
void PBP_RR(float x1, float y1, float z1, float x2, float y2, float z2, float e1,float e2,int *pDensity, int Image_size, float z_plane)
{
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    
    // Allocate GPU buffers for probability density function 
    int *dev_pDensity;
    cudaMalloc((void**)&dev_pDensity, Image_size * sizeof(int));

    // Launch a Back-projection kernel function on the GPU with one thread for each event and each voxel.
    int num_threads = 128;
    int num_blocks = 128;
    Recon_BP_2D<<<num_blocks, num_threads>>>(x1,y1,z1,x2,y2,z2,e1,e2,dev_pDensity,Image_size,z_plane);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(pDensity, dev_pDensity, sizeof(float) * Image_size, cudaMemcpyDeviceToHost);

   //  // Check the runing state of the kernel function
   //  cudaError_t error = cudaGetLastError();
   //  printf("CUDA error: %s\n",cudaGetErrorString(error));

    // Free the  GPU buffer
    cudaFree(dev_pDensity);
}

// CPU main function
int main()
{
    clock_t t_start,t_itr_start;
    t_start = clock();
    const int Image_arraySize_XY = 128;   //set the pixels' number : 128*128 in XY-plane

    /***********************************Input the List-mode Data & Store*******************************/
    std::vector<float> x1;
    std::vector<float> y1;
    std::vector<float> z1;
    std::vector<float> e1;
    std::vector<float> x2;
    std::vector<float> y2;
    std::vector<float> z2;
    std::vector<float> e2;

    ifstream infile;
    infile.open("CC_data_listmode.txt", ios::in); //this one to be used
    if (!infile.is_open())
    {
       exit(1);
    }
  
    float x1_,y1_,z1_,e1_,x2_,y2_,z2_,e2_;
    int event_ID = 0;
    while (!infile.eof())
    {
       infile >> x1_ >> y1_ >> z1_ >> e1_ >> x2_ >> y2_ >> z2_ >> e2_;  //unit:(mm) & (keV)
       x1.push_back(x1_);
       y1.push_back(y1_);
       z1.push_back(z1_);
       e1.push_back(e1_);
       x2.push_back(x2_);
       y2.push_back(y2_);
       z2.push_back(z2_);
       e2.push_back(e2_); 
       event_ID++;
   }
   cout << "The numbers of events are: " << (event_ID-1) << endl;

   int Image_PixelNum = Image_arraySize_XY*Image_arraySize_XY;
   int *pDensity_single_Event = new int[Image_PixelNum];
   int *pDensity_2D = new int[Image_PixelNum];
   // initialize the pDensity matrix
   for (int i = 0; i < Image_arraySize_XY; i++)
   {
      for (int j = 0; j < Image_arraySize_XY; j++)
      {   
         pDensity_single_Event[i+Image_arraySize_XY*j]=0;
         pDensity_2D[i+Image_arraySize_XY*j]=0;
      }
   }

   /***************************************************Voxels based Pre-Back-projection with resolution recovery (PBP-RR)**************************************/
    const int event_num = event_ID;
   std::vector<std::vector<int> > PDF_2D(event_num);
   int old_pos;
   float z_plane = 1000.0;   // set the Z-plane
   int pixel_Id;
   for(int event_id=0;event_id<event_num;event_id++)
   {
      // Calculate the back-projection results for each event
      PBP_RR(x1[event_id], y1[event_id], z1[event_id], x2[event_id], y2[event_id], z2[event_id], e1[event_id], e2[event_id], pDensity_single_Event, Image_PixelNum, z_plane);

      // Update the probability density functon in the FOV and store the origin ensembles of each event
      for (int i = 0; i < Image_arraySize_XY; i++)
      {
         for (int j = 0; j < Image_arraySize_XY; j++)
         {
            pixel_Id=i+Image_arraySize_XY*j;
            if(pDensity_single_Event[pixel_Id]!=0){
                PDF_2D[event_id].push_back(pixel_Id);
                pDensity_2D[pixel_Id]+=pDensity_single_Event[pixel_Id];
                old_pos=pixel_Id;
                }
            }
      }
      //cout<<"PBP-RR & SD-OE storage: " << event_id << " -th event calculation completed." << endl;
   }
   std::cout << "The PBP-RR running time: " << float((clock() - t_start)) / CLOCKS_PER_SEC << std::endl;

   /*************************************SD-OE iteration*****************************************/
   t_itr_start=clock();
   cout << "Start SD-OE iteration." << endl;
   int new_pos;
   int order_OE_itr = SOE_itr/event_num;   // obtain the numbers of ordered OE iterations
   float ifAccept,Accept_rand;
   for (int itr_n=0;itr_n<order_OE_itr;itr_n++)   // The total numbers of OE iterations = (The numbers of ordered OE iterations) * (The numbers of events)
   {
      for (int e_id = 0; e_id<event_num; e_id++)
      {
        float random_num = rand_FloatRange(0.0,1.0);
        int Pos_Size = PDF_2D[e_id].size();     // obtain the subset of origin ensemble of a event.
        if(Pos_Size>0){
        int i2 = floor(random_num*Pos_Size);
        new_pos=PDF_2D[e_id][i2];

        if (new_pos != old_pos)
        {
            if(pDensity_2D[old_pos]!=0){
              ifAccept = (pDensity_2D[new_pos]+1.0)/(float)(pDensity_2D[old_pos]);
              if(ifAccept>=1){
                 pDensity_2D[old_pos]--;
                 old_pos=new_pos;
                 pDensity_2D[new_pos]++;
              }
              else{
                Accept_rand=rand_FloatRange(0.0,1.0);
                if(ifAccept>Accept_rand){
                 pDensity_2D[old_pos]--;
                 old_pos=new_pos;
                 pDensity_2D[new_pos]++;
              }
              else{
                pDensity_2D[old_pos]++;
              }
              }
            }
            else{
                 old_pos=new_pos;
                 pDensity_2D[new_pos]++;
            }
          }
        }
      } 
  }
  cout << "SD-OE-RR iteration Completed:   " <<  SOE_itr << "  times OE iteration." << endl;

  /**********************************************Output the 2D-pDesnity Matrix(Reconstruction Image)*******************************/
  std::ofstream outList4;
  outList4.open("Reconstruction_results.txt", ios::app);
  if (!outList4)
  {
    cout << "Open the file failure..\n"
         << endl;
  }
  for (int j = 0; j < Image_arraySize_XY; j++)
  {
    for (int k = 0; k < Image_arraySize_XY; k++)
    {
      outList4 << pDensity_2D[j+Image_arraySize_XY*k] << " ";
    }
    outList4 << endl;
    }
   outList4.close();

   std::cout << "OE iteration running time:  " << float((clock() - t_itr_start)) / CLOCKS_PER_SEC << " s." << std::endl;
   return 0;
}
