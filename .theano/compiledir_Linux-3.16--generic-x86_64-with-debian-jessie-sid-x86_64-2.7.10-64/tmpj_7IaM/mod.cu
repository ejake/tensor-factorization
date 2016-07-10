#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include "cuda_ndarray.cuh"
//////////////////////
////  Support Code
//////////////////////


#define INTDIV_POW2(a, b) (a >> b)
#define INTMOD_POW2(a, b) (a & ((1<<b)-1))
        // GpuElemwise{Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))}}[(0, 0)]
// node.op.destroy_map={0: [0]}
//    Input   0 CudaNdarrayType(float32, matrix)
//    Input   1 CudaNdarrayType(float32, (True, True))
//    Input   2 CudaNdarrayType(float32, col)
//    Input   3 CudaNdarrayType(float32, matrix)
//    Input   4 CudaNdarrayType(float32, matrix)
//    Input   5 CudaNdarrayType(float32, (True, True))
//    Input   6 CudaNdarrayType(float32, (True, True))
//    Input   7 CudaNdarrayType(float32, matrix)
//    Input   8 CudaNdarrayType(float32, matrix)
//    Input   9 CudaNdarrayType(float32, matrix)
//    Input   10 CudaNdarrayType(float32, (True, True))
//    Output  0 CudaNdarrayType(float32, matrix)
static __global__ void kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_1(unsigned int numEls
	, const int dim0
	, const float * i0_data, int i0_str_0
	, const float * i1_data, int i1_str_0
	, const float * i2_data, int i2_str_0
	, const float * i3_data, int i3_str_0
	, const float * i4_data, int i4_str_0
	, const float * i5_data, int i5_str_0
	, const float * i6_data, int i6_str_0
	, const float * i7_data, int i7_str_0
	, const float * i8_data, int i8_str_0
	, const float * i9_data, int i9_str_0
	, const float * i10_data, int i10_str_0
	, float * o0_data, int o0_str_0
	)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const float ii_i1_value = i1_data[0];
    const float ii_i5_value = i5_data[0];
    const float ii_i6_value = i6_data[0];
    const float ii_i10_value = i10_data[0];
    for (int i = idx; i < numEls; i += numThreads) {
        int ii = i;
        const float * ii_i0_data = i0_data;
        const float * ii_i2_data = i2_data;
        const float * ii_i3_data = i3_data;
        const float * ii_i4_data = i4_data;
        const float * ii_i7_data = i7_data;
        const float * ii_i8_data = i8_data;
        const float * ii_i9_data = i9_data;
        float * ii_o0_data = o0_data;
        int pos0 = ii;
        ii_i0_data += pos0 * i0_str_0;
        ii_i2_data += pos0 * i2_str_0;
        ii_i3_data += pos0 * i3_str_0;
        ii_i4_data += pos0 * i4_str_0;
        ii_i7_data += pos0 * i7_str_0;
        ii_i8_data += pos0 * i8_str_0;
        ii_i9_data += pos0 * i9_str_0;
        ii_o0_data += pos0 * o0_str_0;
npy_float32 o0_i;
        {
npy_float32 V_DUMMY_ID__tmp1;
V_DUMMY_ID__tmp1 = ii_i10_value - ii_i4_data[0];
npy_float32 V_DUMMY_ID__tmp2;
V_DUMMY_ID__tmp2 = ii_i5_value * ii_i6_value * ii_i9_data[0];
npy_float32 V_DUMMY_ID__tmp3;
V_DUMMY_ID__tmp3 = ii_i2_data[0] * ii_i8_data[0] * ii_i4_data[0];
npy_float32 V_DUMMY_ID__tmp4;
V_DUMMY_ID__tmp4 = ii_i5_value * ii_i6_value * ii_i7_data[0];
npy_float32 V_DUMMY_ID__tmp5;
V_DUMMY_ID__tmp5 = ii_i1_value * ii_i2_data[0] * ii_i3_data[0] * ii_i4_data[0];
npy_float32 V_DUMMY_ID__tmp6;
V_DUMMY_ID__tmp6 = V_DUMMY_ID__tmp3 / V_DUMMY_ID__tmp2;
npy_float32 V_DUMMY_ID__tmp7;
V_DUMMY_ID__tmp7 = V_DUMMY_ID__tmp5 / V_DUMMY_ID__tmp4;
npy_float32 V_DUMMY_ID__tmp8;
V_DUMMY_ID__tmp8 = V_DUMMY_ID__tmp7 + V_DUMMY_ID__tmp6;
o0_i = ii_i0_data[0] * V_DUMMY_ID__tmp8 * V_DUMMY_ID__tmp1;
}

ii_o0_data[0] = o0_i;
    }
}
// GpuElemwise{Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))}}[(0, 0)]
// node.op.destroy_map={0: [0]}
//    Input   0 CudaNdarrayType(float32, matrix)
//    Input   1 CudaNdarrayType(float32, (True, True))
//    Input   2 CudaNdarrayType(float32, col)
//    Input   3 CudaNdarrayType(float32, matrix)
//    Input   4 CudaNdarrayType(float32, matrix)
//    Input   5 CudaNdarrayType(float32, (True, True))
//    Input   6 CudaNdarrayType(float32, (True, True))
//    Input   7 CudaNdarrayType(float32, matrix)
//    Input   8 CudaNdarrayType(float32, matrix)
//    Input   9 CudaNdarrayType(float32, matrix)
//    Input   10 CudaNdarrayType(float32, (True, True))
//    Output  0 CudaNdarrayType(float32, matrix)
static __global__ void kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_2(unsigned int numEls
	, const int dim0, const int dim1
	, const float * i0_data, int i0_str_0, int i0_str_1
	, const float * i1_data, int i1_str_0, int i1_str_1
	, const float * i2_data, int i2_str_0, int i2_str_1
	, const float * i3_data, int i3_str_0, int i3_str_1
	, const float * i4_data, int i4_str_0, int i4_str_1
	, const float * i5_data, int i5_str_0, int i5_str_1
	, const float * i6_data, int i6_str_0, int i6_str_1
	, const float * i7_data, int i7_str_0, int i7_str_1
	, const float * i8_data, int i8_str_0, int i8_str_1
	, const float * i9_data, int i9_str_0, int i9_str_1
	, const float * i10_data, int i10_str_0, int i10_str_1
	, float * o0_data, int o0_str_0, int o0_str_1
	)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const float ii_i1_value = i1_data[0];
    const float ii_i5_value = i5_data[0];
    const float ii_i6_value = i6_data[0];
    const float ii_i10_value = i10_data[0];
    for (int i = idx; i < numEls; i += numThreads) {
        int ii = i;
        const float * ii_i0_data = i0_data;
        const float * ii_i2_data = i2_data;
        const float * ii_i3_data = i3_data;
        const float * ii_i4_data = i4_data;
        const float * ii_i7_data = i7_data;
        const float * ii_i8_data = i8_data;
        const float * ii_i9_data = i9_data;
        float * ii_o0_data = o0_data;
        int pos1 = ii % dim1;
        ii = ii / dim1;
        ii_i0_data += pos1 * i0_str_1;
        ii_i2_data += pos1 * i2_str_1;
        ii_i3_data += pos1 * i3_str_1;
        ii_i4_data += pos1 * i4_str_1;
        ii_i7_data += pos1 * i7_str_1;
        ii_i8_data += pos1 * i8_str_1;
        ii_i9_data += pos1 * i9_str_1;
        ii_o0_data += pos1 * o0_str_1;
        int pos0 = ii;
        ii_i0_data += pos0 * i0_str_0;
        ii_i2_data += pos0 * i2_str_0;
        ii_i3_data += pos0 * i3_str_0;
        ii_i4_data += pos0 * i4_str_0;
        ii_i7_data += pos0 * i7_str_0;
        ii_i8_data += pos0 * i8_str_0;
        ii_i9_data += pos0 * i9_str_0;
        ii_o0_data += pos0 * o0_str_0;
npy_float32 o0_i;
        {
npy_float32 V_DUMMY_ID__tmp1;
V_DUMMY_ID__tmp1 = ii_i10_value - ii_i4_data[0];
npy_float32 V_DUMMY_ID__tmp2;
V_DUMMY_ID__tmp2 = ii_i5_value * ii_i6_value * ii_i9_data[0];
npy_float32 V_DUMMY_ID__tmp3;
V_DUMMY_ID__tmp3 = ii_i2_data[0] * ii_i8_data[0] * ii_i4_data[0];
npy_float32 V_DUMMY_ID__tmp4;
V_DUMMY_ID__tmp4 = ii_i5_value * ii_i6_value * ii_i7_data[0];
npy_float32 V_DUMMY_ID__tmp5;
V_DUMMY_ID__tmp5 = ii_i1_value * ii_i2_data[0] * ii_i3_data[0] * ii_i4_data[0];
npy_float32 V_DUMMY_ID__tmp6;
V_DUMMY_ID__tmp6 = V_DUMMY_ID__tmp3 / V_DUMMY_ID__tmp2;
npy_float32 V_DUMMY_ID__tmp7;
V_DUMMY_ID__tmp7 = V_DUMMY_ID__tmp5 / V_DUMMY_ID__tmp4;
npy_float32 V_DUMMY_ID__tmp8;
V_DUMMY_ID__tmp8 = V_DUMMY_ID__tmp7 + V_DUMMY_ID__tmp6;
o0_i = ii_i0_data[0] * V_DUMMY_ID__tmp8 * V_DUMMY_ID__tmp1;
}

ii_o0_data[0] = o0_i;
    }
}
// GpuElemwise{Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))}}[(0, 0)]
// node.op.destroy_map={0: [0]}
//    Input   0 CudaNdarrayType(float32, matrix)
//    Input   1 CudaNdarrayType(float32, (True, True))
//    Input   2 CudaNdarrayType(float32, col)
//    Input   3 CudaNdarrayType(float32, matrix)
//    Input   4 CudaNdarrayType(float32, matrix)
//    Input   5 CudaNdarrayType(float32, (True, True))
//    Input   6 CudaNdarrayType(float32, (True, True))
//    Input   7 CudaNdarrayType(float32, matrix)
//    Input   8 CudaNdarrayType(float32, matrix)
//    Input   9 CudaNdarrayType(float32, matrix)
//    Input   10 CudaNdarrayType(float32, (True, True))
//    Output  0 CudaNdarrayType(float32, matrix)
static __global__ void kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_Ccontiguous (unsigned int numEls
	, const float * i0_data
	, const float * i1_data
	, const float * i2_data
	, const float * i3_data
	, const float * i4_data
	, const float * i5_data
	, const float * i6_data
	, const float * i7_data
	, const float * i8_data
	, const float * i9_data
	, const float * i10_data
	, float * o0_data
	)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const float ii_i1_value = i1_data[0];
    const float ii_i5_value = i5_data[0];
    const float ii_i6_value = i6_data[0];
    const float ii_i10_value = i10_data[0];
    for (int i = idx; i < numEls; i += numThreads) {
npy_float32 o0_i;
        {
npy_float32 V_DUMMY_ID__tmp1;
V_DUMMY_ID__tmp1 = ii_i10_value - i4_data[i];
npy_float32 V_DUMMY_ID__tmp2;
V_DUMMY_ID__tmp2 = ii_i5_value * ii_i6_value * i9_data[i];
npy_float32 V_DUMMY_ID__tmp3;
V_DUMMY_ID__tmp3 = i2_data[i] * i8_data[i] * i4_data[i];
npy_float32 V_DUMMY_ID__tmp4;
V_DUMMY_ID__tmp4 = ii_i5_value * ii_i6_value * i7_data[i];
npy_float32 V_DUMMY_ID__tmp5;
V_DUMMY_ID__tmp5 = ii_i1_value * i2_data[i] * i3_data[i] * i4_data[i];
npy_float32 V_DUMMY_ID__tmp6;
V_DUMMY_ID__tmp6 = V_DUMMY_ID__tmp3 / V_DUMMY_ID__tmp2;
npy_float32 V_DUMMY_ID__tmp7;
V_DUMMY_ID__tmp7 = V_DUMMY_ID__tmp5 / V_DUMMY_ID__tmp4;
npy_float32 V_DUMMY_ID__tmp8;
V_DUMMY_ID__tmp8 = V_DUMMY_ID__tmp7 + V_DUMMY_ID__tmp6;
o0_i = i0_data[i] * V_DUMMY_ID__tmp8 * V_DUMMY_ID__tmp1;
}

o0_data[i] = o0_i;
    }
}

        static void can_collapse_node_8da615e4463fd0663090428e75863d87_0(int nd, const int * dims, const int * strides, int collapse[])
        {
            //can we collapse dims[i] and dims[i-1]
            for(int i=nd-1;i>0;i--){
                if(strides[i]*dims[i]==strides[i-1]){//the dims nd-1 are not strided again dimension nd
                    collapse[i]=1;
                }else collapse[i]=0;
            }
        }
        

        static int callkernel_node_8da615e4463fd0663090428e75863d87_0(unsigned int numEls, const int d,
            const int * dims,
            const float * i0_data, const int * i0_str, const float * i1_data, const int * i1_str, const float * i2_data, const int * i2_str, const float * i3_data, const int * i3_str, const float * i4_data, const int * i4_str, const float * i5_data, const int * i5_str, const float * i6_data, const int * i6_str, const float * i7_data, const int * i7_str, const float * i8_data, const int * i8_str, const float * i9_data, const int * i9_str, const float * i10_data, const int * i10_str,
            float * o0_data, const int * o0_str)
        {
            numEls = dims[0]*dims[1]*1;
        
int local_dims[2];

            int local_str[11][2];
            int local_ostr[1][2];
            

        int nd_collapse = 2;
        for(int i=0;i<2;i++){//init new dim
          local_dims[i]=dims[i];
        }
        

            for(int i=0;i<2;i++){//init new strides
              local_str[0][i]=i0_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[1][i]=i1_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[2][i]=i2_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[3][i]=i3_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[4][i]=i4_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[5][i]=i5_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[6][i]=i6_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[7][i]=i7_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[8][i]=i8_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[9][i]=i9_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_str[10][i]=i10_str[i];
            }
            

            for(int i=0;i<2;i++){//init new strides
              local_ostr[0][i]=o0_str[i];
            }
            

        for(int id=0;id<nd_collapse;id++){

          bool all_broadcast=true;
          for(int input_id=0;input_id<11;input_id++){
            if(local_str[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          for(int input_id=0;input_id<1;input_id++){
            if(local_ostr[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          if(all_broadcast){
            for(int j=id+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
            for(int input_id=0;input_id<11;input_id++){
              for(int j=id+1;j<nd_collapse;j++){//remove dims i from the array
                local_str[input_id][j-1]=local_str[input_id][j];
              }
            }
            for(int output_id=0;output_id<1;output_id++){
              for(int j=id+1;j<nd_collapse;j++){//remove dims i from the array
                local_ostr[output_id][j-1]=local_ostr[output_id][j];
              }
            }
            nd_collapse--; id--;
          }
        }
        
int nd_collapse_[2] = {1,1};

                        int nd_collapse_0[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[0], nd_collapse_0);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_0[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_2[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[2], nd_collapse_2);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_2[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_3[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[3], nd_collapse_3);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_3[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_4[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[4], nd_collapse_4);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_4[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_7[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[7], nd_collapse_7);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_7[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_8[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[8], nd_collapse_8);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_8[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_9[2] = {1,1};

can_collapse_node_8da615e4463fd0663090428e75863d87_0(nd_collapse, local_dims, local_str[9], nd_collapse_9);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_9[i]==0)
nd_collapse_[i]=0;
}
                

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[0][i-1]=local_str[0][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[0][j-1]=local_str[0][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[1][i-1]=local_str[1][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[1][j-1]=local_str[1][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[2][i-1]=local_str[2][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[2][j-1]=local_str[2][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[3][i-1]=local_str[3][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[3][j-1]=local_str[3][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[4][i-1]=local_str[4][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[4][j-1]=local_str[4][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[5][i-1]=local_str[5][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[5][j-1]=local_str[5][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[6][i-1]=local_str[6][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[6][j-1]=local_str[6][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[7][i-1]=local_str[7][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[7][j-1]=local_str[7][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[8][i-1]=local_str[8][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[8][j-1]=local_str[8][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[9][i-1]=local_str[9][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[9][j-1]=local_str[9][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_str[10][i-1]=local_str[10][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_str[10][j-1]=local_str[10][j];
                }
            }
            

            for(int i=nd_collapse-1;i>0;i--){
              if(nd_collapse_[i]==1){
                local_ostr[0][i-1]=local_ostr[0][i];//set new strides
                for(int j=i+1;j<nd_collapse;j++)//remove stride i from the array
                  local_ostr[0][j-1]=local_ostr[0][j];
                }
            }
            

        for(int i=nd_collapse-1;i>0;i--){
          if(nd_collapse_[i]==1){
            local_dims[i-1]*=local_dims[i];//set new dims
            for(int j=i+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
          }
        }
        

        for(int i=1, end=nd_collapse;i<end;i++){
          if(nd_collapse_[i]==1)nd_collapse--;
        }
        if(nd_collapse == 1 
 &&  local_str[0][nd_collapse-1]==1  && local_str[2][nd_collapse-1]==1  && local_str[3][nd_collapse-1]==1  && local_str[4][nd_collapse-1]==1  && local_str[7][nd_collapse-1]==1  && local_str[8][nd_collapse-1]==1  && local_str[9][nd_collapse-1]==1  && local_ostr[0][nd_collapse-1]==1 
){nd_collapse=0;} 
if(numEls==0) return 0;
switch (nd_collapse==0?0:min(2,nd_collapse)) {
case 0: {

                //first use at least a full warp
                int threads_per_block = std::min(numEls,  (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls % threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, i0_data, i1_data, i2_data, i3_data, i4_data, i5_data, i6_data, i7_data, i8_data, i9_data, i10_data, o0_data);

                //std::cerr << "calling callkernel returned\n";
                

                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n    n_blocks=%i threads_per_block=%i\n   Call: %s\n",
                         "GpuElemwise node_8da615e4463fd0663090428e75863d87_0 Composite", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, i0_data, i1_data, i2_data, i3_data, i4_data, i5_data, i6_data, i7_data, i8_data, i9_data, i10_data, o0_data)");
                    return -1;

                }
                
                return 0;
                
        } break;
case 1: {

                //first use at least a full warp
                int threads_per_block = std::min(numEls, (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls % threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);

                kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_1<<<n_blocks, threads_per_block>>>(numEls, local_dims[0], i0_data, local_str[0][0], i1_data, local_str[1][0], i2_data, local_str[2][0], i3_data, local_str[3][0], i4_data, local_str[4][0], i5_data, local_str[5][0], i6_data, local_str[6][0], i7_data, local_str[7][0], i8_data, local_str[8][0], i9_data, local_str[9][0], i10_data, local_str[10][0], o0_data, local_ostr[0][0]);
                

                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n    n_blocks=%i threads_per_block=%i\n   Call: %s\n",
                         "GpuElemwise node_8da615e4463fd0663090428e75863d87_0 Composite", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, local_dims[0], i0_data, local_str[0][0], i1_data, local_str[1][0], i2_data, local_str[2][0], i3_data, local_str[3][0], i4_data, local_str[4][0], i5_data, local_str[5][0], i6_data, local_str[6][0], i7_data, local_str[7][0], i8_data, local_str[8][0], i9_data, local_str[9][0], i10_data, local_str[10][0], o0_data, local_ostr[0][0])");
                    return -1;

                }
                return 0;
                
        } break;
case 2: {

                //first use at least a full warp
                int threads_per_block = std::min(numEls, (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls % threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);

                kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_2<<<n_blocks, threads_per_block>>>(numEls, local_dims[0], local_dims[1], i0_data, local_str[0][0], local_str[0][1], i1_data, local_str[1][0], local_str[1][1], i2_data, local_str[2][0], local_str[2][1], i3_data, local_str[3][0], local_str[3][1], i4_data, local_str[4][0], local_str[4][1], i5_data, local_str[5][0], local_str[5][1], i6_data, local_str[6][0], local_str[6][1], i7_data, local_str[7][0], local_str[7][1], i8_data, local_str[8][0], local_str[8][1], i9_data, local_str[9][0], local_str[9][1], i10_data, local_str[10][0], local_str[10][1], o0_data, local_ostr[0][0], local_ostr[0][1]);
                

                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n    n_blocks=%i threads_per_block=%i\n   Call: %s\n",
                         "GpuElemwise node_8da615e4463fd0663090428e75863d87_0 Composite", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_Composite_node_8da615e4463fd0663090428e75863d87_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, local_dims[0], local_dims[1], i0_data, local_str[0][0], local_str[0][1], i1_data, local_str[1][0], local_str[1][1], i2_data, local_str[2][0], local_str[2][1], i3_data, local_str[3][0], local_str[3][1], i4_data, local_str[4][0], local_str[4][1], i5_data, local_str[5][0], local_str[5][1], i6_data, local_str[6][0], local_str[6][1], i7_data, local_str[7][0], local_str[7][1], i8_data, local_str[8][0], local_str[8][1], i9_data, local_str[9][0], local_str[9][1], i10_data, local_str[10][0], local_str[10][1], o0_data, local_ostr[0][0], local_ostr[0][1])");
                    return -1;

                }
                return 0;
                
        } break;
}
return -2;
}


    namespace {
    struct __struct_compiled_op_8da615e4463fd0663090428e75863d87 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V9;
PyObject* storage_V11;
PyObject* storage_V13;
PyObject* storage_V15;
PyObject* storage_V17;
PyObject* storage_V19;
PyObject* storage_V21;
PyObject* storage_V23;
PyObject* storage_V1;
        

        __struct_compiled_op_8da615e4463fd0663090428e75863d87() {
            // This is only somewhat safe because we:
            //  1) Are not a virtual class
            //  2) Do not use any virtual classes in the members
            //  3) Deal with mostly POD and pointers

            // If this changes, we would have to revise this, but for
            // now I am tired of chasing segfaults because
            // initialization code had an error and some pointer has
            // a junk value.
            memset(this, 0, sizeof(*this));
        }
        ~__struct_compiled_op_8da615e4463fd0663090428e75863d87(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V5, PyObject* storage_V7, PyObject* storage_V9, PyObject* storage_V11, PyObject* storage_V13, PyObject* storage_V15, PyObject* storage_V17, PyObject* storage_V19, PyObject* storage_V21, PyObject* storage_V23, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V5);
Py_XINCREF(storage_V7);
Py_XINCREF(storage_V9);
Py_XINCREF(storage_V11);
Py_XINCREF(storage_V13);
Py_XINCREF(storage_V15);
Py_XINCREF(storage_V17);
Py_XINCREF(storage_V19);
Py_XINCREF(storage_V21);
Py_XINCREF(storage_V23);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V5 = storage_V5;
this->storage_V7 = storage_V7;
this->storage_V9 = storage_V9;
this->storage_V11 = storage_V11;
this->storage_V13 = storage_V13;
this->storage_V15 = storage_V15;
this->storage_V17 = storage_V17;
this->storage_V19 = storage_V19;
this->storage_V21 = storage_V21;
this->storage_V23 = storage_V23;
this->storage_V1 = storage_V1;
            













            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

double __DUMMY_5;
__label_7:

double __DUMMY_7;
__label_9:

double __DUMMY_9;
__label_11:

double __DUMMY_11;
__label_13:

double __DUMMY_13;
__label_15:

double __DUMMY_15;
__label_17:

double __DUMMY_17;
__label_19:

double __DUMMY_19;
__label_21:

double __DUMMY_21;
__label_23:

double __DUMMY_23;
__label_26:

double __DUMMY_26;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V7);
Py_XDECREF(this->storage_V9);
Py_XDECREF(this->storage_V11);
Py_XDECREF(this->storage_V13);
Py_XDECREF(this->storage_V15);
Py_XDECREF(this->storage_V17);
Py_XDECREF(this->storage_V19);
Py_XDECREF(this->storage_V21);
Py_XDECREF(this->storage_V23);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
     CudaNdarray * V1;
    PyObject* py_V3;
     CudaNdarray * V3;
    PyObject* py_V5;
     CudaNdarray * V5;
    PyObject* py_V7;
     CudaNdarray * V7;
    PyObject* py_V9;
     CudaNdarray * V9;
    PyObject* py_V11;
     CudaNdarray * V11;
    PyObject* py_V13;
     CudaNdarray * V13;
    PyObject* py_V15;
     CudaNdarray * V15;
    PyObject* py_V17;
     CudaNdarray * V17;
    PyObject* py_V19;
     CudaNdarray * V19;
    PyObject* py_V21;
     CudaNdarray * V21;
    PyObject* py_V23;
     CudaNdarray * V23;
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            V1 = NULL;
        }
        else
        {
            
        assert(py_V1->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V1))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V1, (py_V1->ob_refcnt));
            V1 = (CudaNdarray*)py_V1;
            //std::cerr << "c_extract " << V1 << '\n';
        

                if (V1->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V1->nd);
                    V1 = NULL;
                    {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;};
                }
                //std::cerr << "c_extract " << V1 << " nd check passed\n";
            

                assert(V1);
                Py_INCREF(py_V1);
            }
            else if (py_V1 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V1 = NULL;
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V1, (py_V1->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V1 = NULL;
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;};
            }
            //std::cerr << "c_extract done " << V1 << '\n';
            

        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
        assert(py_V3->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V3))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V3, (py_V3->ob_refcnt));
            V3 = (CudaNdarray*)py_V3;
            //std::cerr << "c_extract " << V3 << '\n';
        

                if (V3->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V3->nd);
                    V3 = NULL;
                    {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;};
                }
                //std::cerr << "c_extract " << V3 << " nd check passed\n";
            

                assert(V3);
                Py_INCREF(py_V3);
            }
            else if (py_V3 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V3 = NULL;
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V3, (py_V3->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V3 = NULL;
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;};
            }
            //std::cerr << "c_extract done " << V3 << '\n';
            

{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        assert(py_V5->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V5))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V5, (py_V5->ob_refcnt));
            V5 = (CudaNdarray*)py_V5;
            //std::cerr << "c_extract " << V5 << '\n';
        

                if (V5->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V5->nd);
                    V5 = NULL;
                    {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
                }
                //std::cerr << "c_extract " << V5 << " nd check passed\n";
            

                if (CudaNdarray_HOST_DIMS(V5)[0] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V5)[0], 0);
                    V5 = NULL;
                    {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
                }
                //std::cerr << "c_extract " << V5 << "dim check 0 passed\n";
                //std::cerr << "c_extract " << V5 << "checking bcast 0 <" << V5->str<< ">\n";
                //std::cerr << "c_extract " << V5->str[0] << "\n";
                if (CudaNdarray_HOST_STRIDES(V5)[0])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V5)[0], 0);
                    V5 = NULL;
                    {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
                }
                //std::cerr << "c_extract " << V5 << "bcast check 0 passed\n";
                    

                if (CudaNdarray_HOST_DIMS(V5)[1] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V5)[1], 1);
                    V5 = NULL;
                    {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
                }
                //std::cerr << "c_extract " << V5 << "dim check 1 passed\n";
                //std::cerr << "c_extract " << V5 << "checking bcast 1 <" << V5->str<< ">\n";
                //std::cerr << "c_extract " << V5->str[1] << "\n";
                if (CudaNdarray_HOST_STRIDES(V5)[1])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V5)[1], 1);
                    V5 = NULL;
                    {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
                }
                //std::cerr << "c_extract " << V5 << "bcast check 1 passed\n";
                    

                assert(V5);
                Py_INCREF(py_V5);
            }
            else if (py_V5 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V5 = NULL;
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V5, (py_V5->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V5 = NULL;
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;};
            }
            //std::cerr << "c_extract done " << V5 << '\n';
            

{

    py_V7 = PyList_GET_ITEM(storage_V7, 0);
    {Py_XINCREF(py_V7);}
    
        assert(py_V7->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V7))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V7, (py_V7->ob_refcnt));
            V7 = (CudaNdarray*)py_V7;
            //std::cerr << "c_extract " << V7 << '\n';
        

                if (V7->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V7->nd);
                    V7 = NULL;
                    {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;};
                }
                //std::cerr << "c_extract " << V7 << " nd check passed\n";
            

                if (CudaNdarray_HOST_DIMS(V7)[1] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V7)[1], 1);
                    V7 = NULL;
                    {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;};
                }
                //std::cerr << "c_extract " << V7 << "dim check 1 passed\n";
                //std::cerr << "c_extract " << V7 << "checking bcast 1 <" << V7->str<< ">\n";
                //std::cerr << "c_extract " << V7->str[1] << "\n";
                if (CudaNdarray_HOST_STRIDES(V7)[1])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V7)[1], 1);
                    V7 = NULL;
                    {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;};
                }
                //std::cerr << "c_extract " << V7 << "bcast check 1 passed\n";
                    

                assert(V7);
                Py_INCREF(py_V7);
            }
            else if (py_V7 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V7 = NULL;
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V7, (py_V7->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V7 = NULL;
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;};
            }
            //std::cerr << "c_extract done " << V7 << '\n';
            

{

    py_V9 = PyList_GET_ITEM(storage_V9, 0);
    {Py_XINCREF(py_V9);}
    
        assert(py_V9->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V9))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V9, (py_V9->ob_refcnt));
            V9 = (CudaNdarray*)py_V9;
            //std::cerr << "c_extract " << V9 << '\n';
        

                if (V9->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V9->nd);
                    V9 = NULL;
                    {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;};
                }
                //std::cerr << "c_extract " << V9 << " nd check passed\n";
            

                assert(V9);
                Py_INCREF(py_V9);
            }
            else if (py_V9 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V9 = NULL;
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V9, (py_V9->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V9 = NULL;
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;};
            }
            //std::cerr << "c_extract done " << V9 << '\n';
            

{

    py_V11 = PyList_GET_ITEM(storage_V11, 0);
    {Py_XINCREF(py_V11);}
    
        assert(py_V11->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V11))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V11, (py_V11->ob_refcnt));
            V11 = (CudaNdarray*)py_V11;
            //std::cerr << "c_extract " << V11 << '\n';
        

                if (V11->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V11->nd);
                    V11 = NULL;
                    {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;};
                }
                //std::cerr << "c_extract " << V11 << " nd check passed\n";
            

                assert(V11);
                Py_INCREF(py_V11);
            }
            else if (py_V11 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V11 = NULL;
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V11, (py_V11->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V11 = NULL;
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;};
            }
            //std::cerr << "c_extract done " << V11 << '\n';
            

{

    py_V13 = PyList_GET_ITEM(storage_V13, 0);
    {Py_XINCREF(py_V13);}
    
        assert(py_V13->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V13))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V13, (py_V13->ob_refcnt));
            V13 = (CudaNdarray*)py_V13;
            //std::cerr << "c_extract " << V13 << '\n';
        

                if (V13->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V13->nd);
                    V13 = NULL;
                    {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
                }
                //std::cerr << "c_extract " << V13 << " nd check passed\n";
            

                if (CudaNdarray_HOST_DIMS(V13)[0] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V13)[0], 0);
                    V13 = NULL;
                    {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
                }
                //std::cerr << "c_extract " << V13 << "dim check 0 passed\n";
                //std::cerr << "c_extract " << V13 << "checking bcast 0 <" << V13->str<< ">\n";
                //std::cerr << "c_extract " << V13->str[0] << "\n";
                if (CudaNdarray_HOST_STRIDES(V13)[0])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V13)[0], 0);
                    V13 = NULL;
                    {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
                }
                //std::cerr << "c_extract " << V13 << "bcast check 0 passed\n";
                    

                if (CudaNdarray_HOST_DIMS(V13)[1] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V13)[1], 1);
                    V13 = NULL;
                    {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
                }
                //std::cerr << "c_extract " << V13 << "dim check 1 passed\n";
                //std::cerr << "c_extract " << V13 << "checking bcast 1 <" << V13->str<< ">\n";
                //std::cerr << "c_extract " << V13->str[1] << "\n";
                if (CudaNdarray_HOST_STRIDES(V13)[1])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V13)[1], 1);
                    V13 = NULL;
                    {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
                }
                //std::cerr << "c_extract " << V13 << "bcast check 1 passed\n";
                    

                assert(V13);
                Py_INCREF(py_V13);
            }
            else if (py_V13 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V13 = NULL;
                {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V13, (py_V13->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V13 = NULL;
                {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_14;};
            }
            //std::cerr << "c_extract done " << V13 << '\n';
            

{

    py_V15 = PyList_GET_ITEM(storage_V15, 0);
    {Py_XINCREF(py_V15);}
    
        assert(py_V15->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V15))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V15, (py_V15->ob_refcnt));
            V15 = (CudaNdarray*)py_V15;
            //std::cerr << "c_extract " << V15 << '\n';
        

                if (V15->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V15->nd);
                    V15 = NULL;
                    {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
                }
                //std::cerr << "c_extract " << V15 << " nd check passed\n";
            

                if (CudaNdarray_HOST_DIMS(V15)[0] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V15)[0], 0);
                    V15 = NULL;
                    {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
                }
                //std::cerr << "c_extract " << V15 << "dim check 0 passed\n";
                //std::cerr << "c_extract " << V15 << "checking bcast 0 <" << V15->str<< ">\n";
                //std::cerr << "c_extract " << V15->str[0] << "\n";
                if (CudaNdarray_HOST_STRIDES(V15)[0])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V15)[0], 0);
                    V15 = NULL;
                    {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
                }
                //std::cerr << "c_extract " << V15 << "bcast check 0 passed\n";
                    

                if (CudaNdarray_HOST_DIMS(V15)[1] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V15)[1], 1);
                    V15 = NULL;
                    {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
                }
                //std::cerr << "c_extract " << V15 << "dim check 1 passed\n";
                //std::cerr << "c_extract " << V15 << "checking bcast 1 <" << V15->str<< ">\n";
                //std::cerr << "c_extract " << V15->str[1] << "\n";
                if (CudaNdarray_HOST_STRIDES(V15)[1])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V15)[1], 1);
                    V15 = NULL;
                    {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
                }
                //std::cerr << "c_extract " << V15 << "bcast check 1 passed\n";
                    

                assert(V15);
                Py_INCREF(py_V15);
            }
            else if (py_V15 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V15 = NULL;
                {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V15, (py_V15->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V15 = NULL;
                {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_16;};
            }
            //std::cerr << "c_extract done " << V15 << '\n';
            

{

    py_V17 = PyList_GET_ITEM(storage_V17, 0);
    {Py_XINCREF(py_V17);}
    
        assert(py_V17->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V17))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V17, (py_V17->ob_refcnt));
            V17 = (CudaNdarray*)py_V17;
            //std::cerr << "c_extract " << V17 << '\n';
        

                if (V17->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V17->nd);
                    V17 = NULL;
                    {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_18;};
                }
                //std::cerr << "c_extract " << V17 << " nd check passed\n";
            

                assert(V17);
                Py_INCREF(py_V17);
            }
            else if (py_V17 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V17 = NULL;
                {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_18;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V17, (py_V17->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V17 = NULL;
                {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_18;};
            }
            //std::cerr << "c_extract done " << V17 << '\n';
            

{

    py_V19 = PyList_GET_ITEM(storage_V19, 0);
    {Py_XINCREF(py_V19);}
    
        assert(py_V19->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V19))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V19, (py_V19->ob_refcnt));
            V19 = (CudaNdarray*)py_V19;
            //std::cerr << "c_extract " << V19 << '\n';
        

                if (V19->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V19->nd);
                    V19 = NULL;
                    {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_20;};
                }
                //std::cerr << "c_extract " << V19 << " nd check passed\n";
            

                assert(V19);
                Py_INCREF(py_V19);
            }
            else if (py_V19 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V19 = NULL;
                {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_20;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V19, (py_V19->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V19 = NULL;
                {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_20;};
            }
            //std::cerr << "c_extract done " << V19 << '\n';
            

{

    py_V21 = PyList_GET_ITEM(storage_V21, 0);
    {Py_XINCREF(py_V21);}
    
        assert(py_V21->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V21))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V21, (py_V21->ob_refcnt));
            V21 = (CudaNdarray*)py_V21;
            //std::cerr << "c_extract " << V21 << '\n';
        

                if (V21->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V21->nd);
                    V21 = NULL;
                    {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_22;};
                }
                //std::cerr << "c_extract " << V21 << " nd check passed\n";
            

                assert(V21);
                Py_INCREF(py_V21);
            }
            else if (py_V21 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V21 = NULL;
                {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_22;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V21, (py_V21->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V21 = NULL;
                {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_22;};
            }
            //std::cerr << "c_extract done " << V21 << '\n';
            

{

    py_V23 = PyList_GET_ITEM(storage_V23, 0);
    {Py_XINCREF(py_V23);}
    
        assert(py_V23->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V23))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V23, (py_V23->ob_refcnt));
            V23 = (CudaNdarray*)py_V23;
            //std::cerr << "c_extract " << V23 << '\n';
        

                if (V23->nd != 2)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 2",
                                 V23->nd);
                    V23 = NULL;
                    {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
                }
                //std::cerr << "c_extract " << V23 << " nd check passed\n";
            

                if (CudaNdarray_HOST_DIMS(V23)[0] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V23)[0], 0);
                    V23 = NULL;
                    {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
                }
                //std::cerr << "c_extract " << V23 << "dim check 0 passed\n";
                //std::cerr << "c_extract " << V23 << "checking bcast 0 <" << V23->str<< ">\n";
                //std::cerr << "c_extract " << V23->str[0] << "\n";
                if (CudaNdarray_HOST_STRIDES(V23)[0])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V23)[0], 0);
                    V23 = NULL;
                    {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
                }
                //std::cerr << "c_extract " << V23 << "bcast check 0 passed\n";
                    

                if (CudaNdarray_HOST_DIMS(V23)[1] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %i on broadcastable dimension %i",
                                 CudaNdarray_HOST_DIMS(V23)[1], 1);
                    V23 = NULL;
                    {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
                }
                //std::cerr << "c_extract " << V23 << "dim check 1 passed\n";
                //std::cerr << "c_extract " << V23 << "checking bcast 1 <" << V23->str<< ">\n";
                //std::cerr << "c_extract " << V23->str[1] << "\n";
                if (CudaNdarray_HOST_STRIDES(V23)[1])
                {
                    //std::cerr << "c_extract bad stride detected...\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %i on a broadcastable dimension %i",
                                 CudaNdarray_HOST_STRIDES(V23)[1], 1);
                    V23 = NULL;
                    {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
                }
                //std::cerr << "c_extract " << V23 << "bcast check 1 passed\n";
                    

                assert(V23);
                Py_INCREF(py_V23);
            }
            else if (py_V23 == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                V23 = NULL;
                {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %p %i\n", py_V23, (py_V23->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                V23 = NULL;
                {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_24;};
            }
            //std::cerr << "c_extract done " << V23 << '\n';
            

{
// Op class GpuElemwise

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} START\n";
        //standard elemwise size checks
            

            int dims[2] = {1,1};
            

                int broadcasts_V3[2] = {0, 0};
                

                int broadcasts_V5[2] = {1, 1};
                

                int broadcasts_V7[2] = {0, 1};
                

                int broadcasts_V9[2] = {0, 0};
                

                int broadcasts_V11[2] = {0, 0};
                

                int broadcasts_V13[2] = {1, 1};
                

                int broadcasts_V15[2] = {1, 1};
                

                int broadcasts_V17[2] = {0, 0};
                

                int broadcasts_V19[2] = {0, 0};
                

                int broadcasts_V21[2] = {0, 0};
                

                int broadcasts_V23[2] = {1, 1};
                

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V3\n";
        if (2 != V3->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V3->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V3)[i] : dims[i];
            if ((!(broadcasts_V3[i] &&
                 CudaNdarray_HOST_DIMS(V3)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V3)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V3 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 0 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V3)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V5\n";
        if (2 != V5->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V5->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V5)[i] : dims[i];
            if ((!(broadcasts_V5[i] &&
                 CudaNdarray_HOST_DIMS(V5)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V5)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V5 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 1 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V5)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V7\n";
        if (2 != V7->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V7->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V7)[i] : dims[i];
            if ((!(broadcasts_V7[i] &&
                 CudaNdarray_HOST_DIMS(V7)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V7)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V7 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 2 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V7)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V9\n";
        if (2 != V9->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V9->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V9)[i] : dims[i];
            if ((!(broadcasts_V9[i] &&
                 CudaNdarray_HOST_DIMS(V9)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V9)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V9 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 3 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V9)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V11\n";
        if (2 != V11->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V11->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V11)[i] : dims[i];
            if ((!(broadcasts_V11[i] &&
                 CudaNdarray_HOST_DIMS(V11)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V11)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V11 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 4 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V11)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V13\n";
        if (2 != V13->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V13->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V13)[i] : dims[i];
            if ((!(broadcasts_V13[i] &&
                 CudaNdarray_HOST_DIMS(V13)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V13)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V13 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 5 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V13)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V15\n";
        if (2 != V15->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V15->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V15)[i] : dims[i];
            if ((!(broadcasts_V15[i] &&
                 CudaNdarray_HOST_DIMS(V15)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V15)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V15 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 6 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V15)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V17\n";
        if (2 != V17->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V17->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V17)[i] : dims[i];
            if ((!(broadcasts_V17[i] &&
                 CudaNdarray_HOST_DIMS(V17)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V17)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V17 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 7 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V17)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V19\n";
        if (2 != V19->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V19->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V19)[i] : dims[i];
            if ((!(broadcasts_V19[i] &&
                 CudaNdarray_HOST_DIMS(V19)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V19)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V19 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 8 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V19)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V21\n";
        if (2 != V21->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V21->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V21)[i] : dims[i];
            if ((!(broadcasts_V21[i] &&
                 CudaNdarray_HOST_DIMS(V21)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V21)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V21 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 9 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V21)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V23\n";
        if (2 != V23->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 2 dims, not %i", V23->nd);
            {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
        }
        for (int i = 0; i< 2; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V23)[i] : dims[i];
            if ((!(broadcasts_V23[i] &&
                 CudaNdarray_HOST_DIMS(V23)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V23)[i]))
            {
                //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} checking input V23 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 10 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V23)[i],
                             dims[i]
                            );
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
            

        Py_XDECREF(V1);
        V1 = V3;
        Py_INCREF(V1);
        for (int i = 0; (i< 2) && (V1); ++i) {
            if (dims[i] != CudaNdarray_HOST_DIMS(V1)[i])
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Output dimension mis-match. Output"
                             " 0 (indices start at 0), working inplace"
                             " on input 0, has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V1)[i],
                             dims[i]
                            );
                Py_DECREF(V1);
                V1 = NULL;
                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
        }
        //std::cerr << "ELEMWISE NEW V1 nd" << V1->nd << "\n";
        //std::cerr << "ELEMWISE NEW V1 data" << V1->devdata << "\n";
        

        {
            //new block so that failure gotos don't skip over variable initialization
            //std::cerr << "calling callkernel\n";
            if (callkernel_node_8da615e4463fd0663090428e75863d87_0(1, 0, dims
            

                        , CudaNdarray_DEV_DATA(V3), CudaNdarray_HOST_STRIDES(V3)
            

                        , CudaNdarray_DEV_DATA(V5), CudaNdarray_HOST_STRIDES(V5)
            

                        , CudaNdarray_DEV_DATA(V7), CudaNdarray_HOST_STRIDES(V7)
            

                        , CudaNdarray_DEV_DATA(V9), CudaNdarray_HOST_STRIDES(V9)
            

                        , CudaNdarray_DEV_DATA(V11), CudaNdarray_HOST_STRIDES(V11)
            

                        , CudaNdarray_DEV_DATA(V13), CudaNdarray_HOST_STRIDES(V13)
            

                        , CudaNdarray_DEV_DATA(V15), CudaNdarray_HOST_STRIDES(V15)
            

                        , CudaNdarray_DEV_DATA(V17), CudaNdarray_HOST_STRIDES(V17)
            

                        , CudaNdarray_DEV_DATA(V19), CudaNdarray_HOST_STRIDES(V19)
            

                        , CudaNdarray_DEV_DATA(V21), CudaNdarray_HOST_STRIDES(V21)
            

                        , CudaNdarray_DEV_DATA(V23), CudaNdarray_HOST_STRIDES(V23)
            

                        , CudaNdarray_DEV_DATA(V1), CudaNdarray_HOST_STRIDES(V1)
            

                        ))
            {
                 // error
            

                Py_DECREF(V1);
                V1 = NULL;
                

                {
        __failure = 25;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_25;};
            }
            else // no error
            {
            }
        }
        //std::cerr << "C_CODE Composite{(i0 * (((i1 * i2 * i3 * i4) / (i5 * i6 * i7)) + ((i2 * i8 * i4) / (i5 * i6 * i9))) * (i10 - i4))} END\n";
        
__label_25:

double __DUMMY_25;

}
__label_24:

        //std::cerr << "cleanup " << py_V23 << " " << V23 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V23, (py_V23->ob_refcnt));
        if (V23)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V23, (V23->ob_refcnt));
            Py_XDECREF(V23);
        }
        //std::cerr << "cleanup done" << py_V23 << "\n";
        
    {Py_XDECREF(py_V23);}
    
double __DUMMY_24;

}
__label_22:

        //std::cerr << "cleanup " << py_V21 << " " << V21 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V21, (py_V21->ob_refcnt));
        if (V21)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V21, (V21->ob_refcnt));
            Py_XDECREF(V21);
        }
        //std::cerr << "cleanup done" << py_V21 << "\n";
        
    {Py_XDECREF(py_V21);}
    
double __DUMMY_22;

}
__label_20:

        //std::cerr << "cleanup " << py_V19 << " " << V19 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V19, (py_V19->ob_refcnt));
        if (V19)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V19, (V19->ob_refcnt));
            Py_XDECREF(V19);
        }
        //std::cerr << "cleanup done" << py_V19 << "\n";
        
    {Py_XDECREF(py_V19);}
    
double __DUMMY_20;

}
__label_18:

        //std::cerr << "cleanup " << py_V17 << " " << V17 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V17, (py_V17->ob_refcnt));
        if (V17)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V17, (V17->ob_refcnt));
            Py_XDECREF(V17);
        }
        //std::cerr << "cleanup done" << py_V17 << "\n";
        
    {Py_XDECREF(py_V17);}
    
double __DUMMY_18;

}
__label_16:

        //std::cerr << "cleanup " << py_V15 << " " << V15 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V15, (py_V15->ob_refcnt));
        if (V15)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V15, (V15->ob_refcnt));
            Py_XDECREF(V15);
        }
        //std::cerr << "cleanup done" << py_V15 << "\n";
        
    {Py_XDECREF(py_V15);}
    
double __DUMMY_16;

}
__label_14:

        //std::cerr << "cleanup " << py_V13 << " " << V13 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V13, (py_V13->ob_refcnt));
        if (V13)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V13, (V13->ob_refcnt));
            Py_XDECREF(V13);
        }
        //std::cerr << "cleanup done" << py_V13 << "\n";
        
    {Py_XDECREF(py_V13);}
    
double __DUMMY_14;

}
__label_12:

        //std::cerr << "cleanup " << py_V11 << " " << V11 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V11, (py_V11->ob_refcnt));
        if (V11)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V11, (V11->ob_refcnt));
            Py_XDECREF(V11);
        }
        //std::cerr << "cleanup done" << py_V11 << "\n";
        
    {Py_XDECREF(py_V11);}
    
double __DUMMY_12;

}
__label_10:

        //std::cerr << "cleanup " << py_V9 << " " << V9 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V9, (py_V9->ob_refcnt));
        if (V9)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V9, (V9->ob_refcnt));
            Py_XDECREF(V9);
        }
        //std::cerr << "cleanup done" << py_V9 << "\n";
        
    {Py_XDECREF(py_V9);}
    
double __DUMMY_10;

}
__label_8:

        //std::cerr << "cleanup " << py_V7 << " " << V7 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V7, (py_V7->ob_refcnt));
        if (V7)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V7, (V7->ob_refcnt));
            Py_XDECREF(V7);
        }
        //std::cerr << "cleanup done" << py_V7 << "\n";
        
    {Py_XDECREF(py_V7);}
    
double __DUMMY_8;

}
__label_6:

        //std::cerr << "cleanup " << py_V5 << " " << V5 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V5, (py_V5->ob_refcnt));
        if (V5)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V5, (V5->ob_refcnt));
            Py_XDECREF(V5);
        }
        //std::cerr << "cleanup done" << py_V5 << "\n";
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

        //std::cerr << "cleanup " << py_V3 << " " << V3 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V3, (py_V3->ob_refcnt));
        if (V3)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V3, (V3->ob_refcnt));
            Py_XDECREF(V3);
        }
        //std::cerr << "cleanup done" << py_V3 << "\n";
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        //std::cerr << "sync\n";
        if (NULL == V1) {
            // failure: sync None to storage
            Py_XDECREF(py_V1);
            py_V1 = Py_None;
            Py_INCREF(py_V1);
        }
        else
        {
            if (py_V1 != (PyObject*)V1)
            {
                Py_XDECREF(py_V1);
                py_V1 = (PyObject*)V1;
                Py_INCREF(py_V1);
            }
            assert(py_V1->ob_refcnt);
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        //std::cerr << "cleanup " << py_V1 << " " << V1 << "\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %p %i\n", py_V1, (py_V1->ob_refcnt));
        if (V1)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %p %i\n", V1, (V1->ob_refcnt));
            Py_XDECREF(V1);
        }
        //std::cerr << "cleanup done" << py_V1 << "\n";
        
    {Py_XDECREF(py_V1);}
    
double __DUMMY_2;

}

            
        if (__failure) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return __failure;
        
        }
    };
    }
    

        static int __struct_compiled_op_8da615e4463fd0663090428e75863d87_executor(__struct_compiled_op_8da615e4463fd0663090428e75863d87* self) {
            return self->run();
        }

        static void __struct_compiled_op_8da615e4463fd0663090428e75863d87_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_8da615e4463fd0663090428e75863d87*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (13 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 13, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_8da615e4463fd0663090428e75863d87* struct_ptr = new __struct_compiled_op_8da615e4463fd0663090428e75863d87();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4),PyTuple_GET_ITEM(argtuple, 5),PyTuple_GET_ITEM(argtuple, 6),PyTuple_GET_ITEM(argtuple, 7),PyTuple_GET_ITEM(argtuple, 8),PyTuple_GET_ITEM(argtuple, 9),PyTuple_GET_ITEM(argtuple, 10),PyTuple_GET_ITEM(argtuple, 11),PyTuple_GET_ITEM(argtuple, 12) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_8da615e4463fd0663090428e75863d87_executor), struct_ptr, __struct_compiled_op_8da615e4463fd0663090428e75863d87_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init8da615e4463fd0663090428e75863d87(void){
   (void) Py_InitModule("8da615e4463fd0663090428e75863d87", MyMethods);
}
