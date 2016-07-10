#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include "cuda_ndarray.cuh"
//////////////////////
////  Support Code
//////////////////////


#define INTDIV_POW2(a, b) (a >> b)
#define INTMOD_POW2(a, b) (a & ((1<<b)-1))
        // GpuElemwise{Composite{(((-i0) / i1) * i2)}}[(0, 0)]
// node.op.destroy_map={0: [0]}
//    Input   0 CudaNdarrayType(float32, vector)
//    Input   1 CudaNdarrayType(float32, (True,))
//    Input   2 CudaNdarrayType(float32, vector)
//    Output  0 CudaNdarrayType(float32, vector)
static __global__ void kernel_Composite_node_3d6ea06e14a2eac7e20cc661f60176f0_0_1(unsigned int numEls
	, const int dim0
	, const float * i0_data, int i0_str_0
	, const float * i1_data, int i1_str_0
	, const float * i2_data, int i2_str_0
	, float * o0_data, int o0_str_0
	)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const float ii_i1_value = i1_data[0];
    for (int i = idx; i < numEls; i += numThreads) {
        int ii = i;
        const float * ii_i0_data = i0_data;
        const float * ii_i2_data = i2_data;
        float * ii_o0_data = o0_data;
        int pos0 = ii;
        ii_i0_data += pos0 * i0_str_0;
        ii_i2_data += pos0 * i2_str_0;
        ii_o0_data += pos0 * o0_str_0;
npy_float32 o0_i;
        {
npy_float32 V_DUMMY_ID__tmp1;
V_DUMMY_ID__tmp1 = -ii_i0_data[0];
npy_float32 V_DUMMY_ID__tmp2;
V_DUMMY_ID__tmp2 = V_DUMMY_ID__tmp1 / ii_i1_value;
o0_i = V_DUMMY_ID__tmp2 * ii_i2_data[0];
}

ii_o0_data[0] = o0_i;
    }
}
// GpuElemwise{Composite{(((-i0) / i1) * i2)}}[(0, 0)]
// node.op.destroy_map={0: [0]}
//    Input   0 CudaNdarrayType(float32, vector)
//    Input   1 CudaNdarrayType(float32, (True,))
//    Input   2 CudaNdarrayType(float32, vector)
//    Output  0 CudaNdarrayType(float32, vector)
static __global__ void kernel_Composite_node_3d6ea06e14a2eac7e20cc661f60176f0_0_Ccontiguous (unsigned int numEls
	, const float * i0_data
	, const float * i1_data
	, const float * i2_data
	, float * o0_data
	)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    const float ii_i1_value = i1_data[0];
    for (int i = idx; i < numEls; i += numThreads) {
npy_float32 o0_i;
        {
npy_float32 V_DUMMY_ID__tmp1;
V_DUMMY_ID__tmp1 = -i0_data[i];
npy_float32 V_DUMMY_ID__tmp2;
V_DUMMY_ID__tmp2 = V_DUMMY_ID__tmp1 / ii_i1_value;
o0_i = V_DUMMY_ID__tmp2 * i2_data[i];
}

o0_data[i] = o0_i;
    }
}

        static void can_collapse_node_3d6ea06e14a2eac7e20cc661f60176f0_0(int nd, const int * dims, const int * strides, int collapse[])
        {
            //can we collapse dims[i] and dims[i-1]
            for(int i=nd-1;i>0;i--){
                if(strides[i]*dims[i]==strides[i-1]){//the dims nd-1 are not strided again dimension nd
                    collapse[i]=1;
                }else collapse[i]=0;
            }
        }
        

        static int callkernel_node_3d6ea06e14a2eac7e20cc661f60176f0_0(unsigned int numEls, const int d,
            const int * dims,
            const float * i0_data, const int * i0_str, const float * i1_data, const int * i1_str, const float * i2_data, const int * i2_str,
            float * o0_data, const int * o0_str)
        {
            numEls = dims[0]*1;
        
int local_dims[1];

            int local_str[3][1];
            int local_ostr[1][1];
            

        int nd_collapse = 1;
        for(int i=0;i<1;i++){//init new dim
          local_dims[i]=dims[i];
        }
        

            for(int i=0;i<1;i++){//init new strides
              local_str[0][i]=i0_str[i];
            }
            

            for(int i=0;i<1;i++){//init new strides
              local_str[1][i]=i1_str[i];
            }
            

            for(int i=0;i<1;i++){//init new strides
              local_str[2][i]=i2_str[i];
            }
            

            for(int i=0;i<1;i++){//init new strides
              local_ostr[0][i]=o0_str[i];
            }
            

        for(int id=0;id<nd_collapse;id++){

          bool all_broadcast=true;
          for(int input_id=0;input_id<3;input_id++){
            if(local_str[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          for(int input_id=0;input_id<1;input_id++){
            if(local_ostr[input_id][id]!=0 || local_dims[id]!=1) all_broadcast= false;
          }
          if(all_broadcast){
            for(int j=id+1;j<nd_collapse;j++)//remove dims i from the array
              local_dims[j-1]=local_dims[j];
            for(int input_id=0;input_id<3;input_id++){
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
        
int nd_collapse_[1] = {1};

                        int nd_collapse_0[1] = {1};

can_collapse_node_3d6ea06e14a2eac7e20cc661f60176f0_0(nd_collapse, local_dims, local_str[0], nd_collapse_0);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_0[i]==0)
nd_collapse_[i]=0;
}
                

                        int nd_collapse_2[1] = {1};

can_collapse_node_3d6ea06e14a2eac7e20cc661f60176f0_0(nd_collapse, local_dims, local_str[2], nd_collapse_2);
for(int i=0;i<nd_collapse;i++){
if(nd_collapse_2[i]==0)
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
 &&  local_str[0][nd_collapse-1]==1  && local_str[2][nd_collapse-1]==1  && local_ostr[0][nd_collapse-1]==1 
){nd_collapse=0;} 
if(numEls==0) return 0;
switch (nd_collapse==0?0:min(1,nd_collapse)) {
case 0: {

                //first use at least a full warp
                int threads_per_block = std::min(numEls,  (unsigned int)32); //WARP SIZE

                //next start adding multiprocessors
                int n_blocks = std::min(numEls/threads_per_block + (numEls % threads_per_block?1:0), (unsigned int)30); // UP TO NUMBER OF MULTIPROCESSORS

                // next start adding more warps per multiprocessor
                if (threads_per_block * n_blocks < numEls)
                    threads_per_block = std::min(numEls/n_blocks, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                kernel_Composite_node_3d6ea06e14a2eac7e20cc661f60176f0_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, i0_data, i1_data, i2_data, o0_data);

                //std::cerr << "calling callkernel returned\n";
                

                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n    n_blocks=%i threads_per_block=%i\n   Call: %s\n",
                         "GpuElemwise node_3d6ea06e14a2eac7e20cc661f60176f0_0 Composite", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_Composite_node_3d6ea06e14a2eac7e20cc661f60176f0_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, i0_data, i1_data, i2_data, o0_data)");
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

                kernel_Composite_node_3d6ea06e14a2eac7e20cc661f60176f0_0_1<<<n_blocks, threads_per_block>>>(numEls, local_dims[0], i0_data, local_str[0][0], i1_data, local_str[1][0], i2_data, local_str[2][0], o0_data, local_ostr[0][0]);
                

                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s.\n    n_blocks=%i threads_per_block=%i\n   Call: %s\n",
                         "GpuElemwise node_3d6ea06e14a2eac7e20cc661f60176f0_0 Composite", cudaGetErrorString(err),
                         n_blocks, threads_per_block,
                         "kernel_Composite_node_3d6ea06e14a2eac7e20cc661f60176f0_0_Ccontiguous<<<n_blocks, threads_per_block>>>(numEls, local_dims[0], i0_data, local_str[0][0], i1_data, local_str[1][0], i2_data, local_str[2][0], o0_data, local_ostr[0][0])");
                    return -1;

                }
                return 0;
                
        } break;
}
return -2;
}


    namespace {
    struct __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V1;
        

        __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0() {
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
        ~__struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V5, PyObject* storage_V7, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V5);
Py_XINCREF(storage_V7);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V5 = storage_V5;
this->storage_V7 = storage_V7;
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
__label_10:

double __DUMMY_10;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V7);
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
        

                if (V1->nd != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 1",
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
        

                if (V3->nd != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 1",
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
        

                if (V5->nd != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 1",
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
        

                if (V7->nd != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %i, it was supposed to have rank 1",
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
// Op class GpuElemwise

        //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} START\n";
        //standard elemwise size checks
            

            int dims[1] = {1};
            

                int broadcasts_V3[1] = {0};
                

                int broadcasts_V5[1] = {1};
                

                int broadcasts_V7[1] = {0};
                

        //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} checking input V3\n";
        if (1 != V3->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 1 dims, not %i", V3->nd);
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
        }
        for (int i = 0; i< 1; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V3)[i] : dims[i];
            if ((!(broadcasts_V3[i] &&
                 CudaNdarray_HOST_DIMS(V3)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V3)[i]))
            {
                //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} checking input V3 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 0 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V3)[i],
                             dims[i]
                            );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} checking input V5\n";
        if (1 != V5->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 1 dims, not %i", V5->nd);
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
        }
        for (int i = 0; i< 1; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V5)[i] : dims[i];
            if ((!(broadcasts_V5[i] &&
                 CudaNdarray_HOST_DIMS(V5)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V5)[i]))
            {
                //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} checking input V5 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 1 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V5)[i],
                             dims[i]
                            );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
            }
        }
            

        //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} checking input V7\n";
        if (1 != V7->nd)
        {
            PyErr_Format(PyExc_TypeError,
                         "need 1 dims, not %i", V7->nd);
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
        }
        for (int i = 0; i< 1; ++i)
        {
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(V7)[i] : dims[i];
            if ((!(broadcasts_V7[i] &&
                 CudaNdarray_HOST_DIMS(V7)[i] == 1)) &&
                (dims[i] != CudaNdarray_HOST_DIMS(V7)[i]))
            {
                //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} checking input V7 failed\n";
                PyErr_Format(PyExc_ValueError,
                             "GpuElemwise. Input dimension mis-match. Input"
                             " 2 (indices start at 0) has shape[%i] == %i"
                             ", but the output's size on that axis is %i.",
                             i,
                             CudaNdarray_HOST_DIMS(V7)[i],
                             dims[i]
                            );
                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
            }
        }
            

        Py_XDECREF(V1);
        V1 = V3;
        Py_INCREF(V1);
        for (int i = 0; (i< 1) && (V1); ++i) {
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
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
            }
        }
        //std::cerr << "ELEMWISE NEW V1 nd" << V1->nd << "\n";
        //std::cerr << "ELEMWISE NEW V1 data" << V1->devdata << "\n";
        

        {
            //new block so that failure gotos don't skip over variable initialization
            //std::cerr << "calling callkernel\n";
            if (callkernel_node_3d6ea06e14a2eac7e20cc661f60176f0_0(1, 0, dims
            

                        , CudaNdarray_DEV_DATA(V3), CudaNdarray_HOST_STRIDES(V3)
            

                        , CudaNdarray_DEV_DATA(V5), CudaNdarray_HOST_STRIDES(V5)
            

                        , CudaNdarray_DEV_DATA(V7), CudaNdarray_HOST_STRIDES(V7)
            

                        , CudaNdarray_DEV_DATA(V1), CudaNdarray_HOST_STRIDES(V1)
            

                        ))
            {
                 // error
            

                Py_DECREF(V1);
                V1 = NULL;
                

                {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
            }
            else // no error
            {
            }
        }
        //std::cerr << "C_CODE Composite{(((-i0) / i1) * i2)} END\n";
        
__label_9:

double __DUMMY_9;

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
    

        static int __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0_executor(__struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0* self) {
            return self->run();
        }

        static void __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (5 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 5, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0* struct_ptr = new __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0_executor), struct_ptr, __struct_compiled_op_3d6ea06e14a2eac7e20cc661f60176f0_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init3d6ea06e14a2eac7e20cc661f60176f0(void){
   (void) Py_InitModule("3d6ea06e14a2eac7e20cc661f60176f0", MyMethods);
}
