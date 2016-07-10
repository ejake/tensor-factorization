#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include "cuda_ndarray.cuh"
//////////////////////
////  Support Code
//////////////////////


            static __global__ void kernel_reduce_010_node_49998ffbdefdf18d464596bd95591cdb_0(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A, const int sA0,
                    const int sA1, const int sA2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }


                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        float myresult = 0;
                        for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                        {
                            myresult = myresult + A[i0 * sA0 + i1 * sA1 + i2 * sA2];;
                        }
                        
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < warpSize)
        {
            //round up all the partial sums into the first `warpSize` elements
            for (int i = threadNum + warpSize; i < threadCount; i += warpSize)
            {
                myresult = myresult + buf[i];
            }
            buf[threadNum] = myresult;
        /*Comment this optimization as it don't work on Fermi GPU.
        TODO: find why it don't work or put the GPU compute capability into the version
            // no sync because only one warp is running
            if(threadCount >32)
            {buf[threadNum] = buf[threadNum] + buf[threadNum+16];buf[threadNum] = buf[threadNum] + buf[threadNum+8];buf[threadNum] = buf[threadNum] + buf[threadNum+4];buf[threadNum] = buf[threadNum] + buf[threadNum+2];buf[threadNum] = buf[threadNum] + buf[threadNum+1];
                if (threadNum == 0)
                {
                    Z[i0 * sZ0 + i2*sZ1] = buf[0];
                }

            }
            else */
            if (threadNum < 16)
            {
                //reduce so that threadNum 0 has the reduction of everything
                if (threadNum + 16 < threadCount) buf[threadNum] = buf[threadNum] + buf[threadNum+16];if (threadNum + 8 < threadCount) buf[threadNum] = buf[threadNum] + buf[threadNum+8];if (threadNum + 4 < threadCount) buf[threadNum] = buf[threadNum] + buf[threadNum+4];if (threadNum + 2 < threadCount) buf[threadNum] = buf[threadNum] + buf[threadNum+2];if (threadNum + 1 < threadCount) buf[threadNum] = buf[threadNum] + buf[threadNum+1];
                if (threadNum == 0)
                {
                    Z[i0 * sZ0 + i2*sZ1] = buf[0];
                }
            }
        }
        
                    }
                }

            }
            

            static __global__ void kernel_reduce_010_AD_node_49998ffbdefdf18d464596bd95591cdb_0(
                    const int A,
                    const int B,
                    const int C,
                    const int D,
                    //const int E, // THIS is 32
                    const float *X, const int sX0,
                    const int sX1, const int sX2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                float myresult = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int a = blockIdx.x; a < A; a += gridDim.x)
                {
                    for (int i2_D = blockIdx.y; i2_D < D; i2_D += gridDim.y)
                    {
                        int c = i2_D * 32 + threadIdx.x;
                        if (c < C)
                        {
                            myresult = 0;
                            for (int b = 0; b < B; ++b)
                            {
                                myresult = myresult + X[a * sX0 + b * sX1 + c * sX2];;
                            }
                            Z[a * sZ0 + c * sZ1] = myresult;
                        }
                    }
                }

            }
            


    namespace {
    struct __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb() {
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
        ~__struct_compiled_op_49998ffbdefdf18d464596bd95591cdb(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
            



            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_6:

double __DUMMY_6;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
     CudaNdarray * V1;
    PyObject* py_V3;
     CudaNdarray * V3;
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
// Op class GpuCAReduce

        if (V3->nd != 2)
        {
            PyErr_Format(PyExc_TypeError,
                         "required nd=2, got nd=%i", V3->nd);
            {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;};
        }
        

        if (  !V1
           || (V1->nd != 1)
        
 || (CudaNdarray_HOST_DIMS(V1)[0] != CudaNdarray_HOST_DIMS(V3)[1]) 

           )
        {
            
int new_dims[1]; 
new_dims[0] = CudaNdarray_HOST_DIMS(V3)[1];

            Py_XDECREF(V1);
            V1 = (CudaNdarray*) CudaNdarray_NewDims(1, new_dims);
            if (NULL == V1)
            {
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;};
            }
        }
        

        if (CudaNdarray_SIZE(V1) && ! CudaNdarray_SIZE(V3)){
            cudaMemset(V1->devdata, 0, CudaNdarray_SIZE(V1) * sizeof(float));
        }
        else if (CudaNdarray_SIZE(V1))
        {
        

    {
        int verbose = 0;
        if(CudaNdarray_HOST_STRIDES(V3)[0] >
           CudaNdarray_HOST_STRIDES(V3)[1]){
                // If there are a lot of summations to do, then we can use simple parallelization -
                // use each thread to do one sum.

                // we might as well launch blocks of 32 threads because that's the warp size.
                // we could schedule more threads if we were maxing out the gridsize below, but
                // the gridsize is way more than the physical hardware and I think 32 threads
                // on a huge grid is enough to fully use the hardware.
                dim3 n_threads(32,1,1);

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                int A = 1;
                int B = CudaNdarray_HOST_DIMS(V3)[0];
                int C = CudaNdarray_HOST_DIMS(V3)[1];
                int D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                dim3 n_blocks(A,D);
                if (n_blocks.x > NUM_VECTOR_OP_BLOCKS) n_blocks.x = NUM_VECTOR_OP_BLOCKS;
                if (n_blocks.x*n_blocks.y > NUM_VECTOR_OP_BLOCKS) n_blocks.y = NUM_VECTOR_OP_BLOCKS/n_blocks.x;
                kernel_reduce_010_AD_node_49998ffbdefdf18d464596bd95591cdb_0<<<n_blocks, n_threads>>>(
                A,B,C,D,
                        CudaNdarray_DEV_DATA(V3),
                        1,
                        CudaNdarray_HOST_STRIDES(V3)[0],
                        CudaNdarray_HOST_STRIDES(V3)[1],
                        CudaNdarray_DEV_DATA(V1),
                        1,
                        CudaNdarray_HOST_STRIDES(V1)[0]
                        );

            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %s: %s."
                    " (grid: %i x %i; block: %i x %i x %i)\n",
                    "kernel_reduce_10_ADnode_49998ffbdefdf18d464596bd95591cdb_0",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;};
            }
        }else{
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(V3)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1,
                std::min(CudaNdarray_HOST_DIMS(V3)[1],
                    NUM_VECTOR_OP_BLOCKS));
            if (verbose) {
              fprintf(stderr,
                "running kernel_reduce_10_node_49998ffbdefdf18d464596bd95591cdb_0 n_blocks=(%i,%i)\n",
                n_blocks.x,
                n_blocks.y);
            }
            assert(CudaNdarray_HOST_DIMS(V3)[1] == CudaNdarray_HOST_DIMS(V1)[0]);
            int n_shared = sizeof(float) * n_threads.x;
            kernel_reduce_010_node_49998ffbdefdf18d464596bd95591cdb_0<<<n_blocks, n_threads, n_shared>>>(
                    1,
                    CudaNdarray_HOST_DIMS(V3)[0],
                    CudaNdarray_HOST_DIMS(V3)[1],
                    CudaNdarray_DEV_DATA(V3),
                    1,
                    CudaNdarray_HOST_STRIDES(V3)[0],
                    CudaNdarray_HOST_STRIDES(V3)[1],
                    CudaNdarray_DEV_DATA(V1),
                    1,
                    CudaNdarray_HOST_STRIDES(V1)[0]
                    );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %s: %s."
                    " (grid: %i x %i; block: %i x %i x %i)\n",
                    "kernel_reduce_010_node_49998ffbdefdf18d464596bd95591cdb_0",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;};
            }
        }
    }
        

        }
        
__label_5:

double __DUMMY_5;

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
    

        static int __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb_executor(__struct_compiled_op_49998ffbdefdf18d464596bd95591cdb* self) {
            return self->run();
        }

        static void __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_49998ffbdefdf18d464596bd95591cdb*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (3 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 3, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb* struct_ptr = new __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_49998ffbdefdf18d464596bd95591cdb_executor), struct_ptr, __struct_compiled_op_49998ffbdefdf18d464596bd95591cdb_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init49998ffbdefdf18d464596bd95591cdb(void){
   (void) Py_InitModule("49998ffbdefdf18d464596bd95591cdb", MyMethods);
}
