#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include "cuda_ndarray.cuh"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////


        #if NPY_API_VERSION >= 0x00000008
        typedef void (*inplace_map_binop)(PyArrayMapIterObject *,
                                          PyArrayIterObject *, int inc_or_set);
        
    #if defined(NPY_INT8)
    static void npy_int8_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int8*)mit->dataptr)[0] = (inc_or_set ? ((npy_int8*)mit->dataptr)[0] : 0) + ((npy_int8*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT16)
    static void npy_int16_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int16*)mit->dataptr)[0] = (inc_or_set ? ((npy_int16*)mit->dataptr)[0] : 0) + ((npy_int16*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT32)
    static void npy_int32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int32*)mit->dataptr)[0] = (inc_or_set ? ((npy_int32*)mit->dataptr)[0] : 0) + ((npy_int32*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT64)
    static void npy_int64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int64*)mit->dataptr)[0] = (inc_or_set ? ((npy_int64*)mit->dataptr)[0] : 0) + ((npy_int64*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT128)
    static void npy_int128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int128*)mit->dataptr)[0] = (inc_or_set ? ((npy_int128*)mit->dataptr)[0] : 0) + ((npy_int128*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT256)
    static void npy_int256_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int256*)mit->dataptr)[0] = (inc_or_set ? ((npy_int256*)mit->dataptr)[0] : 0) + ((npy_int256*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT8)
    static void npy_uint8_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint8*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint8*)mit->dataptr)[0] : 0) + ((npy_uint8*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT16)
    static void npy_uint16_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint16*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint16*)mit->dataptr)[0] : 0) + ((npy_uint16*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT32)
    static void npy_uint32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint32*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint32*)mit->dataptr)[0] : 0) + ((npy_uint32*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT64)
    static void npy_uint64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint64*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint64*)mit->dataptr)[0] : 0) + ((npy_uint64*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT128)
    static void npy_uint128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint128*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint128*)mit->dataptr)[0] : 0) + ((npy_uint128*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT256)
    static void npy_uint256_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint256*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint256*)mit->dataptr)[0] : 0) + ((npy_uint256*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT16)
    static void npy_float16_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float16*)mit->dataptr)[0] = (inc_or_set ? ((npy_float16*)mit->dataptr)[0] : 0) + ((npy_float16*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT32)
    static void npy_float32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float32*)mit->dataptr)[0] = (inc_or_set ? ((npy_float32*)mit->dataptr)[0] : 0) + ((npy_float32*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT64)
    static void npy_float64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float64*)mit->dataptr)[0] = (inc_or_set ? ((npy_float64*)mit->dataptr)[0] : 0) + ((npy_float64*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT80)
    static void npy_float80_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float80*)mit->dataptr)[0] = (inc_or_set ? ((npy_float80*)mit->dataptr)[0] : 0) + ((npy_float80*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT96)
    static void npy_float96_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float96*)mit->dataptr)[0] = (inc_or_set ? ((npy_float96*)mit->dataptr)[0] : 0) + ((npy_float96*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT128)
    static void npy_float128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float128*)mit->dataptr)[0] = (inc_or_set ? ((npy_float128*)mit->dataptr)[0] : 0) + ((npy_float128*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT256)
    static void npy_float256_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float256*)mit->dataptr)[0] = (inc_or_set ? ((npy_float256*)mit->dataptr)[0] : 0) + ((npy_float256*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX32)
    static void npy_complex32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex32*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex32*)mit->dataptr)[0].real : 0)
        + ((npy_complex32*)it->dataptr)[0].real;
    ((npy_complex32*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex32*)mit->dataptr)[0].imag : 0)
        + ((npy_complex32*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX64)
    static void npy_complex64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex64*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex64*)mit->dataptr)[0].real : 0)
        + ((npy_complex64*)it->dataptr)[0].real;
    ((npy_complex64*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex64*)mit->dataptr)[0].imag : 0)
        + ((npy_complex64*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX128)
    static void npy_complex128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex128*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex128*)mit->dataptr)[0].real : 0)
        + ((npy_complex128*)it->dataptr)[0].real;
    ((npy_complex128*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex128*)mit->dataptr)[0].imag : 0)
        + ((npy_complex128*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX160)
    static void npy_complex160_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex160*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex160*)mit->dataptr)[0].real : 0)
        + ((npy_complex160*)it->dataptr)[0].real;
    ((npy_complex160*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex160*)mit->dataptr)[0].imag : 0)
        + ((npy_complex160*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX192)
    static void npy_complex192_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex192*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex192*)mit->dataptr)[0].real : 0)
        + ((npy_complex192*)it->dataptr)[0].real;
    ((npy_complex192*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex192*)mit->dataptr)[0].imag : 0)
        + ((npy_complex192*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX512)
    static void npy_complex512_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex512*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex512*)mit->dataptr)[0].real : 0)
        + ((npy_complex512*)it->dataptr)[0].real;
    ((npy_complex512*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex512*)mit->dataptr)[0].imag : 0)
        + ((npy_complex512*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    static inplace_map_binop addition_funcs[] = {
#if defined(NPY_INT8)
npy_int8_inplace_add,
#endif

#if defined(NPY_INT16)
npy_int16_inplace_add,
#endif

#if defined(NPY_INT32)
npy_int32_inplace_add,
#endif

#if defined(NPY_INT64)
npy_int64_inplace_add,
#endif

#if defined(NPY_INT128)
npy_int128_inplace_add,
#endif

#if defined(NPY_INT256)
npy_int256_inplace_add,
#endif

#if defined(NPY_UINT8)
npy_uint8_inplace_add,
#endif

#if defined(NPY_UINT16)
npy_uint16_inplace_add,
#endif

#if defined(NPY_UINT32)
npy_uint32_inplace_add,
#endif

#if defined(NPY_UINT64)
npy_uint64_inplace_add,
#endif

#if defined(NPY_UINT128)
npy_uint128_inplace_add,
#endif

#if defined(NPY_UINT256)
npy_uint256_inplace_add,
#endif

#if defined(NPY_FLOAT16)
npy_float16_inplace_add,
#endif

#if defined(NPY_FLOAT32)
npy_float32_inplace_add,
#endif

#if defined(NPY_FLOAT64)
npy_float64_inplace_add,
#endif

#if defined(NPY_FLOAT80)
npy_float80_inplace_add,
#endif

#if defined(NPY_FLOAT96)
npy_float96_inplace_add,
#endif

#if defined(NPY_FLOAT128)
npy_float128_inplace_add,
#endif

#if defined(NPY_FLOAT256)
npy_float256_inplace_add,
#endif

#if defined(NPY_COMPLEX32)
npy_complex32_inplace_add,
#endif

#if defined(NPY_COMPLEX64)
npy_complex64_inplace_add,
#endif

#if defined(NPY_COMPLEX128)
npy_complex128_inplace_add,
#endif

#if defined(NPY_COMPLEX160)
npy_complex160_inplace_add,
#endif

#if defined(NPY_COMPLEX192)
npy_complex192_inplace_add,
#endif

#if defined(NPY_COMPLEX512)
npy_complex512_inplace_add,
#endif
NULL};
static int type_numbers[] = {
#if defined(NPY_INT8)
NPY_INT8,
#endif

#if defined(NPY_INT16)
NPY_INT16,
#endif

#if defined(NPY_INT32)
NPY_INT32,
#endif

#if defined(NPY_INT64)
NPY_INT64,
#endif

#if defined(NPY_INT128)
NPY_INT128,
#endif

#if defined(NPY_INT256)
NPY_INT256,
#endif

#if defined(NPY_UINT8)
NPY_UINT8,
#endif

#if defined(NPY_UINT16)
NPY_UINT16,
#endif

#if defined(NPY_UINT32)
NPY_UINT32,
#endif

#if defined(NPY_UINT64)
NPY_UINT64,
#endif

#if defined(NPY_UINT128)
NPY_UINT128,
#endif

#if defined(NPY_UINT256)
NPY_UINT256,
#endif

#if defined(NPY_FLOAT16)
NPY_FLOAT16,
#endif

#if defined(NPY_FLOAT32)
NPY_FLOAT32,
#endif

#if defined(NPY_FLOAT64)
NPY_FLOAT64,
#endif

#if defined(NPY_FLOAT80)
NPY_FLOAT80,
#endif

#if defined(NPY_FLOAT96)
NPY_FLOAT96,
#endif

#if defined(NPY_FLOAT128)
NPY_FLOAT128,
#endif

#if defined(NPY_FLOAT256)
NPY_FLOAT256,
#endif

#if defined(NPY_COMPLEX32)
NPY_COMPLEX32,
#endif

#if defined(NPY_COMPLEX64)
NPY_COMPLEX64,
#endif

#if defined(NPY_COMPLEX128)
NPY_COMPLEX128,
#endif

#if defined(NPY_COMPLEX160)
NPY_COMPLEX160,
#endif

#if defined(NPY_COMPLEX192)
NPY_COMPLEX192,
#endif

#if defined(NPY_COMPLEX512)
NPY_COMPLEX512,
#endif
-1000};
static int
map_increment(PyArrayMapIterObject *mit, PyObject *op,
              inplace_map_binop add_inplace, int inc_or_set)
{
    PyArrayObject *arr = NULL;
    PyArrayIterObject *it;
    PyArray_Descr *descr;
    if (mit->ait == NULL) {
        return -1;
    }
    descr = PyArray_DESCR(mit->ait->ao);
    Py_INCREF(descr);
    arr = (PyArrayObject *)PyArray_FromAny(op, descr,
                                0, 0, NPY_ARRAY_FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&arr, 0);
        if (arr == NULL) {
            return -1;
        }
    }
    it = (PyArrayIterObject*)
            PyArray_BroadcastToShape((PyObject*)arr, mit->dimensions, mit->nd);
    if (it  == NULL) {
        Py_DECREF(arr);
        return -1;
    }

    (*add_inplace)(mit, it, inc_or_set);

    Py_DECREF(arr);
    Py_DECREF(it);
    return 0;
}


static PyObject *
inplace_increment(PyObject *dummy, PyObject *args)
{
    PyObject *arg_a = NULL, *index=NULL, *inc=NULL;
    int inc_or_set = 1;
    PyArrayObject *a;
    inplace_map_binop add_inplace = NULL;
    int type_number = -1;
    int i = 0;
    PyArrayMapIterObject * mit;

    if (!PyArg_ParseTuple(args, "OOO|i", &arg_a, &index,
            &inc, &inc_or_set)) {
        return NULL;
    }
    if (!PyArray_Check(arg_a)) {
        PyErr_SetString(PyExc_ValueError,
                        "needs an ndarray as first argument");
        return NULL;
    }

    a = (PyArrayObject *) arg_a;

    if (PyArray_FailUnlessWriteable(a, "input/output array") < 0) {
        return NULL;
    }

    if (PyArray_NDIM(a) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return NULL;
    }
    type_number = PyArray_TYPE(a);



    while (type_numbers[i] >= 0 && addition_funcs[i] != NULL){
        if (type_number == type_numbers[i]) {
            add_inplace = addition_funcs[i];
            break;
        }
        i++ ;
    }

    if (add_inplace == NULL) {
        PyErr_SetString(PyExc_TypeError, "unsupported type for a");
        return NULL;
    }
    mit = (PyArrayMapIterObject *) PyArray_MapIterArray(a, index);
    if (mit == NULL) {
        goto fail;
    }
    if (map_increment(mit, inc, add_inplace, inc_or_set) != 0) {
        goto fail;
    }

    Py_DECREF(mit);

    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(mit);

    return NULL;
}
        #endif



        __global__ void k_vector_add_or_replace_fast(int numRowsX,
                                          int numColsX,
                                          int stridesX0,
                                          int stridesX1,
                                          float *X,
                                          int numRowsY,
                                          int numColsY,
                                          int stridesY0,
                                          int stridesY1,
                                          float *Y ,
                                          long *d_indices_arr,
                                          int num,
                                          const int set_instead_of_inc,
                                          int* err)
        {
             for (int i = (blockIdx.x); i < num; i += gridDim.x)
             {
                  for(int j = (threadIdx.x); j < numColsX;j += blockDim.x)
                  {
                      int x_row = d_indices_arr[i];
                      if(x_row < 0)
                          x_row += numRowsX;
                      int y_row = i;
                      if(x_row < numRowsX && x_row >= 0){
                        if(set_instead_of_inc){
                            atomicExch(&X[(x_row * stridesX0) + (j * stridesX1)],
                                  Y[(y_row * stridesY0) + (j * stridesY1)]);
                        } else{
                            atomicAdd(&X[(x_row * stridesX0) + (j * stridesX1)],
                                  Y[(y_row * stridesY0) + (j * stridesY1)]);
                        }
                      } else {
                        *err = 1;
                      }
                  }
             }
             return;
        }

        int CudaNdarray_vector_add_or_replace_fast(CudaNdarray* py_self,
            CudaNdarray* py_other, PyArrayObject *indices_arr,
            const int set_instead_of_inc)
        {
            if(init_err_var()!= 0) return -1;

            const int *shapeX = CudaNdarray_HOST_DIMS(py_self);
            const int *shapeY = CudaNdarray_HOST_DIMS(py_other);
            const int *strX   = CudaNdarray_HOST_STRIDES(py_self);
            const int *strY   = CudaNdarray_HOST_STRIDES(py_other);
            unsigned int size = (unsigned int)PyArray_SIZE(indices_arr);
            if(size == 0){
                return 0;
            }
            unsigned int numcolsX = shapeX[1];
            unsigned int num_threads_per_block = std::min(
                numcolsX, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
            unsigned int num_blocks = std::min(
                size, (unsigned int)NUM_VECTOR_OP_BLOCKS);

            dim3 n_blocks(num_blocks);
            dim3 n_threads(num_threads_per_block);
            long *d_indices_arr = NULL;
            PyArrayObject *cpu_indices_arr = PyArray_GETCONTIGUOUS(
                indices_arr);
            d_indices_arr = (long*)device_malloc(
                PyArray_NBYTES(cpu_indices_arr));

            if(!d_indices_arr)
                return -1;

            cudaError_t err = cudaMemcpy(d_indices_arr,
                                         PyArray_DATA(cpu_indices_arr),
                                         PyArray_NBYTES(cpu_indices_arr),
                                         cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuAdvancedIncSubtensor1_dev20:"
                    " cudaMemcpy returned an error: %s",
                    cudaGetErrorString(err));
                return -1;
            }

            k_vector_add_or_replace_fast<<<n_blocks, n_threads>>>(
                shapeX[0],
                shapeX[1],
                strX[0],
                strX[1],
                CudaNdarray_DEV_DATA(py_self),
                shapeY[0],
                shapeY[1],
                strY[0],
                strY[1],
                CudaNdarray_DEV_DATA(py_other),
                d_indices_arr,
                PyArray_SIZE(indices_arr),
                set_instead_of_inc,
                err_var
            );
            int index_err = check_err_var();

            device_free(d_indices_arr);
            Py_XDECREF(cpu_indices_arr);

            if(index_err != 0) return -1;

            err = cudaGetLastError();
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuAdvancedIncSubtensor1_dev20: cuda error: %s",
                    cudaGetErrorString(err));
                return -1;
            }
            return 0;
        }

        

    namespace {
    struct __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V1;
        

        __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9() {
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
        ~__struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9(void) {
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
    
        PyArrayObject* V7;
        
            typedef npy_int64 dtype_V7;
            
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
    
            V7 = NULL;
            if (py_V7 == Py_None) {
                // We can either fail here or set V7 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;}
            }
            if (!PyArray_Check(py_V7)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;}
            }
            // We expect NPY_INT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V7)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V7;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_INT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_INT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V7),
                             (long int) PyArray_NDIM(tmp),
                             (long int) PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1,
                             (long int) PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1,
                             (long int) PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1,
                             (long int) PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1,
                             (long int) PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1,
                             (long int) PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1
            );
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V7) != NPY_INT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_INT64) got %d",
                             NPY_INT64, PyArray_TYPE((PyArrayObject*) py_V7));
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_8;}
            }
            
        V7 = (PyArrayObject*)(py_V7);
        Py_XINCREF(V7);
        
{
// Op class GpuAdvancedIncSubtensor1_dev20

        Py_XDECREF(V1);
        if (!1) {
            V1 = (CudaNdarray*)CudaNdarray_Copy(V3);
        } else {
            V1 = V3;
            Py_XINCREF(V1);
        }

        if (CudaNdarray_vector_add_or_replace_fast(V1, V5, V7, 0) != 0){
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;}
        }

        if (!V1) {
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;}
        }
        __label_9:

double __DUMMY_9;

}
__label_8:

        if (V7) {
            Py_XDECREF(V7);
        }
        
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
    

        static int __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9_executor(__struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9* self) {
            return self->run();
        }

        static void __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9*)self);
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
  __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9* struct_ptr = new __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9_executor), struct_ptr, __struct_compiled_op_3b9d591b3c10087d1e880cbac3b1d2b9_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init3b9d591b3c10087d1e880cbac3b1d2b9(void){
   import_array();
   (void) Py_InitModule("3b9d591b3c10087d1e880cbac3b1d2b9", MyMethods);
}