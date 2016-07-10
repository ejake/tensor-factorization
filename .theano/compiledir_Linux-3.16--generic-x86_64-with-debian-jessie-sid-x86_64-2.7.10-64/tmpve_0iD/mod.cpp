#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
//////////////////////
////  Support Code
//////////////////////


    extern "C"
    {

        void xerbla_(char*, void *);

    /***********/
    /* Level 1 */
    /***********/

    /* Single Precision */

        void srot_(const int*, float *, const int*, float *, const int*, const float *, const float *);
        void srotg_(float *,float *,float *,float *);
        void srotm_( const int*, float *, const int*, float *, const int*, const float *);
        void srotmg_(float *,float *,float *,const float *, float *);
        void sswap_( const int*, float *, const int*, float *, const int*);
        void scopy_( const int*, const float *, const int*, float *, const int*);
        void saxpy_( const int*, const float *, const float *, const int*, float *, const int*);
        float sdot_(const int*, const float *, const int*, const float *, const int*);
        void sdot_sub_(const int*, const float *, const int*, const float *, const int*, float *);
        void sdsdot_sub_( const int*, const float *, const float *, const int*, const float *, const int*, float *);
        void sscal_( const int*, const float *, float *, const int*);
        void snrm2_sub_( const int*, const float *, const int*, float *);
        void sasum_sub_( const int*, const float *, const int*, float *);
        void isamax_sub_( const int*, const float * , const int*, const int*);

    /* Double Precision */

        void drot_(const int*, double *, const int*, double *, const int*, const double *, const double *);
        void drotg_(double *,double *,double *,double *);
        void drotm_( const int*, double *, const int*, double *, const int*, const double *);
        void drotmg_(double *,double *,double *,const double *, double *);
        void dswap_( const int*, double *, const int*, double *, const int*);
        void dcopy_( const int*, const double *, const int*, double *, const int*);
        void daxpy_( const int*, const double *, const double *, const int*, double *, const int*);
        void dswap_( const int*, double *, const int*, double *, const int*);
        double ddot_(const int*, const double *, const int*, const double *, const int*);
        void dsdot_sub_(const int*, const float *, const int*, const float *, const int*, double *);
        void ddot_sub_( const int*, const double *, const int*, const double *, const int*, double *);
        void dscal_( const int*, const double *, double *, const int*);
        void dnrm2_sub_( const int*, const double *, const int*, double *);
        void dasum_sub_( const int*, const double *, const int*, double *);
        void idamax_sub_( const int*, const double * , const int*, const int*);

    /* Single Complex Precision */

        void cswap_( const int*, void *, const int*, void *, const int*);
        void ccopy_( const int*, const void *, const int*, void *, const int*);
        void caxpy_( const int*, const void *, const void *, const int*, void *, const int*);
        void cswap_( const int*, void *, const int*, void *, const int*);
        void cdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void cdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void cscal_( const int*, const void *, void *, const int*);
        void icamax_sub_( const int*, const void *, const int*, const int*);
        void csscal_( const int*, const float *, void *, const int*);
        void scnrm2_sub_( const int*, const void *, const int*, float *);
        void scasum_sub_( const int*, const void *, const int*, float *);

    /* Double Complex Precision */

        void zswap_( const int*, void *, const int*, void *, const int*);
        void zcopy_( const int*, const void *, const int*, void *, const int*);
        void zaxpy_( const int*, const void *, const void *, const int*, void *, const int*);
        void zswap_( const int*, void *, const int*, void *, const int*);
        void zdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void zdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void zdscal_( const int*, const double *, void *, const int*);
        void zscal_( const int*, const void *, void *, const int*);
        void dznrm2_sub_( const int*, const void *, const int*, double *);
        void dzasum_sub_( const int*, const void *, const int*, double *);
        void izamax_sub_( const int*, const void *, const int*, const int*);

    /***********/
    /* Level 2 */
    /***********/

    /* Single Precision */

        void sgemv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void sgbmv_(char*, const int*, const int*, const int*, const int*, const float *,  const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssymv_(char*, const int*, const float *, const float *, const int*, const float *,  const int*, const float *, float *, const int*);
        void ssbmv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void sspmv_(char*, const int*, const float *, const float *, const float *, const int*, const float *, float *, const int*);
        void strmv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
        void stbmv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
        void strsv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
        void stbsv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
        void stpmv_( char*, char*, char*, const int*, const float *, float *, const int*);
        void stpsv_( char*, char*, char*, const int*, const float *, float *, const int*);
        void sger_( const int*, const int*, const float *, const float *, const int*, const float *, const int*, float *, const int*);
        void ssyr_(char*, const int*, const float *, const float *, const int*, float *, const int*);
        void sspr_(char*, const int*, const float *, const float *, const int*, float *);
        void sspr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *);
        void ssyr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *, const int*);

    /* Double Precision */

        void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dgbmv_(char*, const int*, const int*, const int*, const int*, const double *,  const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsymv_(char*, const int*, const double *, const double *, const int*, const double *,  const int*, const double *, double *, const int*);
        void dsbmv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dspmv_(char*, const int*, const double *, const double *, const double *, const int*, const double *, double *, const int*);
        void dtrmv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
        void dtbmv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
        void dtrsv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
        void dtbsv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
        void dtpmv_( char*, char*, char*, const int*, const double *, double *, const int*);
        void dtpsv_( char*, char*, char*, const int*, const double *, double *, const int*);
        void dger_( const int*, const int*, const double *, const double *, const int*, const double *, const int*, double *, const int*);
        void dsyr_(char*, const int*, const double *, const double *, const int*, double *, const int*);
        void dspr_(char*, const int*, const double *, const double *, const int*, double *);
        void dspr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *);
        void dsyr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *, const int*);

    /* Single Complex Precision */

        void cgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void cgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
        void ctrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ctbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ctpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
        void ctrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ctbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ctpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
        void cgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void cgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
        void cher_(char*, const int*, const float *, const void *, const int*, void *, const int*);
        void cher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void chpr_(char*, const int*, const float *, const void *, const int*, void *);
        void chpr2_(char*, const int*, const float *, const void *, const int*, const void *, const int*, void *);

    /* Double Complex Precision */

        void zgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
        void ztrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ztbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ztpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
        void ztrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ztbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ztpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
        void zgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void zgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
        void zher_(char*, const int*, const double *, const void *, const int*, void *, const int*);
        void zher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void zhpr_(char*, const int*, const double *, const void *, const int*, void *);
        void zhpr2_(char*, const int*, const double *, const void *, const int*, const void *, const int*, void *);

    /***********/
    /* Level 3 */
    /***********/

    /* Single Precision */

        void sgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void ssyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void strmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
        void strsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

    /* Double Precision */

        void dgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void dsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dtrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
        void dtrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

    /* Single Complex Precision */

        void cgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void csymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void chemm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void csyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void cherk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void csyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void cher2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ctrmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
        void ctrsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

    /* Double Complex Precision */

        void zgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zhemm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void zherk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void zsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zher2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void ztrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
        void ztrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

    }
    

    namespace {
    struct __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V9;
PyObject* storage_V11;
PyObject* storage_V1;
        

        __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3() {
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
        ~__struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V5, PyObject* storage_V7, PyObject* storage_V9, PyObject* storage_V11, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V5);
Py_XINCREF(storage_V7);
Py_XINCREF(storage_V9);
Py_XINCREF(storage_V11);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V5 = storage_V5;
this->storage_V7 = storage_V7;
this->storage_V9 = storage_V9;
this->storage_V11 = storage_V11;
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
__label_14:

double __DUMMY_14;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V7);
Py_XDECREF(this->storage_V9);
Py_XDECREF(this->storage_V11);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
            typedef npy_float64 dtype_V1;
            
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
            typedef npy_float64 dtype_V3;
            
    PyObject* py_V5;
    
        PyArrayObject* V5;
        
            typedef npy_float64 dtype_V5;
            
    PyObject* py_V7;
    
        PyArrayObject* V7;
        
            typedef npy_float64 dtype_V7;
            
    PyObject* py_V9;
    
        PyArrayObject* V9;
        
            typedef npy_float64 dtype_V9;
            
    PyObject* py_V11;
    
        PyArrayObject* V11;
        
            typedef npy_float64 dtype_V11;
            
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
            V1 = NULL;
            if (py_V1 == Py_None) {
                // We can either fail here or set V1 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;}
            }
            if (!PyArray_Check(py_V1)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V1;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V1),
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
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V1) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V1));
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;}
            }
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
            V3 = NULL;
            if (py_V3 == Py_None) {
                // We can either fail here or set V3 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            if (!PyArray_Check(py_V3)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V3)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V3;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V3),
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
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V3) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V3));
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
            V5 = NULL;
            if (py_V5 == Py_None) {
                // We can either fail here or set V5 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;}
            }
            if (!PyArray_Check(py_V5)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V5)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V5;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V5),
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
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V5) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V5));
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_6;}
            }
            
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
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
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V7)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V7;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
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
            if (PyArray_TYPE((PyArrayObject*) py_V7) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V7));
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

    py_V9 = PyList_GET_ITEM(storage_V9, 0);
    {Py_XINCREF(py_V9);}
    
            V9 = NULL;
            if (py_V9 == Py_None) {
                // We can either fail here or set V9 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;}
            }
            if (!PyArray_Check(py_V9)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V9)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V9;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V9),
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
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V9) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V9));
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_10;}
            }
            
        V9 = (PyArrayObject*)(py_V9);
        Py_XINCREF(V9);
        
{

    py_V11 = PyList_GET_ITEM(storage_V11, 0);
    {Py_XINCREF(py_V11);}
    
            V11 = NULL;
            if (py_V11 == Py_None) {
                // We can either fail here or set V11 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;}
            }
            if (!PyArray_Check(py_V11)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V11)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V11;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V11),
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
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V11) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V11));
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_12;}
            }
            
        V11 = (PyArrayObject*)(py_V11);
        Py_XINCREF(V11);
        
{
// Op class CGemv


    int elemsize ;
    float fbeta;
    double dbeta;

    if (PyArray_NDIM(V3) != 1)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(aa) != 1");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }
    if (PyArray_NDIM(V7) != 2)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(xx) != 2");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }
    if (PyArray_NDIM(V9) != 1)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(yy) != 1");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }
    if (PyArray_NDIM(V5) != 0)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(alpha) != 0");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }
    if (PyArray_NDIM(V11) != 0)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Gemv: rank(beta) != 0");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }

    if (PyArray_DESCR(V3)->type_num != PyArray_DESCR(V7)->type_num)
    { PyErr_SetString(PyExc_TypeError, "Gemv: aa vs. xx"); {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}; }
    if (PyArray_DESCR(V3)->type_num != PyArray_DESCR(V9)->type_num)
    { PyErr_SetString(PyExc_TypeError, "Gemv: aa vs. yy"); {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}; }

    if (PyArray_DIMS(V7)[0] != PyArray_DIMS(V3)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[0] != x.shape[0]");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }
    if (PyArray_DIMS(V7)[1] != PyArray_DIMS(V9)[0])
    {
        PyErr_SetString(PyExc_ValueError,
                        "Shape mismatch: A.shape[1] != y.shape[0]");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }

    if  (PyArray_DESCR(V3)->type_num == NPY_DOUBLE) { elemsize = 8; }
    else if (PyArray_DESCR(V3)->type_num == NPY_FLOAT) { elemsize = 4;}
    else {
        PyErr_SetString(PyExc_NotImplementedError, "complex Gemv");
        {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;};
    }

    fbeta = dbeta = ((dtype_V11*)PyArray_DATA(V11))[0];

    // copy aa if not destructive
    if (!1)
    {
        if ((NULL == V1)
            || (PyArray_DIMS(V1)[0] != PyArray_DIMS(V3)[0]))
        {
            Py_XDECREF(V1);
            V1 = (PyArrayObject*)PyArray_SimpleNew(1,
                PyArray_DIMS(V3), PyArray_TYPE(V3));
            if(!V1) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemv output");
                {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
            }
        }
        if (V1 == V3)
        {
            PyErr_SetString(PyExc_AssertionError, "V1 != V3");
            {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
        }
        if (dbeta != 0)
        {
            if (PyArray_DESCR(V1)->type_num == NPY_FLOAT)
            {
                float * zoutdata = (float*)PyArray_DATA(V1);
                const float * zdata = (float*)PyArray_DATA(V3);
                int Ai = PyArray_STRIDES(V3)[0]/sizeof(float);
                int Zi = PyArray_STRIDES(V1)[0]/sizeof(float);
                for (int i = 0; i < PyArray_DIMS(V3)[0]; ++i)
                {
                    zoutdata[Zi*i] = fbeta * zdata[Ai*i];
                }
            }
            else if (PyArray_DESCR(V1)->type_num == NPY_DOUBLE)
            {
                double * zoutdata = (double*) PyArray_DATA(V1);
                const double * zdata = (double*)PyArray_DATA(V3);
                int Ai = PyArray_STRIDES(V3)[0]/sizeof(double);
                int Zi = PyArray_STRIDES(V1)[0]/sizeof(double);
                for (int i = 0; i < PyArray_DIMS(V3)[0]; ++i)
                {
                    zoutdata[Zi*i] = dbeta * zdata[Ai*i];
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError,
                                "neither float nor double dtype");
                {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
            }
            fbeta = dbeta = 1.0;
        }
        else if (0)
        {
            if (PyArray_CHKFLAGS(V1, NPY_ARRAY_C_CONTIGUOUS))
            {
                memset((void *)PyArray_DATA(V1), 0, PyArray_SIZE(V1)*PyArray_ITEMSIZE(V1));
            }
            else
            {
                if (PyArray_DESCR(V1)->type_num == NPY_FLOAT)
                {
                    float *zoutdata = (float *)PyArray_DATA(V1);
                    int Zi = PyArray_STRIDES(V1)[0]/sizeof(float);
                    for (int i = 0; i < PyArray_DIMS(V3)[0]; ++i)
                    {
                        zoutdata[Zi*i] = 0.0f;
                    }
                }
                else if (PyArray_DESCR(V1)->type_num == NPY_DOUBLE)
                {
                    double *zoutdata = (double *)PyArray_DATA(V1);
                    int Zi = PyArray_STRIDES(V1)[0]/sizeof(double);
                    for (int i = 0; i < PyArray_DIMS(V3)[0]; ++i)
                    {
                        zoutdata[Zi*i] = 0.0;
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
                }
            }
        }
    }
    else
    {
        //fprintf(stderr, "Gemv working in-place \n");
        if (V1 != V3)
        {
            if (V1) { Py_DECREF(V1); }
            V1 = V3;
            Py_INCREF(V1);
        }
    }
    {
        char TRANS = 'T';
        char NOTRANS = 'N';
        int Nx0 = PyArray_DIMS(V7)[0];
        int Nx1 = PyArray_DIMS(V7)[1];
        /* This formula is needed in the case where xx is actually a row or
         * column matrix, because BLAS sometimes insists that the strides:
         *  - are not smaller than the number of elements in the array
         *  - are not 0.
         */
        int Sx0 = (Nx0 > 1) ? (PyArray_STRIDES(V7)[0] / elemsize) : (Nx1 + 1);
        int Sx1 = (Nx1 > 1) ? (PyArray_STRIDES(V7)[1] / elemsize) : (Nx0 + 1);
        int Sz = PyArray_STRIDES(V1)[0] / elemsize;
        int Sy = PyArray_STRIDES(V9)[0] / elemsize;

        dtype_V9* yy_data = (dtype_V9*) PyArray_DATA(V9);
        dtype_V1* zz_data = (dtype_V1*) PyArray_DATA(V1);
        // gemv expects pointers to the beginning of memory arrays,
        // but numpy provides provides a pointer to the first element,
        // so when the stride is negative, we need to get the last one.
        if (Sy < 0)
            yy_data += (Nx1 - 1) * Sy;
        if (Sz < 0)
            zz_data += (Nx0 - 1) * Sz;

        if (Nx0 * Nx1)
        {
            // If xx is neither C- nor F-contiguous, we make a copy.
            // TODO:
            // - if one stride is equal to "- elemsize", we can still call
            //   gemv on reversed matrix and vectors
            // - if the copy is too long, maybe call vector/vector dot on
            //   each row instead
            if ((PyArray_STRIDES(V7)[0] < 0)
                || (PyArray_STRIDES(V7)[1] < 0)
                || ((PyArray_STRIDES(V7)[0] != elemsize)
                    && (PyArray_STRIDES(V7)[1] != elemsize)))
            {
                npy_intp dims[2];
                dims[0] = Nx0;
                dims[1] = Nx1;

                PyArrayObject * xx_copy = (PyArrayObject *) PyArray_Copy(
                                                                    V7);
                if (!xx_copy)
                    {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
                Py_XDECREF(V7);
                V7 = xx_copy;
                Sx0 = (Nx0 > 1) ? (PyArray_STRIDES(V7)[0] / elemsize) : (Nx1 + 1);
                Sx1 = (Nx1 > 1) ? (PyArray_STRIDES(V7)[1] / elemsize) : (Nx0 + 1);
            }

            if (PyArray_STRIDES(V7)[0] == elemsize)
            {
                if (PyArray_DESCR(V7)->type_num == NPY_FLOAT)
                {
                    //fprintf(stderr, "A\n");
                    float alpha = ((dtype_V5*)PyArray_DATA(V5))[0];
                    sgemv_(&NOTRANS, &Nx0, &Nx1,
                        &alpha,
                        (float*)(PyArray_DATA(V7)), &Sx1,
                        (float*)yy_data, &Sy,
                        &fbeta,
                        (float*)zz_data, &Sz);
                }
                else if (PyArray_DESCR(V7)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_V5*)PyArray_DATA(V5))[0];
                    dgemv_(&NOTRANS, &Nx0, &Nx1,
                        &alpha,
                        (double*)(PyArray_DATA(V7)), &Sx1,
                        (double*)yy_data, &Sy,
                        &dbeta,
                        (double*)zz_data, &Sz);
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
                }
            }
            else if (PyArray_STRIDES(V7)[1] == elemsize)
            {
                if (PyArray_DESCR(V7)->type_num == NPY_FLOAT)
                {
                    float alpha = ((dtype_V5*)PyArray_DATA(V5))[0];

                    // Check for vector-vector dot (Nx0 == 1). The code may work
                    // for Sx1 != 1 as well, but has not been tested for this case,
                    // so Sx1 == 1 is required for safety.
                    if (Nx0 == 1 && Sx1 == 1)
                    {
                        zz_data[0] = fbeta*zz_data[0] + alpha*sdot_(&Nx1,
                            (float*)(PyArray_DATA(V7)), &Sx1,
                            (float*)yy_data, &Sy);
                    }
                    else
                    {
                        sgemv_(&TRANS, &Nx1, &Nx0,
                            &alpha,
                            (float*)(PyArray_DATA(V7)), &Sx0,
                            (float*)yy_data, &Sy,
                            &fbeta,
                            (float*)zz_data, &Sz);
                    }
                }
                else if (PyArray_DESCR(V7)->type_num == NPY_DOUBLE)
                {
                    double alpha = ((dtype_V5*)PyArray_DATA(V5))[0];

                    // Check for vector-vector dot (Nx0 == 1). The code may work
                    // for Sx1 != 1 as well, but has not been tested for this case,
                    // so Sx1 == 1 is required for safety.
                    if (Nx0 == 1 && Sx1 == 1)
                    {
                        zz_data[0] = dbeta*zz_data[0] + alpha*ddot_(&Nx1,
                              (double*)(PyArray_DATA(V7)), &Sx1,
                              (double*)yy_data, &Sy);
                    }
                    else
                    {
                        dgemv_(&TRANS, &Nx1, &Nx0,
                            &alpha,
                            (double*)(PyArray_DATA(V7)), &Sx0,
                            (double*)yy_data, &Sy,
                            &dbeta,
                            (double*)zz_data, &Sz);
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_AssertionError,
                                    "neither float nor double dtype");
                    {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError,
                    "xx is a double-strided matrix, and should have been "
                    "copied into a memory-contiguous one.");
                {
        __failure = 13;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_13;}
            }
        }
        else if (dbeta != 1.0)
        {
            // the matrix has at least one dim of length 0
            // so we do this loop, which either iterates over 0 elements
            // or else it does the right thing for length-0 x.
            dtype_V1 * zptr = (dtype_V1*)(PyArray_DATA(V1));
            for (int i = 0; i < Nx0; ++i)
            {
                zptr[i * Sz] *= dbeta;
            }
        }
    }

    __label_13:

double __DUMMY_13;

}
__label_12:

        if (V11) {
            Py_XDECREF(V11);
        }
        
    {Py_XDECREF(py_V11);}
    
double __DUMMY_12;

}
__label_10:

        if (V9) {
            Py_XDECREF(V9);
        }
        
    {Py_XDECREF(py_V9);}
    
double __DUMMY_10;

}
__label_8:

        if (V7) {
            Py_XDECREF(V7);
        }
        
    {Py_XDECREF(py_V7);}
    
double __DUMMY_8;

}
__label_6:

        if (V5) {
            Py_XDECREF(V5);
        }
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

        if (V3) {
            Py_XDECREF(V3);
        }
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        {Py_XDECREF(py_V1);}
        if (!V1) {
            Py_INCREF(Py_None);
            py_V1 = Py_None;
        }
        else if ((void*)py_V1 != (void*)V1) {
            py_V1 = (PyObject*)V1;
        }

        {Py_XINCREF(py_V1);}

        if (V1 && !PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                         (long int) PyArray_NDIM(V1),
                         (long int) PyArray_NDIM(V1) >= 3 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-3] : -1,
                         (long int) PyArray_NDIM(V1) >= 2 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-2] : -1,
                         (long int) PyArray_NDIM(V1) >= 1 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-1] : -1,
                         (long int) PyArray_NDIM(V1) >= 3 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-3] : -1,
                         (long int) PyArray_NDIM(V1) >= 2 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-2] : -1,
                         (long int) PyArray_NDIM(V1) >= 1 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-1] : -1
        );
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        if (V1) {
            Py_XDECREF(V1);
        }
        
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
    

        static int __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3_executor(__struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3* self) {
            return self->run();
        }

        static void __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (7 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 7, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3* struct_ptr = new __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4),PyTuple_GET_ITEM(argtuple, 5),PyTuple_GET_ITEM(argtuple, 6) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3_executor), struct_ptr, __struct_compiled_op_6b71b97a161fa8e245d434aa4cd241c3_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init6b71b97a161fa8e245d434aa4cd241c3(void){
   import_array();
   (void) Py_InitModule("6b71b97a161fa8e245d434aa4cd241c3", MyMethods);
}
