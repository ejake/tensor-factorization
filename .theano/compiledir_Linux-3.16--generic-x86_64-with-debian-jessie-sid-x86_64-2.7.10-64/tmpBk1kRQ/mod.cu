#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include "cuda_ndarray.cuh"
//////////////////////
////  Support Code
//////////////////////


    namespace {
    struct __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f() {
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
        ~__struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f(void) {
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
    
        PyArrayObject* V1;
        
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
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
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
        

                assert(V3);
                Py_INCREF(py_V3);
            }
            

{
// Op class Shape_i

    if (0>=CudaNdarray_NDIM(V3)){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;}
    }
    
    if(!V1)
        V1=(PyArrayObject*)PyArray_ZEROS(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(V1))[0] =
                              CudaNdarray_HOST_DIMS(V3)[0];
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
    

        static int __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f_executor(__struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f* self) {
            return self->run();
        }

        static void __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f*)self);
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
  __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f* struct_ptr = new __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f_executor), struct_ptr, __struct_compiled_op_a0a7e0eb1a001e30c64fa892c01d5d1f_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC inita0a7e0eb1a001e30c64fa892c01d5d1f(void){
   import_array();
   (void) Py_InitModule("a0a7e0eb1a001e30c64fa892c01d5d1f", MyMethods);
}
