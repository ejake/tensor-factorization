#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include "cuda_ndarray.cuh"
//////////////////////
////  Support Code
//////////////////////


    namespace {
    struct __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V1;
        

        __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6() {
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
        ~__struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V5, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V5);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V5 = storage_V5;
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
__label_8:

double __DUMMY_8;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V5);
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
// Op class GpuDot22

        if (V3->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%i must be 2", V3->nd);
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
        }
        if (V5->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%i must be 2", V5->nd);
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
        }
        if ((NULL == V1)
            || (CudaNdarray_HOST_DIMS(V1)[0] !=
                  CudaNdarray_HOST_DIMS(V3)[0])
            || (CudaNdarray_HOST_DIMS(V1)[1] !=
                  CudaNdarray_HOST_DIMS(V5)[1])
            || (CudaNdarray_HOST_STRIDES(V1)[0] < 0)
            || (CudaNdarray_HOST_STRIDES(V1)[1] < 0)
            || ((CudaNdarray_HOST_DIMS(V1)[0] > 1)
                && (CudaNdarray_HOST_STRIDES(V1)[0] != 1)
                && (CudaNdarray_HOST_DIMS(V1)[1] > 1)
                && (CudaNdarray_HOST_STRIDES(V1)[1] != 1)))
        {
            Py_XDECREF(V1);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(V3)[0];
            dims[1] = CudaNdarray_HOST_DIMS(V5)[1];
            V1 = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == V1) ||
                CudaNdarray_alloc_contiguous(V1, 2, dims))
            {
                if (V1)
                {
                    Py_DECREF(V1);
                    V1 = NULL;
                }
                {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
            }
        }
        if (CudaNdarray_gemm(1.0f, V3, V5, 0.0f, V1))
        {
            if (V1)
            {
                Py_DECREF(V1);
                V1 = NULL;
            }
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
        }
        __label_7:

double __DUMMY_7;

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
    

        static int __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6_executor(__struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6* self) {
            return self->run();
        }

        static void __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (4 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 4, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6* struct_ptr = new __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6_executor), struct_ptr, __struct_compiled_op_6cb5e30303584f8897d57a1f5b2a9eb6_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init6cb5e30303584f8897d57a1f5b2a9eb6(void){
   (void) Py_InitModule("6cb5e30303584f8897d57a1f5b2a9eb6", MyMethods);
}
