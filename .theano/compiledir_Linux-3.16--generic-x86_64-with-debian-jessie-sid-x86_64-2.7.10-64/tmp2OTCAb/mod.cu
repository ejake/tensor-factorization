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


    namespace {
    struct __struct_compiled_op_26154939b61a7c7677747031cde80328 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V1;
        

        __struct_compiled_op_26154939b61a7c7677747031cde80328() {
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
        ~__struct_compiled_op_26154939b61a7c7677747031cde80328(void) {
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
    
        PyArrayObject* V5;
        
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
        

                assert(V1);
                Py_INCREF(py_V1);
            }
            

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

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
{
// Op class GpuReshape

        PyObject *new_shape = PyTuple_New(3);
        size_t total = 1;
        int compute_axis = -1;

        assert (PyArray_NDIM(V5) == 1);
        if (PyArray_DIM(V5, 0) != 3)
        {
            Py_XDECREF(new_shape);
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: given shape is of incorrect "
                         "length (%d should be %d).",
                         PyArray_DIM(V5, 0), 3);
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
        }

        for (size_t i = 0; i < 3; ++i)
        {
            long dimension = ((npy_int64*)(
                    PyArray_BYTES(V5) +
                    i * PyArray_STRIDES(V5)[0]))[0];
            if (dimension == -1)
            {
                if (compute_axis != -1)
                {
                    Py_XDECREF(new_shape);
                    PyErr_Format(PyExc_ValueError,
                                 "GpuReshape: only one -1 is accepted "
                                 "in the new shape, but got two at "
                                 "indices %d and %zu.",
                                 compute_axis, i);
                    {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
                }
                compute_axis = i;
            }
            else
            {
                total *= dimension;
                PyObject *py_dimension = PyInt_FromLong(dimension);
                PyTuple_SetItem(new_shape, i, py_dimension);
            }
        }

        if (compute_axis != -1)
        {
            long dimension = CudaNdarray_SIZE(V3) / total;
            total *= dimension;
            PyObject *py_dimension = PyInt_FromLong(dimension);
            PyTuple_SetItem(new_shape, compute_axis, py_dimension);
        }

        if (total != CudaNdarray_SIZE(V3))
        {
            const int *shape_from_py = CudaNdarray_HOST_DIMS(V3);

            char shape_from[128];
            size_t offset = 0;
            for (size_t i = 0; i < V3->nd; ++i)
            {
                int ws = snprintf(shape_from + offset, 128 - offset,
                        " %d,", shape_from_py[i]);
                offset += ws;
                if ( ws < 0 || offset >= 128 )
                    break;
            }

            shape_from[0]='(';
            if(offset < 128)
                shape_from[offset>0 ? offset-1 : 1] = ')';
            else
                for(size_t i=124; i<127; ++i)
                    shape_from[i] = '.';

            PyObject *shape_to_py = PyObject_Str(new_shape);
            const char *shape_to = PyString_AsString(shape_to_py);
            Py_XDECREF(new_shape);
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: cannot reshape input of shape "
                         "%s to shape %s.", shape_from, shape_to);
            {
        __failure = 7;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_7;};
        }

        Py_XDECREF(V1);
        V1 = (CudaNdarray*) CudaNdarray_Reshape(V3, new_shape);
        Py_XDECREF(new_shape);
        if (V1 == NULL)
        {
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

        if (V5) {
            Py_XDECREF(V5);
        }
        
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
    

        static int __struct_compiled_op_26154939b61a7c7677747031cde80328_executor(__struct_compiled_op_26154939b61a7c7677747031cde80328* self) {
            return self->run();
        }

        static void __struct_compiled_op_26154939b61a7c7677747031cde80328_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_26154939b61a7c7677747031cde80328*)self);
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
  __struct_compiled_op_26154939b61a7c7677747031cde80328* struct_ptr = new __struct_compiled_op_26154939b61a7c7677747031cde80328();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_26154939b61a7c7677747031cde80328_executor), struct_ptr, __struct_compiled_op_26154939b61a7c7677747031cde80328_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init26154939b61a7c7677747031cde80328(void){
   import_array();
   (void) Py_InitModule("26154939b61a7c7677747031cde80328", MyMethods);
}
