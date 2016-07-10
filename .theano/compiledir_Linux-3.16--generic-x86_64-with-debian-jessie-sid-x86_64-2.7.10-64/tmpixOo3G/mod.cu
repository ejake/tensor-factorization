#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include "cudnn.h"
//////////////////////
////  Support Code
//////////////////////

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#endif


    namespace {
    struct __struct_compiled_op_265abc51f7c376c224983485238ff1a5 {
        PyObject* __ERROR;

        PyObject* storage_V1;
        

        __struct_compiled_op_265abc51f7c376c224983485238ff1a5() {
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
        ~__struct_compiled_op_265abc51f7c376c224983485238ff1a5(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V1) {
            Py_XINCREF(storage_V1);
            this->storage_V1 = storage_V1;
            


            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_4:

double __DUMMY_4;

            Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyObject* V1;
        
{

    py_V1 = Py_None;
    {Py_XINCREF(py_V1);}
    
        V1 = NULL;
        
{
// Op class DnnVersion

        #if defined(CUDNN_VERSION)
        V1 = PyTuple_Pack(2, PyInt_FromLong(CUDNN_VERSION), PyInt_FromLong(cudnnGetVersion()));
        #else
        V1 = PyInt_FromLong(-1);
        #endif
        __label_3:

double __DUMMY_3;

}
__label_2:

    if (!__failure) {
      
        assert(py_V1->ob_refcnt > 1);
        Py_DECREF(py_V1);
        py_V1 = V1 ? V1 : Py_None;
        Py_INCREF(py_V1);
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        Py_XDECREF(V1);
        
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
    

        static int __struct_compiled_op_265abc51f7c376c224983485238ff1a5_executor(__struct_compiled_op_265abc51f7c376c224983485238ff1a5* self) {
            return self->run();
        }

        static void __struct_compiled_op_265abc51f7c376c224983485238ff1a5_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_265abc51f7c376c224983485238ff1a5*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (2 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 2, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_265abc51f7c376c224983485238ff1a5* struct_ptr = new __struct_compiled_op_265abc51f7c376c224983485238ff1a5();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_265abc51f7c376c224983485238ff1a5_executor), struct_ptr, __struct_compiled_op_265abc51f7c376c224983485238ff1a5_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init265abc51f7c376c224983485238ff1a5(void){
   (void) Py_InitModule("265abc51f7c376c224983485238ff1a5", MyMethods);
}
