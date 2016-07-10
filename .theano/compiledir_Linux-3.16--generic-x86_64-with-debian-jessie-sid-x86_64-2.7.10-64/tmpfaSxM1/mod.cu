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
    struct __struct_compiled_op_0022808f212c800898bd4f02880e4695 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V1;
        

        __struct_compiled_op_0022808f212c800898bd4f02880e4695() {
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
        ~__struct_compiled_op_0022808f212c800898bd4f02880e4695(void) {
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
    
        npy_int64 V7;
        
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
    
        assert(py_V5->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_V5))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %p %i\n", py_V5, (py_V5->ob_refcnt));
            V5 = (CudaNdarray*)py_V5;
            //std::cerr << "c_extract " << V5 << '\n';
        

                assert(V5);
                Py_INCREF(py_V5);
            }
            

{

    py_V7 = PyList_GET_ITEM(storage_V7, 0);
    {Py_XINCREF(py_V7);}
    
        PyArray_ScalarAsCtype(py_V7, &V7);
        
{
// Op class GpuIncSubtensor
CudaNdarray* zview = NULL;
        if (0)
        {
            if (V3 != V1)
            {
                Py_XDECREF(V1);
                Py_INCREF(V3);
                V1 = V3;
            }
        }
        else
        {
            Py_XDECREF(V1);
            V1 = (CudaNdarray*) CudaNdarray_Copy(V3);
        }
        
        // Argument of the view
        npy_intp xview_dims[3];
        npy_intp xview_strides[3];

        
        // One more argument of the view
        npy_intp xview_offset = 0;

        // The subtensor is created by iterating over the dimensions
        // and updating stride, shape, and data pointers

        int is_slice[] = {1};
        npy_intp subtensor_spec[3];
        subtensor_spec[0] = 9223372036854775806;
subtensor_spec[1] = V7;
subtensor_spec[2] = 9223372036854775806;;
        int spec_pos = 0; //position in subtensor_spec
        int inner_ii = 0; // the current dimension of zview
        int outer_ii = 0; // current dimension of z


        for (; outer_ii < 1; ++outer_ii)
        {
            if (is_slice[outer_ii])
            {
                npy_intp length = CudaNdarray_DIMS(V1)[outer_ii];
                npy_intp slicelength;
                npy_intp start = subtensor_spec[spec_pos+0];
                npy_intp stop  = subtensor_spec[spec_pos+1];
                npy_intp step  = subtensor_spec[spec_pos+2];
                if (step == 9223372036854775806) step = 1;

                npy_intp defstart = step < 0 ? length-1 : 0;
                npy_intp defstop = step < 0 ? -1 : length;

                // logic adapted from
                // PySlice_GetIndicesEx in python source
                if (!step)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "slice step cannot be zero");
                    {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
                }

                if (start == 9223372036854775806)
                {
                    start = defstart;
                }
                else
                {
                    if (start < 0) start += length;
                    if (start < 0) start = (step < 0) ? -1 : 0;
                    if (start >= length)
                        start = (step < 0) ? length - 1 : length;
                }

                if (stop == 9223372036854775806)
                {
                    stop = defstop;
                }
                else
                {
                    if (stop < 0) stop += length;
                    if (stop < 0) stop = (step < 0) ? -1 : 0;
                    if (stop >= length)
                        stop = (step < 0) ? length - 1 : length;
                }

                if ((step < 0 && stop >= start)
                    || (step > 0 && start >= stop)) {
                    slicelength = 0;
                }
                else if (step < 0) {
                    slicelength = (stop-start+1)/step+1;
                }
                else {
                    slicelength = (stop-start-1)/step+1;
                }

                if (0){
                    fprintf(stdout, "start %zi\n", start);
                    fprintf(stdout, "stop %zi\n", stop);
                    fprintf(stdout, "step %zi\n", step);
                    fprintf(stdout, "length %zi\n", length);
                    fprintf(stdout, "slicelength %zi\n", slicelength);
                }

                assert (slicelength <= length);

                xview_offset += (npy_intp)CudaNdarray_STRIDES(V1)[outer_ii]
                    * start * 4;
                xview_dims[inner_ii] = slicelength;
                xview_strides[inner_ii] = (npy_intp)CudaNdarray_STRIDES(V1)[outer_ii] * step;

                inner_ii += 1;
                spec_pos += 3;
            }
            else // tuple coord `outer_ii` is an int
            {
                int idx = subtensor_spec[spec_pos];
                if (idx < 0) idx += CudaNdarray_DIMS(V1)[outer_ii];
                if (idx >= 0)
                {
                    if (idx < CudaNdarray_DIMS(V1)[outer_ii])
                    {
                        xview_offset += (npy_intp)CudaNdarray_STRIDES(V1)[outer_ii] * idx *
                               4;
                    }
                    else
                    {
                        PyErr_Format(PyExc_IndexError,"index out of bounds");
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
                else
                {
                    PyErr_Format(PyExc_IndexError,"index out of bounds");
                    {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
                }

                spec_pos += 1;
            }
        }
        assert (inner_ii <= 3);
        while (inner_ii < 3)
        {
            assert (outer_ii < CudaNdarray_NDIM(V1));
            xview_dims[inner_ii] = CudaNdarray_DIMS(V1)[outer_ii];
            xview_strides[inner_ii] = CudaNdarray_STRIDES(V1)[outer_ii];

            inner_ii += 1;
            outer_ii += 1;
        }
        
        //TODO: give this Op a second output so that this view can be cached
        //TODO: alternatively, fix the memory leak on failure
        zview = (CudaNdarray*) CudaNdarray_New(3);
        if (CudaNdarray_set_device_data(
                zview,
                CudaNdarray_DEV_DATA(V1) + xview_offset/4,
                (PyObject*) V1))
        {
            zview = NULL;
            PyErr_Format(PyExc_RuntimeError,
                         "GpuSubtensor is not able to set the"
                         " devdata field of the view");
        }else{
            cnda_mark_dev_structure_dirty(zview);
            for(int idx=0;idx <3; idx++){
                if(xview_dims[idx]==1)
                    CudaNdarray_set_stride(zview, idx, 0);
                else
                    CudaNdarray_set_stride(zview, idx, xview_strides[idx]);
                CudaNdarray_set_dim(zview, idx, xview_dims[idx]);
            }
        }
        ;
        if (!zview)
        {
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
        }
        
        if (0)
        {
            if (CudaNdarray_CopyFromCudaNdarray(zview, V5, 1)) // does broadcasting
            {
                Py_DECREF(zview);
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
        else
        {
            
        PyObject * add_result = CudaNdarray_inplace_add((PyObject *) zview,
                                                        (PyObject *) V5);

        if (! add_result )
        {
            Py_DECREF(zview);
            {
        __failure = 9;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_9;};
        }
        else
        {
            Py_DECREF(add_result);
        }
        
        }
        Py_DECREF(zview);__label_9:

double __DUMMY_9;

}
__label_8:

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
    

        static int __struct_compiled_op_0022808f212c800898bd4f02880e4695_executor(__struct_compiled_op_0022808f212c800898bd4f02880e4695* self) {
            return self->run();
        }

        static void __struct_compiled_op_0022808f212c800898bd4f02880e4695_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_0022808f212c800898bd4f02880e4695*)self);
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
  __struct_compiled_op_0022808f212c800898bd4f02880e4695* struct_ptr = new __struct_compiled_op_0022808f212c800898bd4f02880e4695();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_0022808f212c800898bd4f02880e4695_executor), struct_ptr, __struct_compiled_op_0022808f212c800898bd4f02880e4695_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init0022808f212c800898bd4f02880e4695(void){
   import_array();
   (void) Py_InitModule("0022808f212c800898bd4f02880e4695", MyMethods);
}
