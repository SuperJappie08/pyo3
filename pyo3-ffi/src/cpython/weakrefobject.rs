#[cfg(not(PyPy))]
#[repr(C)]
#[derive(Debug)]
pub struct _PyWeakReference {
    pub ob_base: crate::PyObject,
    pub wr_object: *mut crate::PyObject,
    pub wr_callback: *mut crate::PyObject,
    pub hash: crate::Py_hash_t,
    pub wr_prev: *mut crate::PyWeakReference,
    pub wr_next: *mut crate::PyWeakReference,
    #[cfg(Py_3_11)]
    pub vectorcall: Option<crate::vectorcallfunc>,
}

// skipped _PyWeakref_ClearRef
// skipped PyWeakRef_GET_OBJECT

extern "C" {

    #[cfg(not(any(PyPy, Py_LIMITED_API)))]
    /// Gets the Weakref count of an object
    ///
    /// Argument `_head`: start of list
    ///
    /// Because the pointer needs to be the head of the list, this is wrapped to external functions, due to lack of documentation.
    pub fn _PyWeakref_GetWeakrefCount(_head: *mut _PyWeakReference) -> crate::Py_ssize_t;
}
