//! An interface for Python's `weakref` module.
use std::{marker::PhantomData, mem, ptr::NonNull};

use crate::{
    err, ffi, gil,
    types::{PyDict, PyString, PyTuple},
    AsPyPointer, IntoPy, IntoPyPointer, Py, PyAny, PyErr, PyObject, PyResult, Python, ToPyObject,
};

/// A representation of a Python [`weakref.ref`](https://docs.python.org/3/library/weakref.html?highlight=weakref#weakref.ref).
///
/// TODO: Maybe a [`weakref.proxy`](https://docs.python.org/3/library/weakref.html?highlight=weakref#weakref.proxy) equivalent.
#[repr(transparent)]
pub struct PyWeak<T>(NonNull<ffi::PyObject>, PhantomData<T>);

// The inner value is only accessed through ways that require proving the gil is held
#[cfg(feature = "nightly")]
unsafe impl<T> crate::marker::Ungil for PyWeak<T> {}
unsafe impl<T> Send for PyWeak<T> {}
unsafe impl<T> Sync for PyWeak<T> {}

impl<T> PyWeak<T> {
    /// Returns whether `self` and `other` point to the same object. To compare
    /// the equality of two objects (the `==` operator), use [`eq`](PyAny::eq).
    ///
    /// This is equivalent to the Python expression `self is other`.
    #[inline]
    pub fn is<U: AsPyPointer>(&self, o: &U) -> bool {
        self.as_ptr() == o.as_ptr()
    }

    /// Gets refcount of the current object
    pub fn get_refcnt(&self, _py: Python<'_>) -> isize {
        unsafe { ffi::Py_REFCNT(self.as_ptr()) }
    }

    /// Get the weakeref count to the current object
    pub fn get_weak_refcnt(&self, _py: Python<'_>) -> isize {
        // (Most)Weakrefs are not weakreferenceble
        0
    }

    /// Makes a clone of `self`.
    ///
    /// This creates another pointer to the same object (A `weakref` reference), increasing its reference count.
    ///
    /// You should prefer using this method over [`Clone`] if you happen to be holding the GIL already.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pyo3::prelude::*;
    ///
    /// #[pyclass(weakref)]
    /// struct MyClass{
    ///     name: String,
    ///     age: i32,
    /// }
    ///
    /// # fn main() {
    /// Python::with_gil(|py| {
    ///     let first: Py<MyClass> = MyClass {
    ///         name: "Joe".to_owned(),
    ///          age: 42
    ///     }.into_py(py).extract(py).unwrap();
    ///     let second = Py::clone_ref(&first, py);
    ///     let first_weak = first.downgrade(py);
    ///
    ///     // Both point to the same object
    ///     assert!(first.is(&second));
    ///     assert!(first_weak.upgrade(py).unwrap().is(&first));
    ///     assert!(first_weak.upgrade(py).unwrap().is(&second));
    ///     # assert_eq!(first.get_refcnt(py), 2);
    ///     # assert_eq!(second.get_refcnt(py), 2);
    ///     # assert_eq!(first_weak.ref_weak_count(py), 1);
    ///     # assert_eq!(first_weak.ref_strong_count(py), 2);
    ///     # assert_eq!(first_weak.get_refcnt(py), 1);
    ///     # assert_eq!(first_weak.get_weak_refcnt(py), 0);
    ///     # assert_eq!(second.as_ref(py).borrow().name, "Joe".to_owned());
    ///     # assert_eq!(second.as_ref(py).borrow().age, 42);
    /// });
    /// # }
    /// ```
    #[inline]
    pub fn clone_ref(&self, py: Python<'_>) -> PyWeak<T> {
        unsafe { PyWeak::from_borrowed_ptr(py, self.as_ptr()) }
    }

    /// Returns whether the `weakref` object is considered to be None.
    ///
    /// This is equivalent to the Python expression `self is None`.
    pub fn is_none(&self, _py: Python<'_>) -> bool {
        unsafe { ffi::Py_None() == self.as_ptr() }
    }

    /// Returns whether the `weakref` object is Ellipsis, e.g. `...`.
    ///
    /// This is equivalent to the Python expression `self is ...`.
    pub fn is_ellipsis(&self) -> bool {
        unsafe { ffi::Py_Ellipsis() == self.as_ptr() }
    }

    /// Returns whether the `weakref` object is considered to be true.
    ///
    /// This is equivalent to the Python expression `bool(self)`.
    pub fn is_true(&self, py: Python<'_>) -> PyResult<bool> {
        let v = unsafe { ffi::PyObject_IsTrue(self.as_ptr()) };
        err::error_on_minusone(py, v)?;
        Ok(v != 0)
    }

    /// Retrieves an attribute value.
    ///
    /// This is equivalent to the Python expression `self.attr_name`.
    ///
    /// If calling this method becomes performance-critical, the [`intern!`](crate::intern) macro
    /// can be used to intern `attr_name`, thereby avoiding repeated temporary allocations of
    /// Python strings.
    ///
    /// # Example: `intern!`ing the attribute name
    ///
    /// ```
    /// # use pyo3::{intern, pyfunction, types::PyModule, IntoPy, Py, Python, PyObject, PyResult};
    /// #
    /// #[pyfunction]
    /// fn version(sys: Py<PyModule>, py: Python<'_>) -> PyResult<PyObject> {
    ///     sys.getattr(py, intern!(py, "version"))
    /// }
    /// #
    /// # Python::with_gil(|py| {
    /// #    let sys = py.import("sys").unwrap().into_py(py);
    /// #    version(sys, py).unwrap();
    /// # });
    /// ```
    pub fn getattr<N>(&self, py: Python<'_>, attr_name: N) -> PyResult<PyObject>
    where
        N: IntoPy<Py<PyString>>,
    {
        let attr_name = attr_name.into_py(py);

        unsafe {
            PyObject::from_owned_ptr_or_err(
                py,
                ffi::PyObject_GetAttr(self.as_ptr(), attr_name.as_ptr()),
            )
        }
    }

    /// Sets an attribute value.
    ///
    /// This is equivalent to the Python expression `self.attr_name = value`.
    ///
    /// To avoid repeated temporary allocations of Python strings, the [`intern!`](crate::intern)
    /// macro can be used to intern `attr_name`.
    ///
    /// # Example: `intern!`ing the attribute name
    ///
    /// ```
    /// # use pyo3::{intern, pyfunction, types::PyModule, IntoPy, PyObject, Python, PyResult};
    /// #
    /// #[pyfunction]
    /// fn set_answer(ob: PyObject, py: Python<'_>) -> PyResult<()> {
    ///     ob.setattr(py, intern!(py, "answer"), 42)
    /// }
    /// #
    /// # Python::with_gil(|py| {
    /// #    let ob = PyModule::new(py, "empty").unwrap().into_py(py);
    /// #    set_answer(ob, py).unwrap();
    /// # });
    /// ```
    pub fn setattr<N, V>(&self, py: Python<'_>, attr_name: N, value: V) -> PyResult<()>
    where
        N: IntoPy<Py<PyString>>,
        V: IntoPy<Py<PyAny>>,
    {
        let attr_name = attr_name.into_py(py);
        let value = value.into_py(py);

        unsafe {
            err::error_on_minusone(
                py,
                ffi::PyObject_SetAttr(self.as_ptr(), attr_name.as_ptr(), value.as_ptr()),
            )
        }
    }

    /// TODO: Does this make sense? you can call with??? Vectorcall???  without arguments to get a (strong) reference to the object
    /// ---------
    ///
    /// Calls the object.
    ///
    /// This is equivalent to the Python expression `self(*args, **kwargs)`.
    pub fn call(
        &self,
        py: Python<'_>,
        args: impl IntoPy<Py<PyTuple>>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let args = args.into_py(py);
        let kwargs = kwargs.into_ptr();

        unsafe {
            let ret = PyObject::from_owned_ptr_or_err(
                py,
                ffi::PyObject_Call(self.as_ptr(), args.as_ptr(), kwargs),
            );
            ffi::Py_XDECREF(kwargs);
            ret
        }
    }

    /// TODO: Does this make sense? you can call without arguments to get a (strong) reference to the object
    /// Calls the object with only positional arguments.
    ///
    /// This is equivalent to the Python expression `self(*args)`.
    pub fn call1(&self, py: Python<'_>, args: impl IntoPy<Py<PyTuple>>) -> PyResult<PyObject> {
        self.call(py, args, None)
    }

    /// Calls the object without arguments.
    ///
    /// This is equivalent to the Python expression `self()`.
    pub fn call0(&self, py: Python<'_>) -> PyResult<PyObject> {
        cfg_if::cfg_if! {
            if #[cfg(all(
                not(PyPy),
                any(Py_3_10, all(not(Py_LIMITED_API), Py_3_9)) // PyObject_CallNoArgs was added to python in 3.9 but to limited API in 3.10
            ))] {
                // Optimized path on python 3.9+
                unsafe {
                    PyObject::from_owned_ptr_or_err(py, ffi::PyObject_CallNoArgs(self.as_ptr()))
                }
            } else {
                self.call(py, (), None)
            }
        }
    }

    // This still makes sense, possible methods are: __repr__, __hash__, __call__, __eq__, __neq__, ?__del__?, ?__getattr__? ...
    /// Calls a method on the object.
    ///
    /// This is equivalent to the Python expression `self.name(*args, **kwargs)`.
    ///
    /// To avoid repeated temporary allocations of Python strings, the [`intern!`](crate::intern)
    /// macro can be used to intern `name`.
    pub fn call_method<N, A>(
        &self,
        py: Python<'_>,
        name: N,
        args: A,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject>
    where
        N: IntoPy<Py<PyString>>,
        A: IntoPy<Py<PyTuple>>,
    {
        let callee = self.getattr(py, name)?;
        let args: Py<PyTuple> = args.into_py(py);
        let kwargs = kwargs.into_ptr();

        unsafe {
            let result = PyObject::from_owned_ptr_or_err(
                py,
                ffi::PyObject_Call(callee.as_ptr(), args.as_ptr(), kwargs),
            );
            ffi::Py_XDECREF(kwargs);
            result
        }
    }

    /// Calls a method on the object with only positional arguments.
    ///
    /// This is equivalent to the Python expression `self.name(*args)`.
    ///
    /// To avoid repeated temporary allocations of Python strings, the [`intern!`](crate::intern)
    /// macro can be used to intern `name`.
    pub fn call_method1<N, A>(&self, py: Python<'_>, name: N, args: A) -> PyResult<PyObject>
    where
        N: IntoPy<Py<PyString>>,
        A: IntoPy<Py<PyTuple>>,
    {
        self.call_method(py, name, args, None)
    }

    /// Calls a method on the object with no arguments.
    ///
    /// This is equivalent to the Python expression `self.name()`.
    ///
    /// To avoid repeated temporary allocations of Python strings, the [`intern!`](crate::intern)
    /// macro can be used to intern `name`.
    pub fn call_method0<N>(&self, py: Python<'_>, name: N) -> PyResult<PyObject>
    where
        N: IntoPy<Py<PyString>>,
    {
        cfg_if::cfg_if! {
            if #[cfg(all(Py_3_9, not(any(Py_LIMITED_API, PyPy))))] {
                // Optimized path on python 3.9+
                unsafe {
                    let name: Py<PyString> = name.into_py(py);
                    PyObject::from_owned_ptr_or_err(py, ffi::PyObject_CallMethodNoArgs(self.as_ptr(), name.as_ptr()))
                }
            } else {
                self.call_method(py, name, (), None)
            }
        }
    }

    /// Create a `PyWeak<T>` instance by taking ownership of the given FFI pointer to a `PyWeakReference`.
    ///
    /// # Safety
    /// `ptr` must be a pointer to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    ///
    /// Callers must own the object referred to by `ptr`, as this function
    /// implicitly takes ownership of that object.
    ///
    /// # Panics
    /// Panics if `ptr` is null.
    #[inline]
    pub unsafe fn from_owned_ptr(py: Python<'_>, ptr: *mut ffi::PyObject) -> PyWeak<T> {
        match NonNull::new(ptr) {
            Some(nonnull_ptr) => PyWeak(nonnull_ptr, PhantomData),
            None => crate::err::panic_after_error(py),
        }
    }

    /// Create a `PyWeak<T>` instance by taking ownership of the given FFI pointer to a `PyWeakReference`.
    ///
    /// If `ptr` is null then the current Python exception is fetched as a [`PyErr`].
    ///
    /// # Safety
    /// If non-null, `ptr` must be a pointer to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    #[inline]
    pub unsafe fn from_owned_ptr_or_err(
        py: Python<'_>,
        ptr: *mut ffi::PyObject,
    ) -> PyResult<PyWeak<T>> {
        match NonNull::new(ptr) {
            Some(nonnull_ptr) => Ok(PyWeak(nonnull_ptr, PhantomData)),
            None => Err(PyErr::fetch(py)),
        }
    }

    /// Create a `PyWeak<T>` instance by taking ownership of the given FFI pointer to a `PyWeakReference`.
    ///
    /// If `ptr` is null then `None` is returned.
    ///
    /// # Safety
    /// If non-null, `ptr` must be a pointer to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    #[inline]
    pub unsafe fn from_owned_ptr_or_opt(_py: Python<'_>, ptr: *mut ffi::PyObject) -> Option<Self> {
        NonNull::new(ptr).map(|nonnull_ptr| PyWeak(nonnull_ptr, PhantomData))
    }

    /// Create a `PyWeak<T>` instance by creating a new reference from the given FFI pointer to a `PyWeakReference`.
    ///
    /// # Safety
    /// `ptr` must be a pointer to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    ///
    /// # Panics
    /// Panics if `ptr` is null.
    #[inline]
    pub unsafe fn from_borrowed_ptr(py: Python<'_>, ptr: *mut ffi::PyObject) -> PyWeak<T> {
        match Self::from_borrowed_ptr_or_opt(py, ptr) {
            Some(slf) => slf,
            None => crate::err::panic_after_error(py),
        }
    }

    /// Create a `PyWeak<T>` instance by creating a new reference from the given FFI pointer.
    ///
    /// If `ptr` is null then the current Python exception is fetched as a [`PyErr`].
    ///
    /// # Safety
    /// `ptr` must be a pointer to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    #[inline]
    pub unsafe fn from_borrowed_ptr_or_err(
        py: Python<'_>,
        ptr: *mut ffi::PyObject,
    ) -> PyResult<Self> {
        Self::from_borrowed_ptr_or_opt(py, ptr).ok_or_else(|| PyErr::fetch(py))
    }

    /// Create a `PyWeak<T>` instance by creating a new reference from the given FFI pointer.
    ///
    /// If `ptr` is null then `None` is returned.
    ///
    /// # Safety
    /// `ptr` must be a pointer to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    #[inline]
    pub unsafe fn from_borrowed_ptr_or_opt(
        _py: Python<'_>,
        ptr: *mut ffi::PyObject,
    ) -> Option<Self> {
        NonNull::new(ptr).map(|nonnull_ptr| {
            ffi::Py_INCREF(ptr);
            Self::from_non_null(nonnull_ptr)
        })
    }

    // TODO: Is this necessary, only used for above to suppress error here...
    /// For internal conversions.
    ///
    /// # Safety
    /// `ptr` must point to a Python [`WeakReference`](ffi::PyWeakReference) object, which references a Python object of type `T`.
    #[inline]
    unsafe fn from_non_null(ptr: NonNull<ffi::PyObject>) -> Self {
        Self(ptr, PhantomData)
    }

    /// Returns the inner pointer without decreasing the refcount.
    #[inline]
    fn into_non_null(self) -> NonNull<ffi::PyObject> {
        let pointer = self.0;
        mem::forget(self);
        pointer
    }
}

/// TODO: This specifier should probably be removed to work with PyAny
impl<T> PyWeak<T>
where
    T: crate::impl_::pyclass::PyClassImpl<WeakRef = crate::impl_::pyclass::PyClassWeakRefSlot>,
{
    /// Gets the count of strong refences to the referenced object.
    pub fn ref_strong_count(&self, _py: Python<'_>) -> isize {
        cfg_if::cfg_if! {
            if #[cfg(not(any(PyPy, Py_LIMITED_API)))] {
                unsafe { ffi::Py_REFCNT(self.as_ptr().cast::<ffi::PyWeakReference>().read().wr_object) }
            } else {
                unsafe { ffi::Py_REFCNT(ffi::PyWeakref_GetObject(self.as_ptr())) }
            }
        }
    }

    /// Gets the count of weak refences to the referenced object.
    /// 
    /// FIXME: Docs do not line up with implementation.
    /// Actually returns the amount of weakreference objects.
    pub fn ref_weak_count(&self, py: Python<'_>) -> isize {
        match self.upgrade(py) {
            Some(obj) => obj.get_weak_refcnt(py),
            None => 0,
        }
    }

    /// Upgrades the `PyWeak<T>` to a `Py<T>`.
    pub fn upgrade(&self, py: Python<'_>) -> Option<Py<T>> {
        // Theory FIXME: This dows not work, since the pointer to PyNone is not null
        let object = NonNull::new(unsafe { ffi::PyWeakref_GetObject(self.as_ptr()) });

        match object {
            Some(ptr) => {
                // unsafe { ffi::Py_IncRef(ptr.as_ptr()) };
                Some(unsafe { Py::from_borrowed_ptr(py, ptr.as_ptr()) })
            }
            None => None,
        }
    }
}

impl<T> ToPyObject for PyWeak<T> {
    /// Converts `PyWeak` instance -> PyObject.
    fn to_object(&self, py: Python<'_>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl<T> IntoPy<PyObject> for PyWeak<T> {
    /// Converts a `PyWeak` instance to `PyObject`.
    /// Consumes `self` without calling `Py_DECREF()`.
    #[inline]
    fn into_py(self, py: Python<'_>) -> PyObject {
        // TODO: Feels a little over thing
        unsafe { PyObject::from_owned_ptr(py, self.as_ptr()) }
        // unsafe { PyObject::from_non_null(self.into_non_null()) }
    }
}

impl<T> AsPyPointer for PyWeak<T> {
    /// Gets the underlying FFI pointer, returns a borrowed pointer.
    #[inline]
    fn as_ptr(&self) -> *mut ffi::PyObject {
        self.0.as_ptr()
    }
}

impl<T> IntoPyPointer for PyWeak<T> {
    /// Gets the underlying FFI pointer, returns a owned pointer.
    #[inline]
    #[must_use]
    fn into_ptr(self) -> *mut ffi::PyObject {
        self.into_non_null().as_ptr()
    }
}

/// If the GIL is held this increments `self`'s reference count.
/// Otherwise this registers the [`PyWeak`]`<T>` instance to have its reference count
/// incremented the next time PyO3 acquires the GIL.
impl<T> Clone for PyWeak<T> {
    fn clone(&self) -> Self {
        unsafe {
            gil::register_incref(self.0.cast::<ffi::PyObject>());
        }
        Self(self.0, PhantomData)
    }
}

/// Dropping a `PyWeak` instance decrements the reference count on the object by 1.
impl<T> Drop for PyWeak<T> {
    fn drop(&mut self) {
        unsafe {
            gil::register_decref(self.0.cast::<ffi::PyObject>());
        }
    }
}

#[cfg(test)]
mod py_weak {
    use super::{Py, PyWeak, Python};
    use crate::intern;
    use crate::prelude::{pyclass, pymethods};

    #[pyclass(crate = "crate", weakref, set_all, get_all)]
    struct MyClass {
        name: String,
        age: i32,
    }

    #[pymethods(crate = "crate")]
    impl MyClass {
        #[new]
        fn py_new(name: String, age: Option<i32>) -> Self {
            Self {
                name,
                age: age.unwrap_or(-1),
            }
        }
    }

    #[test]
    fn test_weakref_cnt() {
        let strong: Py<MyClass> = Python::with_gil(|py| {
            Py::new(
                py,
                MyClass {
                    name: "Is the count increasing".to_string(),
                    age: -4,
                },
            )
            .unwrap()
        });

        Python::with_gil(|py| {
            let weak1: PyWeak<MyClass> = strong.downgrade(py);
            assert_eq!(strong.get_weak_refcnt(py), 1);
            assert_eq!(weak1.ref_weak_count(py), 1);

            let weak1b = strong.downgrade(py);
            assert_eq!(weak1b.ref_weak_count(py), 2);

            // This does not increase the weakcount of `strong`, since their is not a new weakref created
            let weak2 = weak1.clone_ref(py);
            assert_eq!(weak2.ref_weak_count(py), 2);
        })
    }

    #[test]
    fn test_py_weak_refcnt_increase_none_clone() {
        let class1: Py<MyClass> = Python::with_gil(|py| {
            Py::new(
                py,
                MyClass::py_new(String::from("PyO3 is the Best!!"), None),
            )
            .unwrap()
        });

        let weak_ref: PyWeak<MyClass> = Python::with_gil(|py| {
            assert_eq!(class1.get_refcnt(py), 1);
            let weak_1 = class1.downgrade(py);

            assert_eq!(class1.get_refcnt(py), 1);
            let class1_2 = weak_1.upgrade(py).unwrap();

            assert_eq!(class1.get_refcnt(py), 2);

            assert!(class1.is(&class1_2));

            weak_1
        });

        assert!(Python::with_gil(|py| {
            weak_ref.call0(py).unwrap().is(&class1)
        }));

        Python::with_gil(|py| {
            let obj_ref = weak_ref.upgrade(py).unwrap();
            assert_eq!(obj_ref.borrow(py).name, "PyO3 is the Best!!");
            assert_eq!(
                obj_ref
                    .getattr(py, intern!(py, "age"))
                    .unwrap()
                    .extract::<i32>(py)
                    .unwrap(),
                -1
            );

            assert_eq!(obj_ref.get_refcnt(py), 2);
        });

        Python::with_gil(|py| {
            class1
                .setattr(py, intern!(py, "name"), "Pythonium Trioxide")
                .unwrap();
            class1.borrow_mut(py).age = 2147483637;
        });

        Python::with_gil(|py| {
            assert_eq!(class1.get_refcnt(py), 1);

            let obj_ref = weak_ref.upgrade(py).unwrap();
            assert_eq!(
                obj_ref
                    .getattr(py, intern!(py, "name"))
                    .unwrap()
                    .extract::<&str>(py)
                    .unwrap(),
                "Pythonium Trioxide"
            );
            assert_eq!(
                obj_ref
                    .getattr(py, intern!(py, "age"))
                    .unwrap()
                    .extract::<i32>(py)
                    .unwrap(),
                2147483637
            );

            assert_eq!(obj_ref.get_refcnt(py), 2);
        });
    }

    #[test]
    fn test_seg_fault() {
        Python::with_gil(|py| {
            let first: Py<MyClass> = Py::new(
                py,
                MyClass {
                    name: "Joe".to_owned(),
                    age: 42,
                },
            )
            .unwrap();
            let second: Py<MyClass> = Py::clone_ref(&first, py);
            let first_weak: PyWeak<MyClass> = first.downgrade(py);

            // Both point to the same object
            assert!(first.is(&second));
            assert!(first_weak.upgrade(py).unwrap().is(&first));
            assert!(first_weak.upgrade(py).unwrap().is(&second));
            assert_eq!(first.get_refcnt(py), 2);
            assert_eq!(second.get_refcnt(py), 2);
            assert_eq!(first_weak.ref_weak_count(py), 1);
            assert_eq!(first_weak.ref_strong_count(py), 2);
            assert_eq!(first_weak.get_refcnt(py), 1);
            assert_eq!(first_weak.get_weak_refcnt(py), 0);
            assert_eq!(second.borrow(py).name, "Joe".to_owned());
            assert_eq!(second.borrow(py).age, 42);
        });
    }
}
