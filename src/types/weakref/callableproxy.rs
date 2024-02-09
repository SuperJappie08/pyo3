use crate::err::PyResult;
use crate::ffi_ptr_ext::FfiPtrExt;
use crate::types::any::PyAnyMethods;
use crate::{
    ffi, AsPyPointer, Borrowed, Bound, PyAny, PyNativeType, PyTypeCheck, Python, ToPyObject,
};

use super::PyWeakRefMethods;

/// Represents a Python `weakref.ProxyType`.
///
/// In Python this is created by calling `weakref.proxy`.
#[repr(transparent)]
pub struct PyWeakCallableProxy(PyAny);

pyobject_native_type!(
    PyWeakCallableProxy,
    ffi::PyWeakReference,
    pyobject_native_static_type_object!(ffi::_PyWeakref_CallableProxyType),
    #module=Some("weakref"),
    #checkfunction=ffi::PyWeakref_CheckProxy
);

impl PyWeakCallableProxy {
    /// Deprecated form of [`PyWeakCallableProxy::new_bound`].
    #[inline]
    #[track_caller]
    #[cfg_attr(
        not(feature = "gil-refs"),
        deprecated(
            since = "0.21.0",
            note = "`PyWeakCallableProxy::new` will be replaced by `PyWeakCallableProxy::new_bound` in a future PyO3 version"
        )
    )]
    pub fn new<T>(py: Python<'_>, object: T) -> PyResult<&'_ PyWeakCallableProxy>
    where
        T: ToPyObject,
    {
        Self::new_bound(py, object).map(Bound::into_gil_ref)
    }

    /// Constructs a new Weak callable Reference (`weakref.proxy`/`weakref.CallableProxyType`) for the given object.
    ///
    /// Returns a `TypeError` if `object` is not weak referenceable (Most native types and PyClasses without `weakref` flag).
    /// The object should also be callable. For a non-callable weakref proxy see [`PyWeakProxy`](crate::types::weakref::PyWeakProxy).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pyo3::prelude::*;
    /// use pyo3::types::PyWeakCallableProxy;
    /// use pyo3::exceptions::PyReferenceError;
    ///
    /// #[pyclass(weakref)]
    /// struct Foo { /* fields omitted */ }
    ///
    /// #[pymethods]
    /// impl Foo {
    ///     fn __call__(&self) -> &str {
    ///         "This class is callable"
    ///     }
    /// }
    ///
    /// # fn main() -> PyResult<()> {
    /// Python::with_gil(|py| {
    ///     let foo = Bound::new(py, Foo {})?;
    ///     let weakref = PyWeakCallableProxy::new_bound(py, foo.clone())?;
    ///     assert!(
    ///         // In normal situations where a direct `Bound<'py, Foo>` is required use `upgrade::<Foo>`
    ///         weakref.get_object()?
    ///             .is_some_and(|obj| obj.is(&foo))
    ///     );
    ///
    ///     let weakref2 = PyWeakCallableProxy::new_bound(py, foo.clone())?;
    ///     assert!(weakref.is(&weakref2));
    ///
    ///     assert_eq!(weakref.call0()?.to_string(), "This class is callable");
    ///
    ///     drop(foo);
    ///
    ///     assert!(weakref.get_object()?.is_none());
    ///     assert!(weakref.call0().is_err_and(|err| err.is_instance_of::<PyReferenceError>(py)));
    ///
    ///     Ok(())
    /// })
    /// # }
    /// ```
    ///
    /// # Panics
    /// This function panics if the provided object is not callable.
    ///
    #[inline]
    #[track_caller]
    pub fn new_bound<T>(py: Python<'_>, object: T) -> PyResult<Bound<'_, PyWeakCallableProxy>>
    where
        T: ToPyObject,
    {
        unsafe {
            let ptr = object.to_object(py).as_ptr();
            assert_eq!(
                ffi::PyCallable_Check(ptr), 1,
                "An object to be referenced by a PyWeakCallableProxy should be callable. Use PyWeakProxy instead."
            );

            Bound::from_owned_ptr_or_err(py, ffi::PyWeakref_NewProxy(ptr, ffi::Py_None()))
                .map(|obj| obj.downcast_into_unchecked())
        }
    }

    /// Deprecated form of [`PyWeakCallableProxy::new_bound_with`].
    #[inline]
    #[track_caller]
    #[cfg_attr(
        not(feature = "gil-refs"),
        deprecated(
            since = "0.21.0",
            note = "`PyWeakCallableProxy::new_with` will be replaced by `PyWeakCallableProxy::new_bound_with` in a future PyO3 version"
        )
    )]
    pub fn new_with<T, C>(
        py: Python<'_>,
        object: T,
        callback: C,
    ) -> PyResult<&'_ PyWeakCallableProxy>
    where
        T: ToPyObject,
        C: ToPyObject,
    {
        Self::new_bound_with(py, object, callback).map(Bound::into_gil_ref)
    }

    /// Constructs a new Weak Reference (`weakref.proxy`/`weakref.CallableProxyType`) for the given object with a callback.
    ///
    /// Returns a `TypeError` if `object` is not weak referenceable (Most native types and PyClasses without `weakref` flag) or if the `callback` is not callable or None.
    /// The object should also be callable. For a non-callable weakref proxy see [`PyWeakProxy`](crate::types::weakref::PyWeakProxy).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pyo3::prelude::*;
    /// use pyo3::types::PyWeakCallableProxy;
    ///
    /// #[pyclass(weakref)]
    /// struct Foo { /* fields omitted */ }
    ///
    /// #[pymethods]
    /// impl Foo {
    ///     fn __call__(&self) -> &str {
    ///         "This class is callable"
    ///     }
    /// }
    ///
    /// #[pyfunction]
    /// fn callback(wref: Bound<'_, PyWeakCallableProxy>) -> PyResult<()> {
    ///         let py = wref.py();
    ///         assert!(wref.upgrade::<Foo>()?.is_none());
    ///         py.run("counter = 1", None, None)
    /// }
    ///
    /// # fn main() -> PyResult<()> {
    /// Python::with_gil(|py| {
    ///     py.run("counter = 0", None, None)?;
    ///     assert_eq!(py.eval("counter", None, None)?.extract::<u32>()?, 0);
    ///     let foo = Bound::new(py, Foo{})?;
    ///
    ///     // This is fine.
    ///     let weakref = PyWeakCallableProxy::new_bound_with(py, foo.clone(), py.None())?;
    ///     assert!(weakref.upgrade::<Foo>()?.is_some());
    ///     assert!(
    ///         // In normal situations where a direct `Bound<'py, Foo>` is required use `upgrade::<Foo>`
    ///         weakref.get_object()?
    ///             .is_some_and(|obj| obj.is(&foo))
    ///     );
    ///     assert_eq!(py.eval("counter", None, None)?.extract::<u32>()?, 0);
    ///     
    ///     let weakref2 = PyWeakCallableProxy::new_bound_with(py, foo.clone(), wrap_pyfunction!(callback, py)?)?;
    ///     assert!(!weakref.is(&weakref2)); // Not the same weakref
    ///     assert!(weakref.eq(&weakref2)?);  // But Equal, since they point to the same object
    ///
    ///     assert_eq!(weakref.call0()?.to_string(), "This class is callable");
    ///
    ///     drop(foo);
    ///
    ///     assert!(weakref.upgrade::<Foo>()?.is_none());
    ///     assert_eq!(py.eval("counter", None, None)?.extract::<u32>()?, 1);
    ///     Ok(())
    /// })
    /// # }
    /// ```
    /// # Panics
    /// This function panics if the provided object is not callable.
    #[track_caller]
    pub fn new_bound_with<T, C>(
        py: Python<'_>,
        object: T,
        callback: C,
    ) -> PyResult<Bound<'_, PyWeakCallableProxy>>
    where
        T: ToPyObject,
        C: ToPyObject,
    {
        unsafe {
            let ptr = object.to_object(py).as_ptr();
            assert_eq!(
                ffi::PyCallable_Check(ptr), 1,
                "An object to be referenced by a PyWeakCallableProxy should be callable. Use PyWeakProxy instead."
            );

            Bound::from_owned_ptr_or_err(
                py,
                ffi::PyWeakref_NewProxy(ptr, callback.to_object(py).as_ptr()),
            )
            .map(|obj| obj.downcast_into_unchecked())
        }
    }

    /// Upgrade the weakref to a direct object reference.
    ///
    /// It is named `upgrade` to be inline with [rust's `Weak::upgrade`](std::rc::Weak::upgrade).
    /// In Python it would be equivalent to [`PyWeakref_GetObject`].
    ///
    /// # Example
    /// ```rust
    /// use pyo3::prelude::*;
    /// use pyo3::types::PyWeakCallableProxy;
    ///
    /// #[pyclass(weakref)]
    /// struct Foo { /* fields omitted */ }
    ///
    /// #[pymethods]
    /// impl Foo {
    ///     fn __call__(&self) -> &str {
    ///         "This class is callable"
    ///     }
    ///
    ///     fn get_data(&self) -> (&str, u32) {
    ///         ("Dave", 10)
    ///     }
    /// }
    ///
    /// fn parse_data(reference: Borrowed<'_, '_, PyWeakCallableProxy>) -> PyResult<String> {
    ///     if let Some(data_src) = reference.upgrade::<Foo>()? {
    ///         let data = data_src.borrow();
    ///         let (name, score) = data.get_data();
    ///         Ok(format!("Processing '{}': score = {}", name, score))
    ///     } else {
    ///         Ok("The supplied data reference is nolonger relavent.".to_owned())
    ///     }
    /// }
    ///
    /// # fn main() -> PyResult<()> {
    /// Python::with_gil(|py| {
    ///     let data = Bound::new(py, Foo{})?;
    ///     let reference = PyWeakCallableProxy::new_bound(py, data.clone())?;
    ///     
    ///     assert_eq!(
    ///         parse_data(reference.as_borrowed())?,
    ///         "Processing 'Dave': score = 10"
    ///     );
    ///     
    ///     assert_eq!(reference.call0()?.to_string(), "This class is callable");
    ///     
    ///     drop(data);
    ///
    ///     assert_eq!(
    ///         parse_data(reference.as_borrowed())?,
    ///         "The supplied data reference is nolonger relavent."
    ///     );
    ///
    ///     Ok(())
    /// })
    /// # }
    /// ```
    ///
    /// # Panics
    /// This function panics if the requested type `T` is not the the type of the instance refered to by this `weakref.CallableProxyType`.
    ///
    /// [`PyWeakref_GetObject`]: https://docs.python.org/3/c-api/weakref.html#c.PyWeakref_GetObject
    /// [`weakref.CallableProxyType`]: https://docs.python.org/3/library/weakref.html#weakref.CallableProxyType
    /// [`weakref.proxy`]: https://docs.python.org/3/library/weakref.html#weakref.proxy
    pub fn upgrade<T>(&self) -> PyResult<Option<&T::AsRefTarget>>
    where
        T: PyTypeCheck,
    {
        Ok(self.as_borrowed().upgrade::<T>()?.map(Bound::into_gil_ref))
    }

    /// Upgrade the weakref to a [`PyAny`] reference to the target if possible.
    ///
    /// This function returns `Some(&'py PyAny)` if the reference still exists, otherwise `None` will be returned.
    ///
    /// This function gets the optional target of this [`weakref.CallableProxyType`] (result of calling [`weakref.proxy`]).
    /// It produces similair results to using [`PyWeakref_GetObject`] in the C api.
    ///
    /// # Example
    /// ```rust
    /// use pyo3::prelude::*;
    /// use pyo3::types::PyWeakCallableProxy;
    ///
    /// #[pyclass(weakref)]
    /// struct Foo { /* fields omitted */ }
    ///
    /// #[pymethods]
    /// impl Foo {
    ///     fn __call__(&self) -> &str {
    ///         "This class is callable"
    ///     }
    /// }
    ///
    /// fn parse_data(reference: Borrowed<'_, '_, PyWeakCallableProxy>) -> PyResult<String> {
    ///     if let Some(object) = reference.get_object()? {
    ///         Ok(format!("The object '{}' refered by this reference still exists.", object.getattr("__class__")?.getattr("__qualname__")?))
    ///     } else {
    ///         Ok("The object, which this reference refered to, no longer exists".to_owned())
    ///     }
    /// }
    ///
    /// # fn main() -> PyResult<()> {
    /// Python::with_gil(|py| {
    ///     let data = Bound::new(py, Foo{})?;
    ///     let reference = PyWeakCallableProxy::new_bound(py, data.clone())?;
    ///     
    ///     assert_eq!(
    ///         parse_data(reference.as_borrowed())?,
    ///         "The object 'Foo' refered by this reference still exists."
    ///     );
    ///     
    ///     assert_eq!(reference.call0()?.to_string(), "This class is callable");
    ///
    ///     drop(data);
    ///
    ///     assert_eq!(
    ///         parse_data(reference.as_borrowed())?,
    ///         "The object, which this reference refered to, no longer exists"
    ///     );
    ///
    ///     Ok(())
    /// })
    /// # }
    /// ```
    ///
    /// [`PyWeakref_GetObject`]: https://docs.python.org/3/c-api/weakref.html#c.PyWeakref_GetObject
    /// [`weakref.ProxyCallableType`]: https://docs.python.org/3/library/weakref.html#weakref.CallableProxyType
    /// [`weakref.proxy`]: https://docs.python.org/3/library/weakref.html#weakref.proxy
    pub fn get_object(&self) -> PyResult<Option<&'_ PyAny>> {
        Ok(self.as_borrowed().get_object()?.map(Bound::into_gil_ref))
    }

    /// Retrieve to a object pointed to by the weakref.
    ///
    /// This function returns `&'py PyAny`, which is either the object if it still exists, otherwise it will refer to [`PyNone`](crate::types::none::PyNone).
    ///
    /// This function gets the optional target of this [`weakref.CallableProxyType`] (result of calling [`weakref.proxy`]).
    /// It produces similair results to using [`PyWeakref_GetObject`] in the C api.
    ///
    /// # Example
    /// ```rust
    /// use pyo3::prelude::*;
    /// use pyo3::types::PyWeakCallableProxy;
    ///
    /// #[pyclass(weakref)]
    /// struct Foo { /* fields omitted */ }
    ///
    /// #[pymethods]
    /// impl Foo {
    ///     fn __call__(&self) -> &str {
    ///         "This class is callable"
    ///     }
    /// }
    ///
    /// fn get_class(reference: Borrowed<'_, '_, PyWeakCallableProxy>) -> PyResult<String> {
    ///     reference
    ///         .get_object_raw()?
    ///         .getattr("__class__")?
    ///         .repr()?
    ///         .to_str()
    ///         .map(ToOwned::to_owned)
    /// }
    ///
    /// # fn main() -> PyResult<()> {
    /// Python::with_gil(|py| {
    ///     let object = Bound::new(py, Foo{})?;
    ///     let reference = PyWeakCallableProxy::new_bound(py, object.clone())?;
    ///     
    ///     assert_eq!(
    ///         get_class(reference.as_borrowed())?,
    ///         "<class 'builtins.Foo'>"
    ///     );
    ///
    ///     assert_eq!(reference.call0()?.to_string(), "This class is callable");
    ///
    ///     drop(object);
    ///
    ///     assert_eq!(get_class(reference.as_borrowed())?, "<class 'NoneType'>");
    ///
    ///     Ok(())
    /// })
    /// # }
    /// ```
    ///
    /// [`PyWeakref_GetObject`]: https://docs.python.org/3/c-api/weakref.html#c.PyWeakref_GetObject
    /// [`weakref.CallableProxyType`]: https://docs.python.org/3/library/weakref.html#weakref.CallableProxyType
    /// [`weakref.proxy`]: https://docs.python.org/3/library/weakref.html#weakref.proxy
    pub fn get_object_raw(&self) -> PyResult<&'_ PyAny> {
        self.as_borrowed().get_object_raw().map(Bound::into_gil_ref)
    }
}

impl<'py> PyWeakRefMethods<'py> for Bound<'py, PyWeakCallableProxy> {
    fn borrow_object_raw(&self) -> PyResult<Borrowed<'_, 'py, PyAny>> {
        unsafe { ffi::PyWeakref_GetObject(self.as_ptr()).assume_borrowed_or_err(self.py()) }
    }
}

#[cfg(test)]
mod tests {
    use crate::exceptions::{PyAttributeError, PyReferenceError};
    use crate::prelude::{pyclass, pymethods, Py, PyResult, Python};
    use crate::types::any::PyAnyMethods;
    use crate::types::weakref::{PyWeakCallableProxy, PyWeakRefMethods};
    use crate::Bound;

    #[pyclass(weakref, crate = "crate")]
    struct WeakrefablePyClass {}

    #[pymethods(crate = "crate")]
    impl WeakrefablePyClass {
        fn __call__(&self) -> &str {
            "This class is callable!"
        }
    }

    #[test]
    fn test_weakref_proxy_behavior() -> PyResult<()> {
        Python::with_gil(|py| {
            let object = Bound::new(py, WeakrefablePyClass {})?;
            let reference = PyWeakCallableProxy::new_bound(py, object.clone())?;

            assert!(!reference.is(&object));
            assert!(reference.get_object_raw()?.is(&object));
            assert_eq!(
                reference.get_type().to_string(),
                "<class 'weakref.CallableProxyType'>"
            );

            assert_eq!(
                reference.getattr("__class__")?.to_string(),
                "<class 'builtins.WeakrefablePyClass'>"
            );
            assert_eq!(
                reference.repr()?.to_string(),
                format!(
                    "<weakproxy at {:x?} to builtins.WeakrefablePyClass at {:x?}>",
                    reference.as_ptr(),
                    object.as_ptr()
                )
            );

            assert!(reference
                .getattr("__callback__")
                .is_err_and(|err| err.is_instance_of::<PyAttributeError>(py)));

            assert_eq!(reference.call0()?.to_string(), "This class is callable!");

            drop(object);

            assert!(reference.get_object_raw()?.is_none());
            assert!(reference
                .getattr("__class__")
                .is_err_and(|err| err.is_instance_of::<PyReferenceError>(py)));
            assert_eq!(
                reference.repr()?.to_string(),
                format!(
                    "<weakproxy at {:x?} to NoneType at {:x?}>",
                    reference.as_ptr(),
                    py.None().as_ptr()
                )
            );

            assert!(reference
                .getattr("__callback__")
                .is_err_and(|err| err.is_instance_of::<PyReferenceError>(py)));

            assert!(reference
                .call0()
                .is_err_and(|err| err.is_instance_of::<PyReferenceError>(py)
                    & (err.value(py).to_string() == "weakly-referenced object no longer exists")));

            Ok(())
        })
    }

    #[test]
    fn test_weakref_upgrade() -> PyResult<()> {
        Python::with_gil(|py| {
            let object = Py::new(py, WeakrefablePyClass {})?;
            let reference = PyWeakCallableProxy::new_bound(py, object.clone_ref(py))?;

            {
                let obj = reference.upgrade::<WeakrefablePyClass>();

                assert!(obj.is_ok());
                let obj = obj.unwrap();

                assert!(obj.is_some());
                assert!(obj.is_some_and(|obj| obj.as_ptr() == object.as_ptr()));
            }

            drop(object);

            {
                let obj = reference.upgrade::<WeakrefablePyClass>();

                assert!(obj.is_ok());
                let obj = obj.unwrap();

                assert!(obj.is_none());
            }

            Ok(())
        })
    }

    #[test]
    fn test_weakref_get_object() -> PyResult<()> {
        Python::with_gil(|py| {
            let object = Py::new(py, WeakrefablePyClass {})?;
            let reference = PyWeakCallableProxy::new_bound(py, object.clone_ref(py))?;

            assert!(reference.get_object()?.is_some());
            assert!(reference.get_object()?.is_some_and(|obj| obj.is(&object)));

            drop(object);

            assert!(reference.get_object()?.is_none());

            Ok(())
        })
    }

    #[test]
    fn test_weakref_borrrow_object() -> PyResult<()> {
        Python::with_gil(|py| {
            let object = Py::new(py, WeakrefablePyClass {})?;
            let reference = PyWeakCallableProxy::new_bound(py, object.clone_ref(py))?;

            assert!(reference.borrow_object()?.is_some());
            assert!(reference
                .borrow_object()?
                .is_some_and(|obj| obj.is(&object)));

            drop(object);

            assert!(reference.borrow_object()?.is_none());

            Ok(())
        })
    }

    #[test]
    fn test_weakref_get_object_raw() -> PyResult<()> {
        Python::with_gil(|py| {
            let object = Py::new(py, WeakrefablePyClass {})?;
            let reference = PyWeakCallableProxy::new_bound(py, object.clone_ref(py))?;

            assert!(reference.get_object_raw()?.is(&object));

            drop(object);

            assert!(reference.get_object_raw()?.is_none());

            Ok(())
        })
    }

    #[test]
    fn test_weakref_borrow_object_raw() -> PyResult<()> {
        Python::with_gil(|py| {
            let object = Py::new(py, WeakrefablePyClass {})?;
            let reference = PyWeakCallableProxy::new_bound(py, object.clone_ref(py))?;

            assert!(reference.borrow_object_raw()?.is(&object));

            drop(object);

            assert!(reference.borrow_object_raw()?.is_none());

            Ok(())
        })
    }
}
