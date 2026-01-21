use std::fmt::Display;

/// General error type for any kind of error appearing during KDE calculation.
///
/// `message` carries a (hopefully) helpful message as to why the error occurred, `kind` contains an [`ErrorKind`]
/// with the specific error type.
#[derive(Debug)]
pub struct KDEError {
    pub message: String,
    pub kind: ErrorKind,
}

impl KDEError {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        return Self {
            message: message.into(),
            kind,
        };
    }
}

impl Display for KDEError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum ErrorKind {
    /// Unexpected shape of `Array`
    ShapeError,
    /// Index out of bounds
    IndexError,
    /// Singular (or non-positive-definite) covariance matrix
    SingularityError,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::ShapeError => write!(f, "ShapeError"),
            ErrorKind::IndexError => write!(f, "IndexError"),
            ErrorKind::SingularityError => write!(f, "SingularityError"),
        }
    }
}
