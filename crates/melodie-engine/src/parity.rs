//! Parity-testing harness: compare Rust tensors against "golden" tensors dumped
//! from the Python MLX reference (`../heartlib-mlx`). Each port phase is locked to
//! a numeric tolerance (~1e-4) against these goldens before moving on.

use crate::{EngineError, Result};
use candle_core::Tensor;

/// Max absolute element-wise difference between two same-shaped tensors.
pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = (a - b)?.abs()?;
    let v = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    Ok(v)
}

/// Assert two tensors match within `tol`; otherwise return a parity error.
pub fn assert_close(a: &Tensor, b: &Tensor, tol: f32) -> Result<()> {
    let d = max_abs_diff(a, b)?;
    if d > tol {
        return Err(EngineError::Config(format!(
            "parity failed: max|Δ| = {d:.3e} > tol {tol:.3e}"
        )));
    }
    Ok(())
}
