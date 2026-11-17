//! Single-threaded uint8 sampling. Borrowed inputs remain under the Python GIL.
//! Runtime-dispatched SIMD is confined to prepared.rs; no fast-math or OpenCV code.

mod prepared;

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{
    Element, IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

trait Coordinate: Element + Copy {
    fn finite(self) -> bool;
    fn shifted(self, bound: usize, wrap: bool) -> f64;
    fn nearest(value: f64) -> i64;
}

macro_rules! coordinate {
    ($t:ty) => {
        impl Coordinate for $t {
            fn finite(self) -> bool {
                self.is_finite()
            }
            fn shifted(self, bound: usize, wrap: bool) -> f64 {
                // Match NumPy's arithmetic in the coordinate dtype, including
                // the half-pixel subtraction, before computing weights in f32.
                let n = bound as $t;
                let value = if wrap && !(self >= 0.0 && self < n) {
                    let remainder = self % n;
                    if remainder < 0.0 {
                        remainder + n
                    } else {
                        remainder
                    }
                } else {
                    self.clamp(0.0, n)
                };
                (value - 0.5) as f64
            }
            fn nearest(value: f64) -> i64 {
                // A shifted coordinate is at least -0.5, so truncating the
                // restored nonnegative value is equivalent to flooring it.
                (value as $t + 0.5) as i64
            }
        }
    };
}
coordinate!(f32);
coordinate!(f64);

fn lower_tap(value: f64) -> i64 {
    // Shifted values are never below -0.5.
    if value < 0.0 {
        -1
    } else {
        value as i64
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sample_sse41<C: Coordinate>(
    pixels: &[u8],
    coords: &[C],
    mask: Option<&[u8]>,
    shape: (usize, usize, usize),
    wrap: bool,
    nearest: bool,
    output: &mut [u8],
) {
    // Inlining gives the compiler a hardware ties-to-even rounding instruction.
    sample_pixels(pixels, coords, mask, shape, wrap, nearest, output);
}

#[inline(always)]
fn sample_pixels<C: Coordinate>(
    pixels: &[u8],
    coords: &[C],
    mask: Option<&[u8]>,
    shape: (usize, usize, usize),
    wrap: bool,
    nearest: bool,
    output: &mut [u8],
) {
    let (height, width, channels) = shape;
    let clamp_y = |y: i64| y.clamp(0, height as i64 - 1) as usize;
    let index_x = |x: i64| {
        if wrap {
            // shifted() already reduced the coordinate to one revolution.
            // Only a neighboring interpolation tap can cross its boundary.
            if x < 0 {
                (x + width as i64) as usize
            } else if x >= width as i64 {
                (x - width as i64) as usize
            } else {
                x as usize
            }
        } else {
            x.clamp(0, width as i64 - 1) as usize
        }
    };
    for (i, dst) in output.chunks_exact_mut(channels).enumerate() {
        let (cx, cy) = (coords[2 * i], coords[2 * i + 1]);
        if !cx.finite() || !cy.finite() || mask.is_some_and(|m| m[i] == 0) {
            continue;
        }
        let x = cx.shifted(width, wrap);
        let y = cy.shifted(height, false);
        if nearest {
            let index = (clamp_y(C::nearest(y)) * width + index_x(C::nearest(x))) * channels;
            dst.copy_from_slice(&pixels[index..index + channels]);
            continue;
        }
        let (x0, y0) = (lower_tap(x), lower_tap(y));
        let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
        let (left, right) = (index_x(x0), index_x(x0 + 1));
        let (top, bottom) = (clamp_y(y0) * width, clamp_y(y0 + 1) * width);
        let (a, b, c, d) = (
            (top + left) * channels,
            (top + right) * channels,
            (bottom + left) * channels,
            (bottom + right) * channels,
        );
        for (ch, value) in dst.iter_mut().enumerate() {
            let upper = pixels[a + ch] as f32 * (1.0 - fx) + pixels[b + ch] as f32 * fx;
            let lower = pixels[c + ch] as f32 * (1.0 - fx) + pixels[d + ch] as f32 * fx;
            let blended = upper * (1.0 - fy) + lower * fy;
            *value = blended.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
}

fn sample<'py, C: Coordinate>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    xy: PyReadonlyArrayDyn<'py, C>,
    valid: Option<PyReadonlyArrayDyn<'py, u8>>,
    wrap: bool,
    nearest: bool,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    let shape = image.shape();
    if !(shape.len() == 2 || shape.len() == 3) || shape.contains(&0) {
        return Err(PyValueError::new_err("Expected a nonempty HW or HWC image"));
    }
    let map_shape = xy.shape();
    if map_shape.last() != Some(&2) {
        return Err(PyValueError::new_err(
            "Coordinates must have shape (..., 2)",
        ));
    }
    if !image.is_c_contiguous() || !xy.is_c_contiguous() {
        return Err(PyValueError::new_err("Native inputs must be C-contiguous"));
    }
    if !xy.data().is_aligned() {
        return Err(PyValueError::new_err("Native coordinates must be aligned"));
    }
    if let Some(ref mask) = valid {
        if mask.shape() != &map_shape[..map_shape.len() - 1] || !mask.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "Mask must match the coordinate shape and be C-contiguous",
            ));
        }
    }
    let pixels = image.as_slice()?;
    let coords = xy.as_slice()?;
    let mask = valid.as_ref().map(|m| m.as_slice()).transpose()?;
    let (height, width) = (shape[0], shape[1]);
    let channels = if shape.len() == 3 { shape[2] } else { 1 };
    let count = coords.len() / 2;
    let length = count
        .checked_mul(channels)
        .filter(|&length| length <= isize::MAX as usize)
        .ok_or_else(|| PyValueError::new_err("Output size overflow"))?;
    let mut output = vec![0u8; length];
    let image_shape = (height, width, channels);
    #[cfg(target_arch = "x86_64")]
    let sampled = if !nearest && std::is_x86_feature_detected!("sse4.1") {
        // SAFETY: The runtime check guarantees the target feature is available.
        unsafe {
            sample_sse41(
                pixels,
                coords,
                mask,
                image_shape,
                wrap,
                nearest,
                &mut output,
            )
        };
        true
    } else {
        false
    };
    #[cfg(not(target_arch = "x86_64"))]
    let sampled = false;
    if !sampled {
        sample_pixels(
            pixels,
            coords,
            mask,
            image_shape,
            wrap,
            nearest,
            &mut output,
        );
    }
    let mut output_shape = map_shape[..map_shape.len() - 1].to_vec();
    if shape.len() == 3 {
        output_shape.push(channels);
    }
    let array = ArrayD::from_shape_vec(IxDyn(&output_shape), output)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (image, xy, valid=None, wrap=false, nearest=false))]
fn sample_u8_f32<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    xy: PyReadonlyArrayDyn<'py, f32>,
    valid: Option<PyReadonlyArrayDyn<'py, u8>>,
    wrap: bool,
    nearest: bool,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    sample(py, image, xy, valid, wrap, nearest)
}

#[pyfunction]
#[pyo3(signature = (image, xy, valid=None, wrap=false, nearest=false))]
fn sample_u8_f64<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    xy: PyReadonlyArrayDyn<'py, f64>,
    valid: Option<PyReadonlyArrayDyn<'py, u8>>,
    wrap: bool,
    nearest: bool,
) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
    sample(py, image, xy, valid, wrap, nearest)
}

#[pymodule(gil_used = true)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_u8_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sample_u8_f64, m)?)?;
    m.add_class::<prepared::PreparedBilinear>()?;
    m.add_function(wrap_pyfunction!(prepared::avx2_available, m)?)?;
    m.add_function(wrap_pyfunction!(prepared::fused_coordinates, m)?)?;
    Ok(())
}
