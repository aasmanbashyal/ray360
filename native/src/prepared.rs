//! Owned interpolation plans and runtime-dispatched AVX2 uint8 sampling.
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[pyclass(frozen)]
pub struct PreparedBilinear {
    width: usize,
    height: usize,
    shape: Vec<usize>,
    count: usize,
    packed: Box<[PackedBlock]>,
    wide: Box<[WideBlock]>,
}

const VALID: u32 = 1;
const SAME_X: u32 = 2;
const WRAP_X: u32 = 4;
const SAME_Y: u32 = 8;
const ALL_INTERIOR: u32 = 0x1111_1111;
const ALL_INVALID: u32 = 0xaaaa_aaaa; // Zero weights/base; both neighbor offsets zero.
const NEAR_END: usize = 1;
const FRACTION_SCALE: f32 = 1. / 65536.;

#[repr(C)]
struct Header {
    output: usize, // Starts on an eight-pixel boundary; bit 0 flags source-end reads.
    origin: u32,
    flags: u32, // One nibble per lane: validity and exact border behavior.
}

// Exactly one cache line per eight lanes on both 32-bit and 64-bit targets.
// Narrowing is used only after exact round-trip checks, never quantization.
#[repr(C, align(64))]
struct PackedBlock {
    header: Header,
    offsets: [u16; 8],
    fractions: [u32; 8],
}

#[repr(C)]
struct WideBlock {
    header: Header,
    base: [u32; 8],
    fx: [f32; 8],
    fy: [f32; 8],
}

#[derive(Clone, Copy)]
enum BlockRef<'a> {
    Packed(&'a PackedBlock),
    Wide(&'a WideBlock),
}

impl BlockRef<'_> {
    fn header(&self) -> &Header {
        match self {
            Self::Packed(block) => &block.header,
            Self::Wide(block) => &block.header,
        }
    }

    fn lane(self, lane: usize) -> (usize, f32, f32) {
        match self {
            Self::Packed(block) => {
                let weights = block.fractions[lane];
                (
                    block.header.origin as usize + block.offsets[lane] as usize,
                    (weights & 65535) as f32 * FRACTION_SCALE,
                    (weights >> 16) as f32 * FRACTION_SCALE,
                )
            }
            Self::Wide(block) => (block.base[lane] as usize, block.fx[lane], block.fy[lane]),
        }
    }
}

fn exact_fraction(value: f32) -> Option<u16> {
    let packed = (value * 65536.) as u16;
    ((packed as f32 * FRACTION_SCALE).to_bits() == value.to_bits()).then_some(packed)
}

#[pymethods]
impl PreparedBilinear {
    #[new]
    #[pyo3(signature = (width, height, xy, valid=None, wrap=true))]
    fn new(
        width: usize,
        height: usize,
        xy: PyReadonlyArrayDyn<'_, f32>,
        valid: Option<PyReadonlyArrayDyn<'_, u8>>,
        wrap: bool,
    ) -> PyResult<Self> {
        let source_pixels = width
            .checked_mul(height)
            .filter(|&n| n > 0 && n <= u32::MAX as usize)
            .ok_or_else(|| {
                PyValueError::new_err("Source dimensions must be positive and fit u32 indices")
            })?;
        if xy.shape().last() != Some(&2) || !xy.is_c_contiguous() || !xy.data().is_aligned() {
            return Err(PyValueError::new_err(
                "Coordinates must be aligned, contiguous float32 (..., 2)",
            ));
        }
        let shape = xy.shape()[..xy.ndim() - 1].to_vec();
        if let Some(ref mask) = valid {
            if mask.shape() != shape || !mask.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "Mask must match coordinates and be contiguous",
                ));
            }
        }
        let coords = xy.as_slice()?;
        let mask = valid.as_ref().map(|m| m.as_slice()).transpose()?;
        let count = coords.len() / 2;
        let mut packed = Vec::new();
        let mut wide = Vec::new();
        for start in (0..count).step_by(8) {
            let mut header = Header {
                output: start,
                origin: u32::MAX,
                flags: ALL_INVALID,
            };
            let mut max_index = 0;
            let mut bases = [0u32; 8];
            let mut weights_x = [0f32; 8];
            let mut weights_y = [0f32; 8];
            let mut any_valid = false;
            for lane in 0..(count - start).min(8) {
                let i = start + lane;
                let (cx, cy) = (coords[i * 2], coords[i * 2 + 1]);
                if !cx.is_finite() || !cy.is_finite() || mask.is_some_and(|m| m[i] == 0) {
                    continue;
                }
                let x = <f32 as super::Coordinate>::shifted(cx, width, wrap);
                let y = <f32 as super::Coordinate>::shifted(cy, height, false);
                let (ix, iy) = (x.floor() as i64, y.floor() as i64);
                let col = |v: i64| {
                    if wrap {
                        v.rem_euclid(width as i64) as usize
                    } else {
                        v.clamp(0, width as i64 - 1) as usize
                    }
                };
                let row = |v: i64| v.clamp(0, height as i64 - 1) as usize;
                let indices = [
                    row(iy) * width + col(ix),
                    row(iy) * width + col(ix + 1),
                    row(iy + 1) * width + col(ix),
                    row(iy + 1) * width + col(ix + 1),
                ];
                let max = *indices.iter().max().unwrap();
                // Validate before narrowing; owned indices cannot be mutated from Python.
                if max >= source_pixels {
                    return Err(PyValueError::new_err("Invalid source index"));
                }
                let mut flags = VALID;
                if indices[1] == indices[0] {
                    flags |= SAME_X;
                } else if indices[1] < indices[0] {
                    flags |= WRAP_X;
                }
                if indices[2] == indices[0] {
                    flags |= SAME_Y;
                }
                header.flags = (header.flags & !(15 << (lane * 4))) | (flags << (lane * 4));
                header.origin = header.origin.min(indices[0] as u32);
                max_index = max_index.max(max);
                weights_x[lane] = (x - ix as f64) as f32;
                weights_y[lane] = (y - iy as f64) as f32;
                // The other three indices follow from the base and border flags.
                // Keep all weights unquantized, including at seams and clamped edges.
                bases[lane] = indices[0] as u32;
                any_valid = true;
            }
            if any_valid {
                if max_index >= source_pixels.saturating_sub(3) {
                    header.output |= NEAR_END;
                }
                let mut offsets = [0u16; 8];
                let mut fractions = [0u32; 8];
                let mut fits = true;
                for lane in 0..8 {
                    if header.flags & (VALID << (lane * 4)) == 0 {
                        continue;
                    }
                    match (
                        u16::try_from(bases[lane] - header.origin),
                        exact_fraction(weights_x[lane]),
                        exact_fraction(weights_y[lane]),
                    ) {
                        (Ok(offset), Some(x), Some(y)) => {
                            offsets[lane] = offset;
                            fractions[lane] = x as u32 | ((y as u32) << 16);
                        }
                        _ => {
                            fits = false;
                            break;
                        }
                    }
                }
                if fits {
                    packed.push(PackedBlock {
                        header,
                        offsets,
                        fractions,
                    });
                } else {
                    wide.push(WideBlock {
                        header,
                        base: bases,
                        fx: weights_x,
                        fy: weights_y,
                    });
                }
            }
        }
        Ok(Self {
            width,
            height,
            shape,
            count,
            packed: packed.into_boxed_slice(),
            wide: wide.into_boxed_slice(),
        })
    }

    #[getter]
    fn nbytes(&self) -> usize {
        // Boxed slices have no spare Vec capacity. Small object/shape headers excluded.
        std::mem::size_of_val(&*self.packed) + std::mem::size_of_val(&*self.wide)
    }

    #[getter]
    fn packed_blocks(&self) -> usize {
        self.packed.len()
    }

    #[getter]
    fn wide_blocks(&self) -> usize {
        self.wide.len()
    }

    #[pyo3(signature = (image, engine="auto"))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'py, u8>,
        engine: &str,
    ) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        if !matches!(engine, "auto" | "scalar" | "avx2") {
            return Err(PyValueError::new_err(
                "engine must be auto, scalar, or avx2",
            ));
        }
        let dims = image.shape();
        if !(dims.len() == 2 || dims.len() == 3)
            || dims.contains(&0)
            || dims[..2] != [self.height, self.width]
            || !image.is_c_contiguous()
        {
            return Err(PyValueError::new_err(
                "Image must be contiguous HW/HWC with the prepared source size",
            ));
        }
        let channels = if dims.len() == 3 { dims[2] } else { 1 };
        let pixels = image.as_slice()?;
        let can_simd =
            avx2_available() && matches!(channels, 1 | 3 | 4) && pixels.len() <= i32::MAX as usize;
        if engine == "avx2" && !can_simd {
            return Err(PyValueError::new_err(
                "AVX2 requires a supported CPU, 1/3/4 channels, and <=2 GiB input",
            ));
        }
        let length = self
            .count
            .checked_mul(channels)
            .ok_or_else(|| PyValueError::new_err("Output size overflow"))?;
        let mut output = vec![0u8; length];
        if engine != "scalar" && can_simd {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: CPU detection and signed gather-offset range checked above.
            // The SIMD routine checks every block's final 4-byte source load.
            unsafe {
                match channels {
                    1 => self.avx2::<1>(pixels, &mut output),
                    3 => self.avx2::<3>(pixels, &mut output),
                    4 => self.avx2::<4>(pixels, &mut output),
                    _ => unreachable!(),
                }
            }
        } else {
            for block in self.blocks() {
                self.scalar(pixels, &mut output, channels, block);
            }
        }
        let mut shape = self.shape.clone();
        if dims.len() == 3 {
            shape.push(channels);
        }
        Ok(ArrayD::from_shape_vec(IxDyn(&shape), output)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }
}

impl PreparedBilinear {
    fn blocks(&self) -> impl Iterator<Item = BlockRef<'_>> {
        self.packed
            .iter()
            .map(BlockRef::Packed)
            .chain(self.wide.iter().map(BlockRef::Wide))
    }

    fn scalar(&self, pixels: &[u8], out: &mut [u8], channels: usize, block: BlockRef<'_>) {
        let header = block.header();
        let output = header.output & !7;
        for lane in 0..8 {
            let flags = (header.flags >> (lane * 4)) & 15;
            if flags & VALID == 0 {
                continue;
            }
            let (a, fx, fy) = block.lane(lane);
            let b = if flags & SAME_X != 0 {
                a
            } else if flags & WRAP_X != 0 {
                a + 1 - self.width
            } else {
                a + 1
            };
            let dy = if flags & SAME_Y != 0 { 0 } else { self.width };
            let indices = [a, b, a + dy, b + dy].map(|index| index * channels);
            for ch in 0..channels {
                let upper = pixels[indices[0] + ch] as f32 * (1. - fx)
                    + pixels[indices[1] + ch] as f32 * fx;
                let lower = pixels[indices[2] + ch] as f32 * (1. - fx)
                    + pixels[indices[3] + ch] as f32 * fx;
                out[(output + lane) * channels + ch] = (upper * (1. - fy) + lower * fy)
                    .round_ties_even()
                    .clamp(0., 255.) as u8;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2<const C: usize>(&self, pixels: &[u8], out: &mut [u8]) {
        use std::arch::x86_64::*;
        // Every owned block contains eight lanes, including the padded tail.
        // Disabled lanes point to zero or the block origin with zero neighbor offsets;
        // only valid lanes can write output, and their output positions were
        // checked during construction. Full 8-lane loads stay inside each block.
        // Unaligned intrinsics are intentional. Gathers may read four bytes for RGB/grayscale pixels:
        // blocks touching the end of the source use the bounds-checked scalar path.
        let one = _mm256_set1_ps(1.);
        let byte_mask = _mm256_set1_epi32(255);
        let column = _mm256_set1_epi32(1);
        let row = _mm256_set1_epi32(self.width as i32);
        let zero = _mm256_setzero_si256();
        for (block_index, block) in self.blocks().enumerate() {
            // Hint two upcoming source pairs before the gathers need them.
            // Interior blocks have validated a+width neighbors. Keep every
            // pointer within the source allocation, even for prefetch hints.
            if let Some(next) = self.packed.get(block_index + 2) {
                if next.header.flags == ALL_INTERIOR {
                    for lane in [0, 4] {
                        let index = next.header.origin as usize + next.offsets[lane] as usize;
                        _mm_prefetch::<_MM_HINT_T0>(pixels.as_ptr().add(index * C).cast());
                        _mm_prefetch::<_MM_HINT_T0>(
                            pixels.as_ptr().add((index + self.width) * C).cast(),
                        );
                    }
                }
            }
            let header = block.header();
            let output = header.output & !7;
            if C != 4 && header.output & NEAR_END != 0 {
                self.scalar(pixels, out, C, block);
                continue;
            }
            let source = pixels.as_ptr().cast::<i32>();
            let (a, fx, fy) = match block {
                BlockRef::Packed(b) => {
                    let delta = _mm256_cvtepu16_epi32(_mm_loadu_si128(b.offsets.as_ptr().cast()));
                    let fractions = _mm256_loadu_si256(b.fractions.as_ptr().cast());
                    let scale = _mm256_set1_ps(FRACTION_SCALE);
                    (
                        _mm256_add_epi32(_mm256_set1_epi32(header.origin as i32), delta),
                        _mm256_mul_ps(
                            _mm256_cvtepi32_ps(_mm256_and_si256(
                                fractions,
                                _mm256_set1_epi32(65535),
                            )),
                            scale,
                        ),
                        _mm256_mul_ps(
                            _mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(fractions)),
                            scale,
                        ),
                    )
                }
                BlockRef::Wide(b) => (
                    _mm256_loadu_si256(b.base.as_ptr().cast()),
                    _mm256_loadu_ps(b.fx.as_ptr()),
                    _mm256_loadu_ps(b.fy.as_ptr()),
                ),
            };
            let (dx, dy) = if header.flags == ALL_INTERIOR {
                (column, row)
            } else {
                let flags = _mm256_srlv_epi32(
                    _mm256_set1_epi32(header.flags as i32),
                    _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28),
                );
                let absent =
                    |bit| _mm256_cmpeq_epi32(_mm256_and_si256(flags, _mm256_set1_epi32(bit)), zero);
                let dx = _mm256_sub_epi32(
                    _mm256_and_si256(absent(SAME_X as i32), column),
                    _mm256_andnot_si256(absent(WRAP_X as i32), row),
                );
                (dx, _mm256_and_si256(absent(SAME_Y as i32), row))
            };
            let b = _mm256_add_epi32(a, dx);
            let c = _mm256_add_epi32(a, dy);
            let d = _mm256_add_epi32(b, dy);
            let gather = |idx| {
                _mm256_i32gather_epi32::<1>(
                    source,
                    _mm256_mullo_epi32(idx, _mm256_set1_epi32(C as i32)),
                )
            };
            let (a, b, c, d) = (gather(a), gather(b), gather(c), gather(d));
            let rx = _mm256_sub_ps(one, fx);
            let ry = _mm256_sub_ps(one, fy);
            let mut packed = _mm256_setzero_si256();
            for ch in 0..C {
                let shift = _mm256_set1_epi32((ch * 8) as i32);
                let unpack = |v| {
                    _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(v, shift), byte_mask))
                };
                // Separate mul/add instructions deliberately avoid FMA rounding differences.
                let upper =
                    _mm256_add_ps(_mm256_mul_ps(unpack(a), rx), _mm256_mul_ps(unpack(b), fx));
                let lower =
                    _mm256_add_ps(_mm256_mul_ps(unpack(c), rx), _mm256_mul_ps(unpack(d), fx));
                let value = _mm256_add_ps(_mm256_mul_ps(upper, ry), _mm256_mul_ps(lower, fy));
                let rounded =
                    _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(value);
                let integer = _mm256_min_epi32(
                    byte_mask,
                    _mm256_max_epi32(_mm256_setzero_si256(), _mm256_cvttps_epi32(rounded)),
                );
                packed = _mm256_or_si256(packed, _mm256_sllv_epi32(integer, shift));
            }
            let mut words = [0u32; 8];
            _mm256_storeu_si256(words.as_mut_ptr().cast(), packed);
            if header.flags & ALL_INTERIOR == ALL_INTERIOR {
                // One output-span bounds check, with no per-lane validity branch.
                let target = &mut out[output * C..(output + 8) * C];
                for (pixel, word) in target.chunks_exact_mut(C).zip(words) {
                    pixel.copy_from_slice(&word.to_le_bytes()[..C]);
                }
            } else {
                for (lane, word) in words.iter().enumerate() {
                    if header.flags & (VALID << (lane * 4)) != 0 {
                        out[(output + lane) * C..(output + lane + 1) * C]
                            .copy_from_slice(&word.to_le_bytes()[..C]);
                    }
                }
            }
        }
    }
}

type Coordinates<'py> = (Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<u8>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fused_coordinates<'py>(
    py: Python<'py>,
    source_size: (usize, usize),
    view_size: (usize, usize),
    output_size: (usize, usize),
    focal: f64,
    rotation: Vec<f64>,
    scale: (f64, f64),
    offset: (f64, f64),
    bounds: (f64, f64, f64, f64),
) -> PyResult<Coordinates<'py>> {
    let (sw, sh) = source_size;
    let (vw, vh) = view_size;
    let (ow, oh) = output_size;
    let values = [
        focal, scale.0, scale.1, offset.0, offset.1, bounds.0, bounds.1, bounds.2, bounds.3,
    ];
    if [sw, sh, vw, vh, ow, oh].contains(&0)
        || !values.iter().all(|x| x.is_finite())
        || focal <= 0.
        || scale.0 <= 0.
        || scale.1 <= 0.
        || rotation.len() != 9
        || !rotation.iter().all(|x| x.is_finite())
        || bounds.0 < 0.
        || bounds.1 < 0.
        || bounds.2 > vw as f64
        || bounds.3 > vh as f64
        || bounds.0 > bounds.2
        || bounds.1 > bounds.3
    {
        return Err(PyValueError::new_err("Invalid fused geometry"));
    }
    let count = ow
        .checked_mul(oh)
        .filter(|&n| n <= isize::MAX as usize / 8)
        .ok_or_else(|| PyValueError::new_err("Output size overflow"))?;
    let mut xy = vec![0f32; count * 2];
    let mut valid = vec![0u8; count];
    for i in 0..count {
        let x = ((i % ow) as f64 + 0.5 - offset.0) / scale.0;
        let y = ((i / ow) as f64 + 0.5 - offset.1) / scale.1;
        if !x.is_finite()
            || !y.is_finite()
            || x < bounds.0 - 1e-8
            || x > bounds.2 + 1e-8
            || y < bounds.1 - 1e-8
            || y > bounds.3 + 1e-8
            || bounds.0 == bounds.2
            || bounds.1 == bounds.3
        {
            continue;
        }
        let lx = (x - vw as f64 / 2.) / focal;
        let ly = (vh as f64 / 2. - y) / focal;
        let rx = lx * rotation[0] + ly * rotation[3] + rotation[6];
        let ry = lx * rotation[1] + ly * rotation[4] + rotation[7];
        let rz = lx * rotation[2] + ly * rotation[5] + rotation[8];
        let longitude = (rx.atan2(rz).to_degrees() + 180.).rem_euclid(360.) - 180.;
        let latitude = ry.atan2(rx.hypot(rz)).to_degrees();
        xy[i * 2] = ((longitude / 360. + 0.5) * sw as f64) as f32;
        xy[i * 2 + 1] = ((0.5 - latitude / 180.) * sh as f64) as f32;
        valid[i] = 1;
    }
    Ok((
        ArrayD::from_shape_vec(IxDyn(&[oh, ow, 2]), xy)
            .unwrap()
            .into_pyarray(py),
        ArrayD::from_shape_vec(IxDyn(&[oh, ow]), valid)
            .unwrap()
            .into_pyarray(py),
    ))
}
