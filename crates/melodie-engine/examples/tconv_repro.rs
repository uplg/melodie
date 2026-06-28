//! Minimal repro of a candle 0.11 `ConvTranspose1d` BATCH bug: with N>=2, batch
//! element 0 matches the exact scatter definition of a transposed conv, but later
//! batch elements are garbage (here n1 worst |Δ|~99 vs 0 for n0). This is why the
//! HeartCodec decoder uses an explicit zero-stuff + conv1d transposed conv instead
//! of the built-in (the codec runs the 2 stereo channels as a batch of 2). Run:
//!     cargo run --release --example tconv_repro
use candle_core::{Device, Tensor};
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, Module};
use melodie_engine::Result;

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let (nb, cin, cout, k, stride, l) = (2usize, 3usize, 2usize, 10usize, 5usize, 12usize);

    // deterministic inputs/weights, distinct per batch
    let xv: Vec<f32> = (0..nb * cin * l).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
    let wv: Vec<f32> = (0..cin * cout * k).map(|i| ((i * 5 % 11) as f32) - 5.0).collect();
    let x = Tensor::from_vec(xv.clone(), (nb, cin, l), &dev)?;
    let w = Tensor::from_vec(wv.clone(), (cin, cout, k), &dev)?;

    let outlen = (l - 1) * stride + k;
    // scatter ref per batch: out[n,oc,p*stride+j] += x[n,ic,p]*w[ic,oc,j]
    let mut refout = vec![0f32; nb * cout * outlen];
    for n in 0..nb {
        for oc in 0..cout {
            for ic in 0..cin {
                for p in 0..l {
                    for j in 0..k {
                        refout[(n * cout + oc) * outlen + p * stride + j] +=
                            xv[(n * cin + ic) * l + p] * wv[(ic * cout + oc) * k + j];
                    }
                }
            }
        }
    }

    let cfg = ConvTranspose1dConfig { stride, ..Default::default() };
    let y = ConvTranspose1d::new(w, None, cfg).forward(&x)?;
    let yv = y.flatten_all()?.to_vec1::<f32>()?;

    for n in 0..nb {
        let (mut wi, mut wd) = (0usize, 0f32);
        for i in 0..cout * outlen {
            let gi = n * cout * outlen + i;
            let d = (yv[gi] - refout[gi]).abs();
            if d > wd {
                wd = d;
                wi = i;
            }
        }
        println!("batch n{n}: worst |Δ|={wd:.3e} at (oc{}, pos{})", wi / outlen, wi % outlen);
    }
    Ok(())
}
