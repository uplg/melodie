//! Micro-benchmark candle Metal matmul/op costs to locate the real bottleneck.
//!     cargo run --release --example bench_matmul

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use melodie_engine::Result;

fn sync(t: &Tensor) -> Result<()> {
    t.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    Ok(())
}

fn bench_dtype(dev: &Device, m: usize, k: usize, n: usize, iters: usize, dt: DType) -> Result<()> {
    let x = Tensor::randn(0f32, 1.0, (m, k), dev)?.to_dtype(dt)?;
    let w = Tensor::randn(0f32, 1.0, (k, n), dev)?.to_dtype(dt)?;
    for _ in 0..20 {
        sync(&x.matmul(&w)?)?;
    }
    let t0 = Instant::now();
    let mut y = x.matmul(&w)?;
    for _ in 1..iters {
        y = x.matmul(&w)?;
    }
    sync(&y)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  {dt:?}: {ms:.4} ms/op");
    Ok(())
}

fn bench_matmul(
    dev: &Device,
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    iters: usize,
) -> Result<()> {
    let x = Tensor::randn(0f32, 1.0, (m, k), dev)?;
    let w = Tensor::randn(0f32, 1.0, (k, n), dev)?;
    for _ in 0..20 {
        sync(&x.matmul(&w)?)?;
    }
    let t0 = Instant::now();
    let mut y = x.matmul(&w)?;
    for _ in 1..iters {
        y = x.matmul(&w)?;
    }
    sync(&y)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gflops = 2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e6);
    println!("{name:18} [{m},{k}]@[{k},{n}] = {ms:.4} ms/op  ({gflops:.0} GFLOP/s)");
    Ok(())
}

// time a tiny elementwise op chain to gauge per-kernel-dispatch overhead
fn bench_op(dev: &Device, name: &str, numel: usize, iters: usize) -> Result<()> {
    let x = Tensor::randn(0f32, 1.0, (numel,), dev)?;
    for _ in 0..20 {
        sync(&(x.affine(1.001, 0.0))?)?;
    }
    let t0 = Instant::now();
    let mut y = x.affine(1.001, 0.0)?;
    for _ in 1..iters {
        y = y.affine(1.001, 0.0)?;
    }
    sync(&y)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("{name:18} affine(n={numel}) = {ms:.4} ms/op");
    Ok(())
}

fn bench_named(
    dev: &Device,
    name: &str,
    shape: (usize, usize, usize, usize),
    dt: DType,
    iters: usize,
    op: impl Fn(&Tensor) -> Result<Tensor>,
) -> Result<()> {
    let x = Tensor::randn(0f32, 1.0, shape, dev)?.to_dtype(dt)?;
    for _ in 0..20 {
        sync(&op(&x)?)?;
    }
    let t0 = Instant::now();
    let mut y = op(&x)?;
    for _ in 1..iters {
        y = op(&x)?;
    }
    sync(&y)?;
    println!(
        "  {name:10} {dt:?}: {:.4} ms/op",
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    );
    Ok(())
}

// rmsnorm exactly as the model does it (f32 reduction, back to input dtype)
fn rmsnorm(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    use candle_core::D;
    let dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let n = xf.broadcast_div(&(ms + 1e-5)?.sqrt()?)?.to_dtype(dt)?;
    Ok(n.broadcast_mul(scale)?)
}

// one transformer layer's compute (matmuls + rmsnorm + silu), 1 token, no attention cache
fn bench_layer(dev: &Device, dt: DType, iters: usize) -> Result<()> {
    let mk = |a, b| -> Result<Tensor> {
        Ok(Tensor::randn(0f32, 1.0, (a, b), dev)?.to_dtype(dt).unwrap())
    };
    let x0 = Tensor::randn(0f32, 1.0, (1, 1, 3072), dev)?.to_dtype(dt)?;
    let (wq, wo) = (mk(3072, 3072)?, mk(3072, 3072)?);
    let (w1, w2, w3) = (mk(3072, 8192)?, mk(8192, 3072)?, mk(3072, 8192)?);
    let (n1, n2) = (
        Tensor::randn(0f32, 1.0, (3072,), dev)?.to_dtype(dt)?,
        Tensor::randn(0f32, 1.0, (3072,), dev)?.to_dtype(dt)?,
    );
    let step = |x: &Tensor| -> Result<Tensor> {
        let h = rmsnorm(x, &n1)?;
        let a = h.broadcast_matmul(&wq)?.broadcast_matmul(&wo)?; // q then o proj
        let r = (x + a)?;
        let h2 = rmsnorm(&r, &n2)?;
        let m = (candle_nn::ops::silu(&h2.broadcast_matmul(&w1)?)? * h2.broadcast_matmul(&w3)?)?
            .broadcast_matmul(&w2)?;
        Ok((r + m)?)
    };
    for _ in 0..30 {
        sync(&step(&x0)?)?;
    }
    let t0 = Instant::now();
    let mut x = step(&x0)?;
    for _ in 1..iters {
        x = step(&x0)?;
    }
    sync(&x)?;
    println!(
        "  layer {dt:?}: {:.4} ms/layer",
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    );
    Ok(())
}

// attention block at the depth-decoder shape (h=8, nkv=4, dh=384, growing KV)
fn bench_attn(dev: &Device, dt: DType, iters: usize) -> Result<()> {
    use candle_core::D;
    let (h, nkv, dh, skv) = (8usize, 4usize, 384usize, 9usize);
    let q = Tensor::randn(0f32, 1.0, (1, h, 1, dh), dev)?.to_dtype(dt)?;
    let k = Tensor::randn(0f32, 1.0, (1, nkv, skv, dh), dev)?.to_dtype(dt)?;
    let v = Tensor::randn(0f32, 1.0, (1, nkv, skv, dh), dev)?.to_dtype(dt)?;
    let expand = |x: &Tensor, rep: usize| -> Result<Tensor> {
        let (b, nk, s, d) = x.dims4()?;
        Ok(x.unsqueeze(2)?
            .broadcast_as((b, nk, rep, s, d))?
            .contiguous()?
            .reshape((b, nk * rep, s, d))?)
    };
    let step = || -> Result<Tensor> {
        let ke = expand(&k, h / nkv)?;
        let ve = expand(&v, h / nkv)?;
        let scores = (q.matmul(&ke.transpose(2, 3)?.contiguous()?)? / (dh as f64).sqrt())?;
        let w = candle_nn::ops::softmax(&scores, D::Minus1)?;
        Ok(w.matmul(&ve)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, 1, h * dh))?)
    };
    for _ in 0..30 {
        sync(&step()?)?;
    }
    let t0 = Instant::now();
    let mut y = step()?;
    for _ in 1..iters {
        y = step()?;
    }
    sync(&y)?;
    println!(
        "  attn {dt:?}: {:.4} ms/op",
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    );
    Ok(())
}

// backbone attention at a given KV-cache length: mimics my Attn.fwd (transpose.contiguous + cat)
fn bench_bb_attn(dev: &Device, skv: usize, dt: DType) -> Result<()> {
    use candle_core::D;
    let (b, h, nkv, dh) = (2usize, 24usize, 8usize, 128usize);
    let q = Tensor::randn(0f32, 1.0, (b, h, 1, dh), dev)?.to_dtype(dt)?; // new token
    let cache_k = Tensor::randn(0f32, 1.0, (b, h, skv, dh), dev)?.to_dtype(dt)?; // expanded-head cache
    let cache_v = Tensor::randn(0f32, 1.0, (b, h, skv, dh), dev)?.to_dtype(dt)?;
    let kn = Tensor::randn(0f32, 1.0, (b, nkv, 1, dh), dev)?.to_dtype(dt)?; // new k (nkv heads)
    let vn = Tensor::randn(0f32, 1.0, (b, nkv, 1, dh), dev)?.to_dtype(dt)?;
    let step = || -> Result<Tensor> {
        // expand new kv 8->24, append to cache (the cat), then attend over skv+1
        let ke = kn
            .unsqueeze(2)?
            .broadcast_as((b, nkv, h / nkv, 1, dh))?
            .contiguous()?
            .reshape((b, h, 1, dh))?;
        let ve = vn
            .unsqueeze(2)?
            .broadcast_as((b, nkv, h / nkv, 1, dh))?
            .contiguous()?
            .reshape((b, h, 1, dh))?;
        let k = Tensor::cat(&[&cache_k, &ke], 2)?;
        let v = Tensor::cat(&[&cache_v, &ve], 2)?;
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (dh as f64).sqrt())?;
        let w = candle_nn::ops::softmax(&scores, D::Minus1)?;
        Ok(w.matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, 1, h * dh))?)
    };
    let step_sdpa = || -> Result<Tensor> {
        let ke = kn
            .unsqueeze(2)?
            .broadcast_as((b, nkv, h / nkv, 1, dh))?
            .contiguous()?
            .reshape((b, h, 1, dh))?;
        let ve = vn
            .unsqueeze(2)?
            .broadcast_as((b, nkv, h / nkv, 1, dh))?
            .contiguous()?
            .reshape((b, h, 1, dh))?;
        let k = Tensor::cat(&[&cache_k, &ke], 2)?;
        let v = Tensor::cat(&[&cache_v, &ve], 2)?;
        let o = candle_nn::ops::sdpa(
            &q,
            &k,
            &v,
            None,
            false,
            (1.0 / (dh as f64).sqrt()) as f32,
            1.0,
        )?;
        Ok(o.transpose(1, 2)?.contiguous()?.reshape((b, 1, h * dh))?)
    };
    let bench = |label: &str, f: &dyn Fn() -> Result<Tensor>| -> Result<()> {
        for _ in 0..30 {
            sync(&f()?)?;
        }
        let t0 = Instant::now();
        let mut y = f()?;
        for _ in 1..1000 {
            y = f()?;
        }
        sync(&y)?;
        println!(
            "  bb_attn skv={skv:<4} {label}: {:.4} ms/op",
            t0.elapsed().as_secs_f64() * 1000.0 / 1000.0
        );
        Ok(())
    };
    bench("manual", &step)?;
    bench("sdpa  ", &step_sdpa)?;
    // correctness: sdpa must match the manual attention (attend-all case here)
    let a = step()?.to_dtype(DType::F32)?;
    let c = step_sdpa()?.to_dtype(DType::F32)?;
    let d = (a - c)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    println!("  bb_attn skv={skv:<4} sdpa-vs-manual max|Δ|={d:.3e}");
    Ok(())
}

// replicate the real autoregressive loop: 28 layers, KV cache GROWN by cat each frame
fn bench_loop(dev: &Device, prompt_len: usize, n_frames: usize, dt: DType) -> Result<()> {
    use candle_core::D;
    let (b, h, nkv, dh, nl) = (2usize, 24usize, 8usize, 128usize, 28usize);
    let mk = |s: usize| {
        Tensor::randn(0f32, 1.0, (b, h, s, dh), dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap()
    };
    let mut ck: Vec<Tensor> = (0..nl).map(|_| mk(prompt_len)).collect();
    let mut cv: Vec<Tensor> = (0..nl).map(|_| mk(prompt_len)).collect();
    let q = Tensor::randn(0f32, 1.0, (b, h, 1, dh), dev)?.to_dtype(dt)?;
    let kn = Tensor::randn(0f32, 1.0, (b, nkv, 1, dh), dev)?.to_dtype(dt)?;
    let vn = Tensor::randn(0f32, 1.0, (b, nkv, 1, dh), dev)?.to_dtype(dt)?;
    let expand = |x: &Tensor| -> Result<Tensor> {
        let (bb, nk, s, d) = x.dims4()?;
        Ok(x.unsqueeze(2)?
            .broadcast_as((bb, nk, h / nkv, s, d))?
            .contiguous()?
            .reshape((bb, h, s, d))?)
    };
    let mut times = Vec::new();
    for _ in 0..n_frames {
        let t0 = Instant::now();
        let mut last = q.clone();
        for l in 0..nl {
            ck[l] = Tensor::cat(&[&ck[l], &expand(&kn)?], 2)?; // GROWING cat (the real pattern)
            cv[l] = Tensor::cat(&[&cv[l], &expand(&vn)?], 2)?;
            let scores = (q.matmul(&ck[l].transpose(2, 3)?.contiguous()?)? / (dh as f64).sqrt())?;
            let w = candle_nn::ops::softmax(&scores, D::Minus1)?;
            last = w.matmul(&cv[l])?;
        }
        sync(&last)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let warm: f64 = times[5..].iter().sum::<f64>() / (times.len() - 5) as f64;
    println!(
        "  loop prompt={prompt_len:<4} {dt:?}: {warm:.1} ms/frame (28 layers, growing-cat KV cache)"
    );
    Ok(())
}

// 28 DISTINCT layers (~6 GB cold weights). `fused`+`twod` toggle the op-count optimizations.
fn bench_cold_backbone(dev: &Device, dt: DType, fused: bool, twod: bool) -> Result<()> {
    let mk = |a, b| {
        Tensor::randn(0f32, 1.0, (a, b), dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap()
    };
    let mkn = || {
        Tensor::randn(0f32, 1.0, (3072,), dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap()
    };
    struct L {
        wq: Tensor,
        wo: Tensor,
        w1: Tensor,
        w2: Tensor,
        w3: Tensor,
        n1: Tensor,
        n2: Tensor,
    }
    let layers: Vec<L> = (0..28)
        .map(|_| L {
            wq: mk(3072, 3072),
            wo: mk(3072, 3072),
            w1: mk(3072, 8192),
            w2: mk(8192, 3072),
            w3: mk(3072, 8192),
            n1: mkn(),
            n2: mkn(),
        })
        .collect();
    let x0 = Tensor::randn(0f32, 1.0, (2, 1, 3072), dev)?.to_dtype(dt)?; // B=2, 1 token
    let norm = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
        if fused {
            Ok(candle_nn::ops::rms_norm(&x.contiguous()?, w, 1e-5)?)
        } else {
            rmsnorm(x, w)
        }
    };
    // matmul: 2D [B*S,D]@[D,N] (weight once) vs 3D broadcast (weight per batch)
    let mm = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
        if twod {
            let (b, s, d) = x.dims3()?;
            let n = w.dim(1)?;
            Ok(x.reshape((b * s, d))?.matmul(w)?.reshape((b, s, n))?)
        } else {
            Ok(x.broadcast_matmul(w)?)
        }
    };
    let step = |x: &Tensor, l: &L| -> Result<Tensor> {
        let h = norm(x, &l.n1)?;
        let a = mm(&mm(&h, &l.wq)?, &l.wo)?;
        let r = (x + a)?;
        let h2 = norm(&r, &l.n2)?;
        let m = mm(
            &(candle_nn::ops::silu(&mm(&h2, &l.w1)?)? * mm(&h2, &l.w3)?)?,
            &l.w2,
        )?;
        Ok((r + m)?)
    };
    let fwd = || -> Result<Tensor> {
        let mut x = x0.clone();
        for l in &layers {
            x = step(&x, l)?;
        }
        Ok(x)
    };
    for _ in 0..10 {
        sync(&fwd()?)?;
    }
    let t0 = Instant::now();
    let mut y = fwd()?;
    for _ in 1..100 {
        y = fwd()?;
    }
    sync(&y)?;
    println!(
        "  cold_backbone fused={fused} twod={twod}: {:.1} ms/forward",
        t0.elapsed().as_secs_f64() * 1000.0 / 100.0
    );
    Ok(())
}

fn main() -> Result<()> {
    use candle_core::D;
    let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("device: {dev:?}");
    println!("--- 3D-batched vs 2D matmul for B=2 (the weight-read-2x hypothesis) ---");
    {
        let w = Tensor::randn(0f32, 1.0, (3072, 3072), &dev)?.to_dtype(DType::BF16)?;
        let x3 = Tensor::randn(0f32, 1.0, (2, 1, 3072), &dev)?.to_dtype(DType::BF16)?; // 3D [B,1,D]
        let x2 = Tensor::randn(0f32, 1.0, (2, 3072), &dev)?.to_dtype(DType::BF16)?; // 2D [B,D]
        for _ in 0..30 {
            sync(&x3.broadcast_matmul(&w)?)?;
            sync(&x2.matmul(&w)?)?;
        }
        let t = Instant::now();
        let mut y = x3.broadcast_matmul(&w)?;
        for _ in 1..3000 {
            y = x3.broadcast_matmul(&w)?;
        }
        sync(&y)?;
        println!(
            "  3D [2,1,3072]@[3072,3072] (broadcast): {:.4} ms/op",
            t.elapsed().as_secs_f64() * 1000.0 / 3000.0
        );
        let t = Instant::now();
        let mut y = x2.matmul(&w)?;
        for _ in 1..3000 {
            y = x2.matmul(&w)?;
        }
        sync(&y)?;
        println!(
            "  2D [2,3072]@[3072,3072]   (M=2)     : {:.4} ms/op",
            t.elapsed().as_secs_f64() * 1000.0 / 3000.0
        );
    }
    println!("--- COLD backbone (28 layers, B=2): op-count optimizations ---");
    bench_cold_backbone(&dev, DType::BF16, false, false)?; // baseline (hand rmsnorm, 3D matmul)
    bench_cold_backbone(&dev, DType::BF16, false, true)?; // + 2D matmul
    bench_cold_backbone(&dev, DType::BF16, true, true)?; // + 2D matmul + fused rms_norm
    println!("--- REAL loop: 28-layer growing-cat KV cache, prompt 8 vs 102 vs 300 ---");
    bench_loop(&dev, 8, 30, DType::BF16)?;
    bench_loop(&dev, 102, 30, DType::BF16)?;
    bench_loop(&dev, 300, 30, DType::BF16)?;
    println!("--- backbone attention vs KV-cache length (the gen_song scaling) ---");
    bench_bb_attn(&dev, 25, DType::BF16)?;
    bench_bb_attn(&dev, 120, DType::BF16)?;
    bench_bb_attn(&dev, 500, DType::BF16)?;
    println!("--- attention block (depth-decoder shape): f32 vs bf16 vs f16 ---");
    bench_attn(&dev, DType::F32, 2000)?;
    bench_attn(&dev, DType::BF16, 2000)?;
    bench_attn(&dev, DType::F16, 2000)?;
    println!("--- full transformer layer (matmuls+rmsnorm+silu): f32 vs bf16 vs f16 ---");
    bench_layer(&dev, DType::F32, 1000)?;
    bench_layer(&dev, DType::BF16, 1000)?;
    bench_layer(&dev, DType::F16, 1000)?;
    println!("--- elementwise/reduction ops: f32 vs bf16 (CPU-fallback hunt) ---");
    for dt in [DType::F32, DType::BF16] {
        bench_named(&dev, "softmax", (1, 24, 1, 256), dt, 3000, |x| {
            Ok(candle_nn::ops::softmax(x, D::Minus1)?)
        })?;
    }
    for dt in [DType::F32, DType::BF16] {
        bench_named(&dev, "silu", (1, 1, 1, 8192), dt, 3000, |x| {
            Ok(candle_nn::ops::silu(x)?)
        })?;
    }
    for dt in [DType::F32, DType::BF16] {
        bench_named(&dev, "add", (1, 1, 1, 3072), dt, 3000, |x| Ok((x + x)?))?;
    }
    for dt in [DType::F32, DType::BF16] {
        bench_named(&dev, "to_f32", (1, 1, 1, 3072), dt, 3000, |x| {
            Ok(x.to_dtype(DType::F32)?)
        })?;
    }
    println!("--- dtype comparison [1,3072]@[3072,3072] (the bf16 question) ---");
    bench_dtype(&dev, 1, 3072, 3072, 3000, DType::F32)?;
    bench_dtype(&dev, 1, 3072, 3072, 3000, DType::BF16)?;
    bench_dtype(&dev, 1, 3072, 3072, 3000, DType::F16)?;
    println!("--- dtype comparison mlp_down [1,8192]@[8192,3072] ---");
    bench_dtype(&dev, 1, 8192, 3072, 3000, DType::F32)?;
    bench_dtype(&dev, 1, 8192, 3072, 3000, DType::F16)?;
    println!("--- matmul (1 token = GEMV) ---");
    bench_matmul(&dev, "q/o_proj", 1, 3072, 3072, 3000)?;
    bench_matmul(&dev, "k/v_proj", 1, 3072, 1024, 3000)?;
    bench_matmul(&dev, "mlp_up/gate", 1, 3072, 8192, 3000)?;
    bench_matmul(&dev, "mlp_down", 1, 8192, 3072, 3000)?;
    bench_matmul(&dev, "c0_head", 1, 3072, 8197, 3000)?;
    println!("--- matmul (prompt S=8) ---");
    bench_matmul(&dev, "q/o_proj S8", 8, 3072, 3072, 3000)?;
    println!("--- per-kernel-dispatch overhead ---");
    bench_op(&dev, "elementwise", 3072, 5000)?;
    bench_op(&dev, "elementwise", 8197, 5000)?;
    Ok(())
}
