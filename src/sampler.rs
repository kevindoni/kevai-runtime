//! Sampling deterministik: argmax / temperature / top-k / top-p.
//! RNG xorshift64 — tidak bergantung pada crate lua.

pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self { temperature: 0.8, top_p: 0.9, top_k: 40, state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        if logits.is_empty() { return 0; }
        if self.temperature <= 0.0 {
            return logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
        }

        // Salin dan kosongkan token dengan probabilitas kecil (top-k).
        let mut v: Vec<f32> = logits.iter().map(|&x| x / self.temperature).collect();
        if self.top_k > 0 && self.top_k < v.len() {
            let mut order: Vec<usize> = (0..v.len()).collect();
            order.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            for &i in order.iter().skip(self.top_k) { v[i] = f32::NEG_INFINITY; }
        }

        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() { return 0; }
        for p in v.iter_mut() { *p = (*p - max).exp(); }
        let sum: f32 = v.iter().sum();
        if sum <= 0.0 { return 0; }
        for p in v.iter_mut() { *p /= sum; }

        // Nucleus (top-p).
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut order: Vec<usize> = (0..v.len()).collect();
            order.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            let mut acc = 0.0f32;
            let mut cutoff = order.len();
            for (n, &i) in order.iter().enumerate() {
                acc += v[i];
                if acc >= self.top_p { cutoff = n + 1; break; }
            }
            for &i in order.iter().skip(cutoff) { v[i] = 0.0; }
            let s: f32 = v.iter().sum();
            if s > 0.0 { for p in v.iter_mut() { *p /= s; } }
        }

        let r = self.next_f32();
        let mut acc = 0.0f32;
        for (i, &p) in v.iter().enumerate() {
            acc += p;
            if r <= acc { return i; }
        }
        v.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_when_greedy() {
        let mut s = Sampler::new(1);
        s.temperature = 0.0;
        assert_eq!(s.sample(&[0.1, 0.5, 0.3]), 1);
    }

    #[test]
    fn sample_returns_valid_index() {
        let mut s = Sampler::new(42);
        for _ in 0..50 {
            let i = s.sample(&[1.0, 2.0, 0.5, 3.0]);
            assert!(i < 4);
        }
    }
}
