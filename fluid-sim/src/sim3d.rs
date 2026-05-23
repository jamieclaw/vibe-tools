//! 3D PIC/FLIP fluid simulation, exposed via wasm-bindgen.
//!
//! Coordinates: world units 1 cell = 1.0. Domain is [0, gx] × [0, gy] × [0, gz].
//! MAC grid: u at faces ⊥ x (size (gx+1)*gy*gz), v at faces ⊥ y, w at faces ⊥ z.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Sim3D {
    gx: usize, gy: usize, gz: usize,

    u: Vec<f32>, u_prev: Vec<f32>, u_w: Vec<f32>,
    v: Vec<f32>, v_prev: Vec<f32>, v_w: Vec<f32>,
    w: Vec<f32>, w_prev: Vec<f32>, w_w: Vec<f32>,

    cell_type: Vec<u8>,

    px: Vec<f32>, py: Vec<f32>, pz: Vec<f32>,
    pvx: Vec<f32>, pvy: Vec<f32>, pvz: Vec<f32>,

    pub gravity: f32,
    pub flip_ratio: f32,
    pub iters: u32,
}

#[inline] fn iu(i: usize, j: usize, k: usize, gx: usize, gy: usize) -> usize { i + (gx + 1) * (j + gy * k) }
#[inline] fn iv(i: usize, j: usize, k: usize, gx: usize, gy: usize) -> usize { i + gx * (j + (gy + 1) * k) }
#[inline] fn iw(i: usize, j: usize, k: usize, gx: usize, gy: usize) -> usize { i + gx * (j + gy * k) }
#[inline] fn ic(i: usize, j: usize, k: usize, gx: usize, gy: usize) -> usize { i + gx * (j + gy * k) }

#[wasm_bindgen]
impl Sim3D {
    #[wasm_bindgen(constructor)]
    pub fn new(gx: usize, gy: usize, gz: usize) -> Sim3D {
        Sim3D {
            gx, gy, gz,
            u: vec![0.0; (gx + 1) * gy * gz],
            u_prev: vec![0.0; (gx + 1) * gy * gz],
            u_w: vec![0.0; (gx + 1) * gy * gz],
            v: vec![0.0; gx * (gy + 1) * gz],
            v_prev: vec![0.0; gx * (gy + 1) * gz],
            v_w: vec![0.0; gx * (gy + 1) * gz],
            w: vec![0.0; gx * gy * (gz + 1)],
            w_prev: vec![0.0; gx * gy * (gz + 1)],
            w_w: vec![0.0; gx * gy * (gz + 1)],
            cell_type: vec![1; gx * gy * gz],
            px: Vec::new(), py: Vec::new(), pz: Vec::new(),
            pvx: Vec::new(), pvy: Vec::new(), pvz: Vec::new(),
            gravity: 9.8,
            flip_ratio: 0.97,
            iters: 30,
        }
    }

    pub fn grid_x(&self) -> usize { self.gx }
    pub fn grid_y(&self) -> usize { self.gy }
    pub fn grid_z(&self) -> usize { self.gz }
    pub fn particle_count(&self) -> usize { self.px.len() }

    pub fn px_ptr(&self) -> *const f32 { self.px.as_ptr() }
    pub fn py_ptr(&self) -> *const f32 { self.py.as_ptr() }
    pub fn pz_ptr(&self) -> *const f32 { self.pz.as_ptr() }
    pub fn pvx_ptr(&self) -> *const f32 { self.pvx.as_ptr() }
    pub fn pvy_ptr(&self) -> *const f32 { self.pvy.as_ptr() }
    pub fn pvz_ptr(&self) -> *const f32 { self.pvz.as_ptr() }

    pub fn reset_particles(&mut self, count: usize, x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32) {
        self.px.clear(); self.py.clear(); self.pz.clear();
        self.pvx.clear(); self.pvy.clear(); self.pvz.clear();
        let rx = x1 - x0;
        let ry = y1 - y0;
        let rz = z1 - z0;
        let total_vol = rx * ry * rz;
        let per_dim = (count as f32 / total_vol).cbrt();
        let nx = (rx * per_dim).ceil() as usize;
        let ny = (ry * per_dim).ceil() as usize;
        let nz = (rz * per_dim).ceil() as usize;
        let sx = rx / nx as f32;
        let sy = ry / ny as f32;
        let sz = rz / nz as f32;
        let mut seed: u32 = 12345;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if self.px.len() >= count { return; }
                    self.px.push(x0 + (i as f32 + 0.5) * sx + rnd() * sx * 0.3);
                    self.py.push(y0 + (j as f32 + 0.5) * sy + rnd() * sy * 0.3);
                    self.pz.push(z0 + (k as f32 + 0.5) * sz + rnd() * sz * 0.3);
                    self.pvx.push(0.0);
                    self.pvy.push(0.0);
                    self.pvz.push(0.0);
                }
            }
        }
    }

    pub fn splash(&mut self, cx: f32, cy: f32, cz: f32, count: usize, vel: f32) {
        let mut seed: u32 = (cx as u32).wrapping_mul(7919).wrapping_add(cy as u32).wrapping_add((cz * 17.0) as u32);
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        for _ in 0..count {
            self.px.push(cx + rnd() * 2.0);
            self.py.push(cy + rnd() * 1.0);
            self.pz.push(cz + rnd() * 2.0);
            self.pvx.push(rnd() * vel);
            self.pvy.push(vel * 0.5 + rnd() * vel);
            self.pvz.push(rnd() * vel);
        }
    }

    pub fn apply_force(&mut self, cx: f32, cy: f32, cz: f32, dvx: f32, dvy: f32, dvz: f32, radius: f32) {
        let r2 = radius * radius;
        for k in 0..self.px.len() {
            let dx = self.px[k] - cx;
            let dy = self.py[k] - cy;
            let dz = self.pz[k] - cz;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > r2 { continue; }
            let fall = 1.0 - d2.sqrt() / radius;
            self.pvx[k] += dvx * fall;
            self.pvy[k] += dvy * fall;
            self.pvz[k] += dvz * fall;
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.classify_cells();
        self.p2g();
        self.apply_forces(dt);
        self.project();
        self.g2p();
        self.advect(dt);
        self.push_apart(2);
    }

    fn classify_cells(&mut self) {
        for c in self.cell_type.iter_mut() { *c = 1; }
        // border solid
        for k in 0..self.gz {
            for j in 0..self.gy {
                self.cell_type[ic(0, j, k, self.gx, self.gy)] = 2;
                self.cell_type[ic(self.gx - 1, j, k, self.gx, self.gy)] = 2;
            }
        }
        for k in 0..self.gz {
            for i in 0..self.gx {
                self.cell_type[ic(i, 0, k, self.gx, self.gy)] = 2;
                self.cell_type[ic(i, self.gy - 1, k, self.gx, self.gy)] = 2;
            }
        }
        for j in 0..self.gy {
            for i in 0..self.gx {
                self.cell_type[ic(i, j, 0, self.gx, self.gy)] = 2;
                self.cell_type[ic(i, j, self.gz - 1, self.gx, self.gy)] = 2;
            }
        }
        for k in 0..self.px.len() {
            let i = self.px[k] as i32;
            let j = self.py[k] as i32;
            let kk = self.pz[k] as i32;
            if i > 0 && j > 0 && kk > 0
               && (i as usize) < self.gx - 1
               && (j as usize) < self.gy - 1
               && (kk as usize) < self.gz - 1 {
                let c = ic(i as usize, j as usize, kk as usize, self.gx, self.gy);
                if self.cell_type[c] != 2 { self.cell_type[c] = 0; }
            }
        }
    }

    fn p2g(&mut self) {
        for x in self.u.iter_mut() { *x = 0.0; }
        for x in self.u_w.iter_mut() { *x = 0.0; }
        for x in self.v.iter_mut() { *x = 0.0; }
        for x in self.v_w.iter_mut() { *x = 0.0; }
        for x in self.w.iter_mut() { *x = 0.0; }
        for x in self.w_w.iter_mut() { *x = 0.0; }

        let gx = self.gx; let gy = self.gy; let gz = self.gz;

        for k in 0..self.px.len() {
            let x = self.px[k]; let y = self.py[k]; let z = self.pz[k];
            // u: (i, j+0.5, k+0.5)
            self.splat_u(x, y - 0.5, z - 0.5, self.pvx[k]);
            // v: (i+0.5, j, k+0.5)
            self.splat_v(x - 0.5, y, z - 0.5, self.pvy[k]);
            // w: (i+0.5, j+0.5, k)
            self.splat_w(x - 0.5, y - 0.5, z, self.pvz[k]);
            let _ = (gx, gy, gz);
        }

        for kk in 0..self.u.len() { if self.u_w[kk] > 0.0 { self.u[kk] /= self.u_w[kk]; } }
        for kk in 0..self.v.len() { if self.v_w[kk] > 0.0 { self.v[kk] /= self.v_w[kk]; } }
        for kk in 0..self.w.len() { if self.w_w[kk] > 0.0 { self.w[kk] /= self.w_w[kk]; } }

        self.u_prev.copy_from_slice(&self.u);
        self.v_prev.copy_from_slice(&self.v);
        self.w_prev.copy_from_slice(&self.w);
    }

    fn splat_u(&mut self, fx: f32, fy: f32, fz: f32, val: f32) {
        let i = fx.floor() as i32;
        let j = fy.floor() as i32;
        let k = fz.floor() as i32;
        if i < 0 || j < 0 || k < 0 { return; }
        let tx = fx - i as f32; let ty = fy - j as f32; let tz = fz - k as f32;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let ii = (i + di) as usize;
                    let jj = (j + dj) as usize;
                    let kk = (k + dk) as usize;
                    if ii > self.gx || jj >= self.gy || kk >= self.gz { continue; }
                    let wx = if di == 0 { 1.0 - tx } else { tx };
                    let wy = if dj == 0 { 1.0 - ty } else { ty };
                    let wz = if dk == 0 { 1.0 - tz } else { tz };
                    let w = wx * wy * wz;
                    let id = iu(ii, jj, kk, self.gx, self.gy);
                    self.u[id] += val * w;
                    self.u_w[id] += w;
                }
            }
        }
    }
    fn splat_v(&mut self, fx: f32, fy: f32, fz: f32, val: f32) {
        let i = fx.floor() as i32;
        let j = fy.floor() as i32;
        let k = fz.floor() as i32;
        if i < 0 || j < 0 || k < 0 { return; }
        let tx = fx - i as f32; let ty = fy - j as f32; let tz = fz - k as f32;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let ii = (i + di) as usize;
                    let jj = (j + dj) as usize;
                    let kk = (k + dk) as usize;
                    if ii >= self.gx || jj > self.gy || kk >= self.gz { continue; }
                    let wx = if di == 0 { 1.0 - tx } else { tx };
                    let wy = if dj == 0 { 1.0 - ty } else { ty };
                    let wz = if dk == 0 { 1.0 - tz } else { tz };
                    let w = wx * wy * wz;
                    let id = iv(ii, jj, kk, self.gx, self.gy);
                    self.v[id] += val * w;
                    self.v_w[id] += w;
                }
            }
        }
    }
    fn splat_w(&mut self, fx: f32, fy: f32, fz: f32, val: f32) {
        let i = fx.floor() as i32;
        let j = fy.floor() as i32;
        let k = fz.floor() as i32;
        if i < 0 || j < 0 || k < 0 { return; }
        let tx = fx - i as f32; let ty = fy - j as f32; let tz = fz - k as f32;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let ii = (i + di) as usize;
                    let jj = (j + dj) as usize;
                    let kk = (k + dk) as usize;
                    if ii >= self.gx || jj >= self.gy || kk > self.gz { continue; }
                    let wx = if di == 0 { 1.0 - tx } else { tx };
                    let wy = if dj == 0 { 1.0 - ty } else { ty };
                    let wz = if dk == 0 { 1.0 - tz } else { tz };
                    let w = wx * wy * wz;
                    let id = iw(ii, jj, kk, self.gx, self.gy);
                    self.w[id] += val * w;
                    self.w_w[id] += w;
                }
            }
        }
    }

    fn apply_forces(&mut self, dt: f32) {
        // gravity in -Y
        for kk in 0..self.gz {
            for j in 0..=self.gy {
                for i in 0..self.gx {
                    self.v[iv(i, j, kk, self.gx, self.gy)] -= self.gravity * dt;
                }
            }
        }
    }

    fn enforce_solid(&mut self) {
        for k in 0..self.gz {
            for j in 0..self.gy {
                for i in 0..self.gx {
                    if self.cell_type[ic(i, j, k, self.gx, self.gy)] == 2 {
                        self.u[iu(i, j, k, self.gx, self.gy)] = 0.0;
                        self.u[iu(i + 1, j, k, self.gx, self.gy)] = 0.0;
                        self.v[iv(i, j, k, self.gx, self.gy)] = 0.0;
                        self.v[iv(i, j + 1, k, self.gx, self.gy)] = 0.0;
                        self.w[iw(i, j, k, self.gx, self.gy)] = 0.0;
                        self.w[iw(i, j, k + 1, self.gx, self.gy)] = 0.0;
                    }
                }
            }
        }
    }

    fn project(&mut self) {
        self.enforce_solid();
        let overrelax = 1.9;
        for _ in 0..self.iters {
            for k in 1..self.gz - 1 {
                for j in 1..self.gy - 1 {
                    for i in 1..self.gx - 1 {
                        if self.cell_type[ic(i, j, k, self.gx, self.gy)] != 0 { continue; }
                        let s_l = if self.cell_type[ic(i - 1, j, k, self.gx, self.gy)] != 2 { 1.0 } else { 0.0 };
                        let s_r = if self.cell_type[ic(i + 1, j, k, self.gx, self.gy)] != 2 { 1.0 } else { 0.0 };
                        let s_d = if self.cell_type[ic(i, j - 1, k, self.gx, self.gy)] != 2 { 1.0 } else { 0.0 };
                        let s_u = if self.cell_type[ic(i, j + 1, k, self.gx, self.gy)] != 2 { 1.0 } else { 0.0 };
                        let s_b = if self.cell_type[ic(i, j, k - 1, self.gx, self.gy)] != 2 { 1.0 } else { 0.0 };
                        let s_f = if self.cell_type[ic(i, j, k + 1, self.gx, self.gy)] != 2 { 1.0 } else { 0.0 };
                        let s = s_l + s_r + s_d + s_u + s_b + s_f;
                        if s == 0.0 { continue; }
                        let div = self.u[iu(i + 1, j, k, self.gx, self.gy)] - self.u[iu(i, j, k, self.gx, self.gy)]
                                + self.v[iv(i, j + 1, k, self.gx, self.gy)] - self.v[iv(i, j, k, self.gx, self.gy)]
                                + self.w[iw(i, j, k + 1, self.gx, self.gy)] - self.w[iw(i, j, k, self.gx, self.gy)];
                        let dp = -div / s * overrelax;
                        self.u[iu(i, j, k, self.gx, self.gy)]     -= dp * s_l;
                        self.u[iu(i + 1, j, k, self.gx, self.gy)] += dp * s_r;
                        self.v[iv(i, j, k, self.gx, self.gy)]     -= dp * s_d;
                        self.v[iv(i, j + 1, k, self.gx, self.gy)] += dp * s_u;
                        self.w[iw(i, j, k, self.gx, self.gy)]     -= dp * s_b;
                        self.w[iw(i, j, k + 1, self.gx, self.gy)] += dp * s_f;
                    }
                }
            }
        }
    }

    fn sample_face(&self, arr: &[f32], fx: f32, fy: f32, fz: f32, dim: u8) -> f32 {
        let i = fx.floor() as i32;
        let j = fy.floor() as i32;
        let k = fz.floor() as i32;
        if i < 0 || j < 0 || k < 0 { return 0.0; }
        let (ix_max, jy_max, kz_max) = match dim {
            0 => (self.gx + 1, self.gy, self.gz),
            1 => (self.gx, self.gy + 1, self.gz),
            _ => (self.gx, self.gy, self.gz + 1),
        };
        if (i as usize) + 1 >= ix_max || (j as usize) + 1 >= jy_max || (k as usize) + 1 >= kz_max {
            return 0.0;
        }
        let tx = fx - i as f32; let ty = fy - j as f32; let tz = fz - k as f32;
        let i = i as usize; let j = j as usize; let k = k as usize;
        let id = |ii: usize, jj: usize, kk: usize| -> usize {
            match dim {
                0 => iu(ii, jj, kk, self.gx, self.gy),
                1 => iv(ii, jj, kk, self.gx, self.gy),
                _ => iw(ii, jj, kk, self.gx, self.gy),
            }
        };
        let c000 = arr[id(i, j, k)];
        let c100 = arr[id(i + 1, j, k)];
        let c010 = arr[id(i, j + 1, k)];
        let c110 = arr[id(i + 1, j + 1, k)];
        let c001 = arr[id(i, j, k + 1)];
        let c101 = arr[id(i + 1, j, k + 1)];
        let c011 = arr[id(i, j + 1, k + 1)];
        let c111 = arr[id(i + 1, j + 1, k + 1)];
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        c0 * (1.0 - tz) + c1 * tz
    }

    fn g2p(&mut self) {
        let flip = self.flip_ratio;
        for k in 0..self.px.len() {
            let x = self.px[k]; let y = self.py[k]; let z = self.pz[k];
            let new_u = self.sample_face(&self.u, x, y - 0.5, z - 0.5, 0);
            let old_u = self.sample_face(&self.u_prev, x, y - 0.5, z - 0.5, 0);
            let new_v = self.sample_face(&self.v, x - 0.5, y, z - 0.5, 1);
            let old_v = self.sample_face(&self.v_prev, x - 0.5, y, z - 0.5, 1);
            let new_w = self.sample_face(&self.w, x - 0.5, y - 0.5, z, 2);
            let old_w = self.sample_face(&self.w_prev, x - 0.5, y - 0.5, z, 2);

            let pic_u = new_u; let flip_u = self.pvx[k] + (new_u - old_u);
            let pic_v = new_v; let flip_v = self.pvy[k] + (new_v - old_v);
            let pic_w = new_w; let flip_w = self.pvz[k] + (new_w - old_w);

            self.pvx[k] = pic_u * (1.0 - flip) + flip_u * flip;
            self.pvy[k] = pic_v * (1.0 - flip) + flip_v * flip;
            self.pvz[k] = pic_w * (1.0 - flip) + flip_w * flip;
        }
    }

    fn advect(&mut self, dt: f32) {
        let gx = self.gx as f32; let gy = self.gy as f32; let gz = self.gz as f32;
        for k in 0..self.px.len() {
            self.px[k] += self.pvx[k] * dt;
            self.py[k] += self.pvy[k] * dt;
            self.pz[k] += self.pvz[k] * dt;
            let m = 1.05;
            if self.px[k] < m { self.px[k] = m; if self.pvx[k] < 0.0 { self.pvx[k] = 0.0; } }
            if self.px[k] > gx - m { self.px[k] = gx - m; if self.pvx[k] > 0.0 { self.pvx[k] = 0.0; } }
            if self.py[k] < m { self.py[k] = m; if self.pvy[k] < 0.0 { self.pvy[k] = 0.0; } }
            if self.py[k] > gy - m { self.py[k] = gy - m; if self.pvy[k] > 0.0 { self.pvy[k] = 0.0; } }
            if self.pz[k] < m { self.pz[k] = m; if self.pvz[k] < 0.0 { self.pvz[k] = 0.0; } }
            if self.pz[k] > gz - m { self.pz[k] = gz - m; if self.pvz[k] > 0.0 { self.pvz[k] = 0.0; } }
        }
    }

    fn push_apart(&mut self, iters: u32) {
        let min_dist: f32 = 0.7;
        let min_dist2 = min_dist * min_dist;
        let cs = 1.0_f32; // bucket = 1 cell
        let cw = self.gx + 1;
        let ch = self.gy + 1;
        let cd = self.gz + 1;
        for _ in 0..iters {
            let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); cw * ch * cd];
            for k in 0..self.px.len() {
                let bx = (self.px[k] / cs) as usize;
                let by = (self.py[k] / cs) as usize;
                let bz = (self.pz[k] / cs) as usize;
                let bi = bx + cw * (by + ch * bz);
                if bi < buckets.len() { buckets[bi].push(k as u32); }
            }
            for k in 0..self.px.len() {
                let bx = (self.px[k] / cs) as i32;
                let by = (self.py[k] / cs) as i32;
                let bz = (self.pz[k] / cs) as i32;
                for dz in -1..=1i32 {
                    for dy in -1..=1i32 {
                        for dx in -1..=1i32 {
                            let nx = bx + dx; let ny = by + dy; let nz = bz + dz;
                            if nx < 0 || ny < 0 || nz < 0
                               || (nx as usize) >= cw || (ny as usize) >= ch || (nz as usize) >= cd { continue; }
                            let bi = nx as usize + cw * (ny as usize + ch * nz as usize);
                            for &j_u32 in &buckets[bi] {
                                let j = j_u32 as usize;
                                if j <= k { continue; }
                                let ddx = self.px[j] - self.px[k];
                                let ddy = self.py[j] - self.py[k];
                                let ddz = self.pz[j] - self.pz[k];
                                let d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                                if d2 >= min_dist2 || d2 < 1e-6 { continue; }
                                let d = d2.sqrt();
                                let push = (min_dist - d) * 0.5;
                                let nx2 = ddx / d;
                                let ny2 = ddy / d;
                                let nz2 = ddz / d;
                                self.px[k] -= nx2 * push;
                                self.py[k] -= ny2 * push;
                                self.pz[k] -= nz2 * push;
                                self.px[j] += nx2 * push;
                                self.py[j] += ny2 * push;
                                self.pz[j] += nz2 * push;
                            }
                        }
                    }
                }
            }
        }
    }
}
