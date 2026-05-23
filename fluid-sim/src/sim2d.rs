//! 2D PIC/FLIP fluid simulation, exposed via wasm-bindgen.
//!
//! Coordinate system: pixel-space x ∈ [0, w_px], y ∈ [0, h_px].
//! Grid cells are square with side = `cell` pixels. Velocities live
//! on a MAC staggered grid: u at vertical edges, v at horizontal edges.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Sim2D {
    // grid dims
    gw: usize,
    gh: usize,
    cell: f32,
    w_px: f32,
    h_px: f32,

    // velocity grids
    u: Vec<f32>,      // (gw+1) * gh
    u_prev: Vec<f32>,
    u_w: Vec<f32>,    // accumulator weight
    v: Vec<f32>,      // gw * (gh+1)
    v_prev: Vec<f32>,
    v_w: Vec<f32>,

    // cell type: 0=fluid, 1=air, 2=solid
    cell_type: Vec<u8>,
    walls: Vec<u8>,

    // particles SoA
    px: Vec<f32>,
    py: Vec<f32>,
    pvx: Vec<f32>,
    pvy: Vec<f32>,

    // density grid for surface render
    density: Vec<f32>,

    // params
    pub gravity: f32,
    pub flip_ratio: f32,
    pub viscosity: f32,
    pub iters: u32,
}

#[inline] fn idx_u(i: usize, j: usize, gw: usize) -> usize { i + (gw + 1) * j }
#[inline] fn idx_v(i: usize, j: usize, gw: usize) -> usize { i + gw * j }
#[inline] fn idx_c(i: usize, j: usize, gw: usize) -> usize { i + gw * j }

#[wasm_bindgen]
impl Sim2D {
    #[wasm_bindgen(constructor)]
    pub fn new(w_px: f32, h_px: f32, cell: f32) -> Sim2D {
        let gw = (w_px / cell).floor() as usize;
        let gh = (h_px / cell).floor() as usize;
        Sim2D {
            gw, gh, cell, w_px, h_px,
            u: vec![0.0; (gw + 1) * gh],
            u_prev: vec![0.0; (gw + 1) * gh],
            u_w: vec![0.0; (gw + 1) * gh],
            v: vec![0.0; gw * (gh + 1)],
            v_prev: vec![0.0; gw * (gh + 1)],
            v_w: vec![0.0; gw * (gh + 1)],
            cell_type: vec![1; gw * gh],
            walls: vec![0; gw * gh],
            px: Vec::new(), py: Vec::new(),
            pvx: Vec::new(), pvy: Vec::new(),
            density: vec![0.0; gw * gh],
            gravity: 9.8 * 60.0,  // pixels/s² scaled
            flip_ratio: 0.97,
            viscosity: 0.0,
            iters: 40,
        }
    }

    pub fn grid_w(&self) -> usize { self.gw }
    pub fn grid_h(&self) -> usize { self.gh }
    pub fn cell(&self) -> f32 { self.cell }
    pub fn particle_count(&self) -> usize { self.px.len() }

    /// Pointers for zero-copy reads from JS.
    pub fn px_ptr(&self) -> *const f32 { self.px.as_ptr() }
    pub fn py_ptr(&self) -> *const f32 { self.py.as_ptr() }
    pub fn density_ptr(&self) -> *const f32 { self.density.as_ptr() }
    pub fn walls_ptr(&self) -> *const u8 { self.walls.as_ptr() }

    pub fn reset_particles(&mut self, count: usize, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.px.clear(); self.py.clear();
        self.pvx.clear(); self.pvy.clear();
        let rw = x1 - x0;
        let rh = y1 - y0;
        let cols = ((count as f32 * rw / rh).sqrt()).ceil() as usize;
        let rows = (count as f32 / cols as f32).ceil() as usize;
        let sx = rw / cols as f32;
        let sy = rh / rows as f32;
        // simple LCG for reproducible jitter without rand crate
        let mut seed: u32 = 12345;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        for j in 0..rows {
            for i in 0..cols {
                if self.px.len() >= count { return; }
                self.px.push(x0 + (i as f32 + 0.5) * sx + rnd() * sx * 0.4);
                self.py.push(y0 + (j as f32 + 0.5) * sy + rnd() * sy * 0.4);
                self.pvx.push(0.0);
                self.pvy.push(0.0);
            }
        }
    }

    pub fn splash(&mut self, cx: f32, cy: f32, count: usize, vel: f32) {
        let mut seed: u32 = (cx as u32).wrapping_mul(7919).wrapping_add(cy as u32);
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        for _ in 0..count {
            self.px.push(cx + rnd() * 80.0);
            self.py.push(cy + rnd() * 30.0);
            self.pvx.push(rnd() * vel * 2.0);
            self.pvy.push(rnd() * vel * 2.0 + vel);
        }
    }

    /// Toggle wall at cell (i, j). val = 0 (clear) or 1 (set).
    pub fn paint_wall(&mut self, ci: i32, cj: i32, radius: i32, val: u8) {
        for dj in -radius..=radius {
            for di in -radius..=radius {
                let i = ci + di;
                let j = cj + dj;
                if i < 1 || j < 1 || (i as usize) >= self.gw - 1 || (j as usize) >= self.gh - 1 {
                    continue;
                }
                if di * di + dj * dj > radius * radius { continue; }
                self.walls[idx_c(i as usize, j as usize, self.gw)] = val;
            }
        }
    }

    pub fn clear_walls(&mut self) {
        self.walls.iter_mut().for_each(|w| *w = 0);
    }

    /// Apply external impulse around (cx, cy) — push (sign=1) or pull (sign=-1).
    pub fn apply_force(&mut self, cx: f32, cy: f32, dvx: f32, dvy: f32, radius: f32, pull: bool, dt: f32) {
        let r2 = radius * radius;
        for k in 0..self.px.len() {
            let dx = self.px[k] - cx;
            let dy = self.py[k] - cy;
            let d2 = dx * dx + dy * dy;
            if d2 > r2 { continue; }
            let d = d2.sqrt().max(1.0);
            let fall = 1.0 - d / radius;
            if pull {
                self.pvx[k] -= dx * 0.04 * fall;
                self.pvy[k] -= dy * 0.04 * fall;
            } else {
                self.pvx[k] += dvx * fall * 0.02;
                self.pvy[k] += dvy * fall * 0.02;
                self.pvx[k] += (dx / d) * 200.0 * fall * dt * 60.0 * 0.4;
                self.pvy[k] += (dy / d) * 200.0 * fall * dt * 60.0 * 0.4;
            }
        }
    }

    /// One simulation step.
    pub fn step(&mut self, dt: f32) {
        self.classify_cells();
        self.p2g();
        self.apply_forces(dt);
        self.project();
        self.g2p();
        self.advect(dt);
        self.push_apart(2);
    }

    pub fn compute_density(&mut self) {
        for d in self.density.iter_mut() { *d = 0.0; }
        let sr = 1.5_f32;
        let sr2 = sr * sr;
        let span = sr.ceil() as i32;
        for k in 0..self.px.len() {
            let fx = self.px[k] / self.cell;
            let fy = self.py[k] / self.cell;
            let ci = fx.floor() as i32;
            let cj = fy.floor() as i32;
            for dj in -span..=span {
                for di in -span..=span {
                    let i = ci + di;
                    let j = cj + dj;
                    if i < 0 || j < 0 || (i as usize) >= self.gw || (j as usize) >= self.gh {
                        continue;
                    }
                    let dx = (i as f32 + 0.5) - fx;
                    let dy = (j as f32 + 0.5) - fy;
                    let d2 = dx * dx + dy * dy;
                    if d2 > sr2 { continue; }
                    let v = 1.0 - d2 / sr2;
                    self.density[idx_c(i as usize, j as usize, self.gw)] += v * v;
                }
            }
        }
        // 2-pass blur
        let mut tmp = vec![0.0f32; self.density.len()];
        for _ in 0..2 {
            for j in 1..self.gh - 1 {
                for i in 1..self.gw - 1 {
                    tmp[idx_c(i, j, self.gw)] = (
                        self.density[idx_c(i - 1, j, self.gw)] +
                        self.density[idx_c(i + 1, j, self.gw)] +
                        self.density[idx_c(i, j - 1, self.gw)] +
                        self.density[idx_c(i, j + 1, self.gw)] +
                        self.density[idx_c(i, j, self.gw)] * 2.0
                    ) / 6.0;
                }
            }
            self.density.copy_from_slice(&tmp);
        }
    }

    // ---------- internals ----------

    fn classify_cells(&mut self) {
        for k in 0..self.cell_type.len() {
            self.cell_type[k] = if self.walls[k] != 0 { 2 } else { 1 };
        }
        // borders solid
        for i in 0..self.gw {
            self.cell_type[idx_c(i, 0, self.gw)] = 2;
            self.cell_type[idx_c(i, self.gh - 1, self.gw)] = 2;
        }
        for j in 0..self.gh {
            self.cell_type[idx_c(0, j, self.gw)] = 2;
            self.cell_type[idx_c(self.gw - 1, j, self.gw)] = 2;
        }
        for k in 0..self.px.len() {
            let i = (self.px[k] / self.cell) as i32;
            let j = (self.py[k] / self.cell) as i32;
            if i > 0 && j > 0 && (i as usize) < self.gw - 1 && (j as usize) < self.gh - 1 {
                let c = idx_c(i as usize, j as usize, self.gw);
                if self.cell_type[c] != 2 { self.cell_type[c] = 0; }
            }
        }
    }

    fn p2g(&mut self) {
        for x in self.u.iter_mut() { *x = 0.0; }
        for x in self.u_w.iter_mut() { *x = 0.0; }
        for x in self.v.iter_mut() { *x = 0.0; }
        for x in self.v_w.iter_mut() { *x = 0.0; }

        for k in 0..self.px.len() {
            let x = self.px[k];
            let y = self.py[k];
            let vx = self.pvx[k];
            let vy = self.pvy[k];
            // u-component: (i*cell, (j+0.5)*cell)
            let fx = x / self.cell;
            let fy = y / self.cell - 0.5;
            let i = fx.floor() as i32;
            let j = fy.floor() as i32;
            let tx = fx - i as f32;
            let ty = fy - j as f32;
            if i >= 0 && (i as usize) < self.gw && j >= 0 && (j as usize) < self.gh {
                let i = i as usize; let j = j as usize;
                let w00 = (1.0 - tx) * (1.0 - ty);
                let w10 = tx * (1.0 - ty);
                let w01 = (1.0 - tx) * ty;
                let w11 = tx * ty;
                self.u[idx_u(i, j, self.gw)] += vx * w00; self.u_w[idx_u(i, j, self.gw)] += w00;
                if i + 1 <= self.gw {
                    self.u[idx_u(i + 1, j, self.gw)] += vx * w10; self.u_w[idx_u(i + 1, j, self.gw)] += w10;
                }
                if j + 1 < self.gh {
                    self.u[idx_u(i, j + 1, self.gw)] += vx * w01; self.u_w[idx_u(i, j + 1, self.gw)] += w01;
                    if i + 1 <= self.gw {
                        self.u[idx_u(i + 1, j + 1, self.gw)] += vx * w11; self.u_w[idx_u(i + 1, j + 1, self.gw)] += w11;
                    }
                }
            }
            // v-component: ((i+0.5)*cell, j*cell)
            let fx = x / self.cell - 0.5;
            let fy = y / self.cell;
            let i = fx.floor() as i32;
            let j = fy.floor() as i32;
            let tx = fx - i as f32;
            let ty = fy - j as f32;
            if i >= 0 && (i as usize) < self.gw && j >= 0 && (j as usize) < self.gh {
                let i = i as usize; let j = j as usize;
                let w00 = (1.0 - tx) * (1.0 - ty);
                let w10 = tx * (1.0 - ty);
                let w01 = (1.0 - tx) * ty;
                let w11 = tx * ty;
                self.v[idx_v(i, j, self.gw)] += vy * w00; self.v_w[idx_v(i, j, self.gw)] += w00;
                if i + 1 < self.gw {
                    self.v[idx_v(i + 1, j, self.gw)] += vy * w10; self.v_w[idx_v(i + 1, j, self.gw)] += w10;
                }
                if j + 1 <= self.gh {
                    self.v[idx_v(i, j + 1, self.gw)] += vy * w01; self.v_w[idx_v(i, j + 1, self.gw)] += w01;
                    if i + 1 < self.gw {
                        self.v[idx_v(i + 1, j + 1, self.gw)] += vy * w11; self.v_w[idx_v(i + 1, j + 1, self.gw)] += w11;
                    }
                }
            }
        }

        for k in 0..self.u.len() {
            if self.u_w[k] > 0.0 { self.u[k] /= self.u_w[k]; }
        }
        for k in 0..self.v.len() {
            if self.v_w[k] > 0.0 { self.v[k] /= self.v_w[k]; }
        }
        self.u_prev.copy_from_slice(&self.u);
        self.v_prev.copy_from_slice(&self.v);
    }

    fn apply_forces(&mut self, dt: f32) {
        for j in 0..=self.gh {
            for i in 0..self.gw {
                self.v[idx_v(i, j, self.gw)] += self.gravity * dt;
            }
        }
        if self.viscosity > 0.0 {
            let k = self.viscosity.min(0.45);
            let u_tmp = self.u.clone();
            let v_tmp = self.v.clone();
            let mut u_out = self.u.clone();
            let mut v_out = self.v.clone();
            for j in 1..self.gh - 1 {
                for i in 1..self.gw {
                    let c = idx_u(i, j, self.gw);
                    u_out[c] = u_tmp[c] + k * (
                        (u_tmp[c - 1] + u_tmp[c + 1] + u_tmp[c - (self.gw + 1)] + u_tmp[c + (self.gw + 1)]) * 0.25
                        - u_tmp[c]
                    );
                }
            }
            for j in 1..self.gh {
                for i in 1..self.gw - 1 {
                    let c = idx_v(i, j, self.gw);
                    v_out[c] = v_tmp[c] + k * (
                        (v_tmp[c - 1] + v_tmp[c + 1] + v_tmp[c - self.gw] + v_tmp[c + self.gw]) * 0.25
                        - v_tmp[c]
                    );
                }
            }
            self.u = u_out;
            self.v = v_out;
        }
    }

    fn enforce_solid_boundary(&mut self) {
        for j in 0..self.gh {
            for i in 0..self.gw {
                if self.cell_type[idx_c(i, j, self.gw)] == 2 {
                    self.u[idx_u(i, j, self.gw)] = 0.0;
                    self.u[idx_u(i + 1, j, self.gw)] = 0.0;
                    self.v[idx_v(i, j, self.gw)] = 0.0;
                    self.v[idx_v(i, j + 1, self.gw)] = 0.0;
                }
            }
        }
        for j in 0..self.gh {
            self.u[idx_u(0, j, self.gw)] = 0.0;
            self.u[idx_u(self.gw, j, self.gw)] = 0.0;
        }
        for i in 0..self.gw {
            self.v[idx_v(i, 0, self.gw)] = 0.0;
            self.v[idx_v(i, self.gh, self.gw)] = 0.0;
        }
    }

    fn project(&mut self) {
        self.enforce_solid_boundary();
        let overrelax = 1.9;
        for _ in 0..self.iters {
            for j in 1..self.gh - 1 {
                for i in 1..self.gw - 1 {
                    if self.cell_type[idx_c(i, j, self.gw)] != 0 { continue; }
                    let s_l = if self.cell_type[idx_c(i - 1, j, self.gw)] != 2 { 1.0 } else { 0.0 };
                    let s_r = if self.cell_type[idx_c(i + 1, j, self.gw)] != 2 { 1.0 } else { 0.0 };
                    let s_u = if self.cell_type[idx_c(i, j - 1, self.gw)] != 2 { 1.0 } else { 0.0 };
                    let s_d = if self.cell_type[idx_c(i, j + 1, self.gw)] != 2 { 1.0 } else { 0.0 };
                    let s = s_l + s_r + s_u + s_d;
                    if s == 0.0 { continue; }
                    let div = self.u[idx_u(i + 1, j, self.gw)] - self.u[idx_u(i, j, self.gw)]
                            + self.v[idx_v(i, j + 1, self.gw)] - self.v[idx_v(i, j, self.gw)];
                    let dp = -div / s * overrelax;
                    self.u[idx_u(i, j, self.gw)]     -= dp * s_l;
                    self.u[idx_u(i + 1, j, self.gw)] += dp * s_r;
                    self.v[idx_v(i, j, self.gw)]     -= dp * s_u;
                    self.v[idx_v(i, j + 1, self.gw)] += dp * s_d;
                }
            }
        }
    }

    fn sample_u(&self, gx: f32, gy: f32, prev: bool) -> f32 {
        let fx = gx;
        let fy = gy - 0.5;
        let i = fx.floor() as i32;
        let j = fy.floor() as i32;
        if i < 0 || (i as usize) >= self.gw || j < 0 || (j + 1) as usize >= self.gh {
            return 0.0;
        }
        let i = i as usize; let j = j as usize;
        let tx = fx - i as f32;
        let ty = fy - j as f32;
        let arr = if prev { &self.u_prev } else { &self.u };
        let a = arr[idx_u(i, j, self.gw)];
        let b = arr[idx_u(i + 1, j, self.gw)];
        let c = arr[idx_u(i, j + 1, self.gw)];
        let d = arr[idx_u(i + 1, j + 1, self.gw)];
        (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty
    }

    fn sample_v(&self, gx: f32, gy: f32, prev: bool) -> f32 {
        let fx = gx - 0.5;
        let fy = gy;
        let i = fx.floor() as i32;
        let j = fy.floor() as i32;
        if i < 0 || (i + 1) as usize >= self.gw || j < 0 || (j as usize) >= self.gh {
            return 0.0;
        }
        let i = i as usize; let j = j as usize;
        let tx = fx - i as f32;
        let ty = fy - j as f32;
        let arr = if prev { &self.v_prev } else { &self.v };
        let a = arr[idx_v(i, j, self.gw)];
        let b = arr[idx_v(i + 1, j, self.gw)];
        let c = arr[idx_v(i, j + 1, self.gw)];
        let d = arr[idx_v(i + 1, j + 1, self.gw)];
        (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty
    }

    fn g2p(&mut self) {
        let flip = self.flip_ratio;
        for k in 0..self.px.len() {
            let gx = self.px[k] / self.cell;
            let gy = self.py[k] / self.cell;
            let new_u = self.sample_u(gx, gy, false);
            let old_u = self.sample_u(gx, gy, true);
            let new_v = self.sample_v(gx, gy, false);
            let old_v = self.sample_v(gx, gy, true);
            let pic_u = new_u;
            let flip_u = self.pvx[k] + (new_u - old_u);
            let pic_v = new_v;
            let flip_v = self.pvy[k] + (new_v - old_v);
            self.pvx[k] = pic_u * (1.0 - flip) + flip_u * flip;
            self.pvy[k] = pic_v * (1.0 - flip) + flip_v * flip;
        }
    }

    fn advect(&mut self, dt: f32) {
        let cell = self.cell;
        let w_px = self.w_px;
        let h_px = self.h_px;
        let gw = self.gw;
        let gh = self.gh;
        for k in 0..self.px.len() {
            self.px[k] += self.pvx[k] * dt;
            self.py[k] += self.pvy[k] * dt;
            // wall push-out
            let i = (self.px[k] / cell) as i32;
            let j = (self.py[k] / cell) as i32;
            if i > 0 && j > 0 && (i as usize) < gw - 1 && (j as usize) < gh - 1 {
                if self.cell_type[idx_c(i as usize, j as usize, gw)] == 2 {
                    let cx = (i as f32 + 0.5) * cell;
                    let cy = (j as f32 + 0.5) * cell;
                    let dx = self.px[k] - cx;
                    let dy = self.py[k] - cy;
                    if dx.abs() > dy.abs() {
                        self.px[k] = if dx > 0.0 { (i + 1) as f32 * cell + 0.1 } else { i as f32 * cell - 0.1 };
                        self.pvx[k] *= -0.2;
                    } else {
                        self.py[k] = if dy > 0.0 { (j + 1) as f32 * cell + 0.1 } else { j as f32 * cell - 0.1 };
                        self.pvy[k] *= -0.2;
                    }
                }
            }
            // domain bounds
            let m = cell + 1.0;
            if self.px[k] < m { self.px[k] = m; if self.pvx[k] < 0.0 { self.pvx[k] = 0.0; } }
            if self.px[k] > w_px - m { self.px[k] = w_px - m; if self.pvx[k] > 0.0 { self.pvx[k] = 0.0; } }
            if self.py[k] < m { self.py[k] = m; if self.pvy[k] < 0.0 { self.pvy[k] = 0.0; } }
            if self.py[k] > h_px - m { self.py[k] = h_px - m; if self.pvy[k] > 0.0 { self.pvy[k] = 0.0; } }
        }
    }

    fn push_apart(&mut self, iters: u32) {
        let min_dist = self.cell * 0.55;
        let min_dist2 = min_dist * min_dist;
        let cs = self.cell;
        let cw = (self.w_px / cs).ceil() as usize + 1;
        let ch = (self.h_px / cs).ceil() as usize + 1;
        for _ in 0..iters {
            let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); cw * ch];
            for k in 0..self.px.len() {
                let bx = (self.px[k] / cs) as usize;
                let by = (self.py[k] / cs) as usize;
                let bi = bx + by * cw;
                if bi < buckets.len() { buckets[bi].push(k as u32); }
            }
            for k in 0..self.px.len() {
                let bx = (self.px[k] / cs) as i32;
                let by = (self.py[k] / cs) as i32;
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        let nx = bx + dx;
                        let ny = by + dy;
                        if nx < 0 || ny < 0 || (nx as usize) >= cw || (ny as usize) >= ch { continue; }
                        let bi = nx as usize + ny as usize * cw;
                        for &j_u32 in &buckets[bi] {
                            let j = j_u32 as usize;
                            if j <= k { continue; }
                            let ddx = self.px[j] - self.px[k];
                            let ddy = self.py[j] - self.py[k];
                            let d2 = ddx * ddx + ddy * ddy;
                            if d2 >= min_dist2 || d2 < 1e-6 { continue; }
                            let d = d2.sqrt();
                            let push = (min_dist - d) * 0.5;
                            let nx2 = ddx / d;
                            let ny2 = ddy / d;
                            self.px[k] -= nx2 * push;
                            self.py[k] -= ny2 * push;
                            self.px[j] += nx2 * push;
                            self.py[j] += ny2 * push;
                        }
                    }
                }
            }
            // re-clamp
            let m = cs + 1.0;
            for k in 0..self.px.len() {
                if self.px[k] < m { self.px[k] = m; }
                if self.px[k] > self.w_px - m { self.px[k] = self.w_px - m; }
                if self.py[k] < m { self.py[k] = m; }
                if self.py[k] > self.h_px - m { self.py[k] = self.h_px - m; }
            }
        }
    }
}
