#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

macro_rules! sqr {
  ($x: expr) => {{
    $x * $x
  }};
}

/// Error when a matrix is singular and thus non-invertible.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct SingularMatrix;

/// Computes the determinant of m of size (n x n), panics if n == 0
pub fn determinant(m: &[f32], n: usize) -> f32 {
  assert_eq!(m.len(), n * n);
  assert_ne!(n, 0);
  match n {
    0 => unreachable!(),
    1 => return m[0],
    2 => return m[0] * m[3] - m[1] * m[2],
    3 =>
      return m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
        - (m[2] * m[4] * m[6] + m[1] * m[3] * m[8] + m[0] * m[5] * m[7]),
    _ => {},
  };
  let minor = |i, j| {
    let mut out = vec![0.; sqr!(n - 1)];
    let mut idx = 0;
    for x in 0..n {
      for y in 0..n {
        if x != i && y != j {
          out[idx] = m[x + y * n];
          idx += 1;
        }
      }
    }
    debug_assert_eq!(idx, sqr!(n - 1));
    out
  };
  // just naively do the first row for now
  (0..n)
    .map(|i| {
      let j = 0;
      let factor = m[i + j * n];
      if factor == 0. {
        return 0.;
      }
      let det = determinant(&minor(i, j), n - 1);
      factor * det.copysign(if (i + j) % 2 == 0 { 1. } else { -1. })
    })
    .sum()
}

pub fn cofactor(m: &[f32], n: usize, out: &mut [f32]) {
  // buffer for the current minor
  let mut buf = vec![0.; sqr!(n - 1)];
  for i in 0..n {
    for j in 0..n {
      let mut idx = 0;
      for x in 0..n {
        if x == i {
          continue;
        }
        for y in 0..n {
          if y == j {
            continue;
          }
          buf[idx] = m[x + y * n];
          idx += 1;
        }
      }
      debug_assert_eq!(idx, sqr!(n - 1));

      out[i + j * n] = determinant(&buf, n - 1).copysign(if (i + j) % 2 == 0 { 1. } else { -1. });
    }
  }
}

/// Transposes x=[n,n] in-place.
pub fn transpose(x: &mut [f32], n: usize) {
  assert_eq!(x.len(), n * n);
  for i in 0..n {
    for j in 0..i {
      x.swap(i + j * n, j + i * n);
    }
  }
}

/// Inverts x=[n,n] into out=[n,n]
pub fn invert(x: &[f32], n: usize, out: &mut [f32]) -> Result<(), SingularMatrix> {
  assert_eq!(x.len(), n * n);
  assert_eq!(x.len(), out.len());
  let det = determinant(x, n);
  if det.abs() < f32::EPSILON {
    return Err(SingularMatrix);
  }
  cofactor(x, n, out);
  transpose(out, n);
  for v in out.iter_mut() {
    *v /= det;
  }
  Ok(())
}

/// Multiplies l=[i,j] * r=[j*k] into out=[i*k].
pub fn matmul(l: &[f32], r: &[f32], (i, j, k): (usize, usize, usize), out: &mut [f32]) {
  assert_eq!(l.len(), i * j);
  assert_eq!(r.len(), j * k);
  assert_eq!(out.len(), i * k);
  for a in 0..i {
    for c in 0..k {
      for b in 0..j {
        out[a + c * k] += l[a + b * j] * r[b + c * k]
      }
    }
  }
}

/// Randomizes the elements in an array to be between \[0,1\].
/// Deterministic given the same series of inputs.
/// Should not be used for crytographic purposes.
pub fn rand(vs: &mut [f32]) {
  static mut SEED: f32 = 4.2;
  for v in vs.iter_mut() {
    unsafe {
      *v = (((37882.93 * *v).fract() + SEED).cos() + 1.) / 2.;
      SEED += *v;
    }
  }
}

#[test]
fn rand_test() {
  let mut a = [13.; 10];
  let mut b = [13.; 10];
  rand(&mut a[..]);
  // check that it actually modifies the seed between runs
  rand(&mut b[..]);
  assert_ne!(a, b);
}

#[test]
fn identity_test() {
  let mut mat = vec![0.; 9];
  for i in 0..3 {
    mat[i + 3 * i] = 1.;
  }
  let det = determinant(&mat, 3);
  assert_eq!(det, 1.);
  let mut out = vec![0.; 9];
  invert(&mat, 3, &mut out[..]);
  assert_eq!(mat, out);

  out.fill(0.);

  matmul(&mat, &mat, (3, 3, 3), &mut out);
  assert_eq!(mat, out);
}

/// Module for using generic constants with matrix operations.
/// Still recursively relies on the above due to limitations on const-generics currently.
pub mod constant {
  use super::SingularMatrix;
  fn minor<const N: usize>(m: &[f32; N * N], i: usize, j: usize) -> [f32; (N - 1) * (N - 1)]
  where
    [f32; (N - 1) * (N - 1)]: , {
    let mut out = [0.; (N - 1) * (N - 1)];
    let mut idx = 0;
    for x in 0..N {
      for y in 0..N {
        if x != i && y != j {
          out[idx] = m[x + y * N];
          idx += 1;
        }
      }
    }
    out
  }
  pub fn determinant<const N: usize>(m: &[f32; N * N]) -> f32
  where
    [f32; (N - 1) * (N - 1)]: , {
    assert_ne!(N, 0);
    match N {
      0 => unreachable!(),
      1 => return m[0],
      2 => return m[0] * m[3] - m[1] * m[2],
      3 =>
        return m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
          - (m[2] * m[4] * m[6] + m[1] * m[3] * m[8] + m[0] * m[5] * m[7]),
      _ => {},
    };
    let mut sum = 0.;
    for i in 0..N {
      let j = 0;
      let factor = m[i + j * N];
      if factor == 0. {
        return 0.;
      }
      // can't recurse here because there's no way to demonstrate it doesn't recur forever yet.
      let det = super::determinant(&minor::<N>(m, i, j), N - 1);
      sum += factor * det.copysign(if (i + j) % 2 == 0 { 1. } else { -1. })
    }
    sum
  }
  pub fn transpose<const N: usize>(x: &mut [f32; N * N]) {
    for i in 0..N {
      for j in 0..i {
        x.swap(i + j * N, j + i * N);
      }
    }
  }
  pub fn cofactor<const N: usize>(m: &[f32; N * N], out: &mut [f32; N * N])
  where
    [f32; (N - 1) * (N - 1)]: , {
    // buffer for the current minor
    let mut buf = [0.; (N - 1) * (N - 1)];
    for i in 0..N {
      for j in 0..N {
        let mut idx = 0;
        for x in 0..N {
          if x == i {
            continue;
          }
          for y in 0..N {
            if y == j {
              continue;
            }
            buf[idx] = m[x + y * N];
            idx += 1;
          }
        }
        debug_assert_eq!(idx, sqr!(N-1));

        // FIXME change to use this modules when possible
        out[i + j * N] =
          super::determinant(&buf, N - 1).copysign(if (i + j) % 2 == 0 { 1. } else { -1. });
      }
    }
  }
  pub fn invert<const N: usize>(
    x: &[f32; N * N],
    out: &mut [f32; N * N],
  ) -> Result<(), SingularMatrix>
  where
    [f32; (N - 1) * (N - 1)]: , {
    let det = determinant::<N>(x);
    if det.abs() < f32::EPSILON {
      return Err(SingularMatrix);
    }
    cofactor::<N>(x, out);
    transpose::<N>(out);
    for v in out.iter_mut() {
      *v /= det;
    }
    Ok(())
  }
  pub fn matmul<const I: usize, const J: usize, const K: usize>(
    l: &[f32; I * J],
    r: &[f32; J * K],
    out: &mut [f32; I * K],
  ) {
    for a in 0..I {
      for c in 0..K {
        for b in 0..J {
          out[a + c * K] += l[a + b * J] * r[b + c * K];
        }
      }
    }
  }
  #[test]
  fn identity_test() {
    let mut mat = [0.; 9];
    for i in 0..3 {
      mat[i + 3 * i] = 1.;
    }
    let det = determinant::<3>(&mat);
    assert_eq!(det, 1.);
    let mut out = [0.; 9];
    invert::<3>(&mat, &mut out);
    assert_eq!(mat, out);

    out.fill(0.);

    matmul::<3, 3, 3>(&mat, &mat, &mut out);
    assert_eq!(mat, out);
  }
}
