macro_rules! sqr {
  ($x: expr) => {{
    $x * $x
  }};
}

pub fn determinant(m: &[f32], n: usize) -> f32 {
  assert_eq!(m.len(), n * n);
  assert_ne!(n, 0);
  if n == 1 {
    return m[0];
  } else if n == 2 {
    return m[0] * m[3] - m[1] * m[2];
  } else if n == 3 {
    // also handle case 3 explicitly for efficiency
    return m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
      - (m[2] * m[4] * m[6] + m[1] * m[3] * m[8] + m[0] * m[5] * m[7]);
  }
  let minor = |i, j| {
    let mut out = vec![-42.; sqr!(n - 1)];
    let mut idx = 0;
    for x in 0..n {
      for y in 0..n {
        if x != i && y != j {
          out[idx] = m[x + y * n];
          idx += 1;
        }
      }
    }
    out
  };
  // just naively do the first row for now
  (0..n)
    .map(|i| {
      let j = 0;
      let m = minor(i, j);
      let det = determinant(&m, n - 1);
      m[i] * det.copysign(if (i + j) % 2 == 0 { 1. } else { -1. })
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

      out[i + j * n] = determinant(&buf, n - 1).copysign(if (i + j) % 2 == 0 { 1. } else { -1. });
    }
  }
}

pub fn transpose(x: &mut [f32], n: usize) {
  assert_eq!(x.len(), n * n);
  for i in 0..n {
    for j in 0..i {
      x.swap(i + j * n, j + i * n);
    }
  }
}

pub fn invert_matrix(x: &[f32], n: usize, out: &mut [f32]) {
  assert_eq!(x.len(), n * n);
  assert_eq!(x.len(), out.len());
  let det = determinant(x, n);
  if det.abs() < f32::EPSILON {
    return;
  }
  cofactor(x, n, out);
  transpose(out, n);
  for v in out.iter_mut() {
    *v /= det;
  }
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
  invert_matrix(&mat, 3, &mut out[..]);
  assert_eq!(mat, out);
}
