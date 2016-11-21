import DiffPrecTest

reload("DiffPrecTest")

function f1()
  srand(1)
  p = 500
  n = 1000
  Sx = randn(n, p)
  Sx = cov(Sx)

  Sy = randn(n, p)
  Sy = cov(Sy)

  m = sprandn(p, p, 0.1)

  @time DiffPrecTest._mult_aMb(m, Sx, Sy, 5, 10)
end

function f2()
  srand(1)
  p = 500
  n = 1000
  Sx = randn(n, p)
  Sx = cov(Sx)

  Sy = randn(n, p)
  Sy = cov(Sy)

  m = sprandn(p, p, 0.1)

  @time DiffPrecTest._mult_aMb1(m, Sx, Sy, 5, 10)
end


f1()
