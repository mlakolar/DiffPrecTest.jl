module SkronTest

using Test
using DiffPrecTest: svec, skron
using SparseArrays, Random, LinearAlgebra

function Q(p::Int)
  _c = 1. / sqrt(2.)
  rp = div((p+1)*p,2)
  Q = spzeros( rp,  p*p )
  ind = 0
  for ci=1:p
    for ri=ci:p
      ind += 1
      if ri == ci
        Q[ind, (ci-1)*p + ri] = 1.
      else
        Q[ind, (ci-1)*p + ri] = _c
        Q[ind, (ri-1)*p + ci] = _c
      end
    end
  end
  Q
end
  
function Qt(p::Int)
  _c = 1. / sqrt(2.)
  rp = div((p+1)*p,2)
  Q = spzeros( p*p, rp )
  ind = 0
  for ci=1:p
    for ri=ci:p
      ind += 1
      if ri == ci
        Q[(ci-1)*p + ri, ind] = 1.
      else
        Q[(ci-1)*p + ri, ind] = _c
        Q[(ri-1)*p + ci, ind] = _c
      end
    end
  end
  Q
end


@testset "skron" begin
  p = 5
  x = randn(p)
  y = randn(p)

  num_row_reduced = div(p*(p+1),2)
  o = zeros( num_row_reduced )
  for i=1:num_row_reduced
      o[i] = svec(x, y, i)
  end

  @test o ≈ Q(p) * kron(x, y)

  o2 = zeros( num_row_reduced, num_row_reduced )
  for ri=1:num_row_reduced
      for ci=1:num_row_reduced
          o2[ri, ci] = skron(x, y, ri, ci) 
      end
  end
 
  @test o2 ≈ Q(p) * ( kron(x*x', y*y') + kron(y*y', x*x') ) * Qt(p) / 2.

  @test skron(x, y, 1:4, 3:6) ≈ (Q(p) * ( kron(x*x', y*y') + kron(y*y', x*x') ) * Qt(p) / 2.)[1:4, 3:6]

  # Q_Sx_skron_Sy_Q
  X = randn(100, 5)
  Y = randn(100, 5)
  Sx = Symmetric( X'*X / 100 )
  Sy = Symmetric( Y'*Y / 100 )

  @test skron(Sx, Sy) ≈ Q(p) * ( kron(Sx, Sy) + kron(Sy, Sx) ) * Qt(p) / 2.

  I = 1:num_row_reduced
  @test skron(Sx, Sy, I) ≈ Q(p) * ( kron(Sx, Sy) + kron(Sy, Sx) ) * Qt(p) / 2.

  I = 3:10
  J = 4:7
  @test skron(Sx, Sy, I, J) ≈ (Q(p) * ( kron(Sx, Sy) + kron(Sy, Sx) ) * Qt(p) / 2.)[I, J]

  I = [2, 5, 7, 8]    
  @test skron(Sx, Sy, I) ≈ (Q(p) * ( kron(Sx, Sy) + kron(Sy, Sx) ) * Qt(p) / 2.)[I, I]

  I = 10:15
  @test skron(Sx, Sy, I) ≈ (Q(p) * ( kron(Sx, Sy) + kron(Sy, Sx) ) * Qt(p) / 2.)[I, I]

  # Q Sx skron yy' Qt
  yy = view(Y, 1, :) * view(Y, 1, :)'
  @test skron(Sx, view(Y, 1, :), I, I) ≈ (Q(p) * ( kron(Sx, yy) + kron(yy, Sx) ) * Qt(p) / 2.)[I, I]
end


@testset "svec" begin
  p = 5
  X = randn(100, p)
  Y = randn(100, p)
  Sx = Symmetric( X'*X / 100 )
  Sy = Symmetric( Y'*Y / 100 )

  # Q_Sx_minus_Sy
  ox = svec(Sx)
  oy = svec(Sy)
  @test ox - oy ≈ Q(p) * vec( Sx - Sy )

  I = 10:15
  @test svec(Sx, I) ≈ (Q(p) * vec( Sx ))[I]

  I = [2, 5, 7, 8]    
  @test svec(Sy, I) ≈ (Q(p) * vec( Sy ))[I]    

  @test svec(view(X, 1, :), view(Y, 1, :), I) ≈ (Q(p) * kron(view(X, 1, :), view(Y, 1, :)))[I]
end

end