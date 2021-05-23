######################################################
# functions related to computation of reduced hessian
######################################################


# Q xx' skron yy' Qt
function skron(
  x::AbstractVector, 
  y::AbstractVector, 
  indRow::AbstractVector,
  indCol::AbstractVector)  

  nRow = length(indRow)
  nCol = length(indCol)
  out = zeros( nRow, nCol )
  for ri=1:nRow
    for ci=1:nCol
      out[ri, ci] = skron(x, y, indRow[ri], indCol[ci])
    end    
  end
  out
end

function skron(
  x::AbstractVector, 
  y::AbstractVector, 
  indRow::Int,
  indCol::Int
  )

  p = length(x)
  row1, col1 = ind2subLowerTriangular(p, indRow)
  row2, col2 = ind2subLowerTriangular(p, indCol)

  vrow = (row1 == col1) ? x[row1] * y[row1] : (x[row1] * y[col1] + y[row1] * x[col1]) / sqrt(2.)
  vcol = (row2 == col2) ? x[row2] * y[row2] : (x[row2] * y[col2] + y[row2] * x[col2]) / sqrt(2.)

  vrow * vcol 
end


# 
# Q Sx skron Sy Qt
function skron(
  Sx::Symmetric, 
  Sy::Symmetric) 

  p = size(Sx, 1)  
  return skron(Sx, Sy, 1:div(p*(p+1),2))
end


function skron(
  Sx::Symmetric, 
  Sy::Symmetric,
  indRow::AbstractVector,
  indCol::AbstractVector)  

  nRow = length(indRow)
  nCol = length(indCol)
  out = zeros( nRow, nCol )
  for ri=1:nRow
    for ci=1:nCol
      out[ri, ci] = skron(Sx, Sy, indRow[ri], indCol[ci])
    end    
  end
  out
end

function skron(
  Sx::Symmetric, 
  Sy::Symmetric,
  I::AbstractVector)

  m = length(I)
  
  out = zeros( m, m )

  for ci = 1:m
    for ri = ci:m
      out[ri, ci] = skron(Sx, Sy, I[ri], I[ci])
      if ri != ci
        out[ci, ri] = out[ri, ci]
      end
    end
  end
  out
end

function skron(
  Sx::Symmetric, 
  Sy::Symmetric, 
  ind_row::Int,
  ind_col::Int
  )

  p = size(Sx, 1)
  row1, col1 = ind2subLowerTriangular(p, ind_row)
  row2, col2 = ind2subLowerTriangular(p, ind_col)

  if row1 == col1
    if row2 == col2
      return Sx[row1, row2] * Sy[row1, row2]
    else
      return ( Sx[row1, row2] * Sy[row1, col2] + Sx[row1, col2] * Sy[row1, row2] ) / sqrt(2.)
    end
  else
    if row2 == col2
      return ( Sx[row1, row2] * Sy[col1, row2] + Sx[col1, row2] * Sy[row1, row2] ) / sqrt(2.)
    else
      return (Sx[row1, row2] * Sy[col1, col2] + 
              Sx[row1, col2] * Sy[col1, row2] + 
              Sx[col1, row2] * Sy[row1, col2] + 
              Sx[col1, col2] * Sy[row1, row2] ) / 2.
    end
  end
end


# 
# Q Sx skron yy' Qt
function skron(
  Sx::Symmetric, 
  y::AbstractVector,
  indRow::AbstractVector,
  indCol::AbstractVector)  

  nRow = length(indRow)
  nCol = length(indCol)
  out = zeros( nRow, nCol )
  for ri=1:nRow
    for ci=1:nCol
      out[ri, ci] = skron(Sx, y, indRow[ri], indCol[ci])
    end    
  end
  out
end

function skron(
  Sx::Symmetric, 
  y::AbstractVector, 
  ind_row::Int,
  ind_col::Int
  )

  p = size(Sx, 1)
  row1, col1 = ind2subLowerTriangular(p, ind_row)
  row2, col2 = ind2subLowerTriangular(p, ind_col)

  if row1 == col1
    if row2 == col2
      return Sx[row1, row2] * y[row1] * y[row2]
    else
      return ( Sx[row1, row2] * y[row1] * y[col2] + Sx[row1, col2] * y[row1] * y[row2] ) / sqrt(2.)
    end
  else
    if row2 == col2
      return ( Sx[row1, row2] * y[col1] * y[row2] + Sx[col1, row2] * y[row1] * y[row2] ) / sqrt(2.)
    else
      return (Sx[row1, row2] * y[col1] * y[col2] + 
              Sx[row1, col2] * y[col1] * y[row2] + 
              Sx[col1, row2] * y[row1] * y[col2] + 
              Sx[col1, col2] * y[row1] * y[row2] ) / 2.
    end
  end
end

######################################################
# functions related to application of Q to Symmetric matrix
######################################################

# Q(S) == svec(S)

function svec(S::Symmetric)
  p = size(S, 1)  
  return svec(S, 1:div(p*(p+1),2))
end

function svec(
  S::Symmetric,
  I::AbstractVector)

  m = length(I)
  out = zeros( m )
  for ri = 1:m
    out[ri] = svec(S, I[ri])
  end
  out
end

function svec(
  S::Symmetric,
  indRow::Int)

  row, col = ind2subLowerTriangular(size(S, 1), indRow)
  (row == col) ? S[row, row] : S[row, col] * sqrt(2.)
end


function svec(
  x::AbstractVector, 
  y::AbstractVector, 
  I::AbstractVector)

  m = length(I)
  out = zeros( m )
  for ri = 1:m
    out[ri] = svec(x, y, I[ri])
  end
  out
end


function svec(
  x::AbstractVector, 
  y::AbstractVector, 
  indRow::Int
  )

  row, col = ind2subLowerTriangular(length(x), indRow)

  if row == col
    return x[row] * y[row]
  else
    return (x[row] * y[col] + y[row] * x[col]) / sqrt(2.)
  end
end