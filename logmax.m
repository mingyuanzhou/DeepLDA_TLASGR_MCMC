function y  =   logmax(x)
if nnz(x < 0)
    error('Negative input for Log function!')
else
    y   =   log(max(realmin, x))    ;
end