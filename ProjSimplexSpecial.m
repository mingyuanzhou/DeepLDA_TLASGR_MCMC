function Phinew  =   ProjSimplexSpecial(Phitmp,Phiold,epsilon)
% Cong Yulai
% 2016 03 18
% Phinew = Phitmp - (sum(Phitmp)-1) * Phiold

%% 
if nargin < 3
    epsilon   =   eps   ;
end

Phinew = Phitmp - bsxfun(@times, sum(Phitmp,1)-1, Phiold)   ;

if nnz(Phinew <= 0)
%     Phinew  =   max(epsilon,Phinew)     ;
    Phinew  =   abs(Phinew) ;
    Phinew  =   bsxfun(@rdivide, Phinew, max(realmin,sum(Phinew,1)) )   ;
end

