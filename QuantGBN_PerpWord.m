function [PerpWordCount,PerpWordCountTrain,PerpWordCountSet,PerpWordCountSetTrain]  ...
            =   QuantGBN_PerpWord(X,Xtest,ParaGlobal,K,Tcurrent,DataType,SuPara,Settings)
% Yulai Cong
% 2016 03 09

%% Settings
ac      =   SuPara.ac   ;       bc      =   SuPara.bc   ;
a0pj    =   SuPara.a0pj   ;     b0pj    =   SuPara.b0pj   ;
e0cj    =   SuPara.e0cj   ;     f0cj    =   SuPara.f0cj   ;
e0c0    =   SuPara.e0c0    ;    f0c0    =   SuPara.f0c0    ;
a0gamma =   SuPara.a0gamma   ;  b0gamma      =   SuPara.b0gamma   ;
eta     =   SuPara.eta    ;

TestBurnin      =   Settings.TestBurnin         ;     
TestCollection    =   Settings.TestCollection     ;        
TestSampleSpace    =   Settings.TestSampleSpace    ;

%%
Yflagtrain   =   (X > 0)   ;
Yflagtest   =   (Xtest > 0)   ;
[V,N] = size(X);

[ii,jj]=find(X>eps);    %Treat values smaller than eps as 0
iijj=find(X>eps);

%% Initial ParaGlobal
Phi     =   ParaGlobal.Phi  ;
r_k     =   ParaGlobal.r_k  ;
gamma0    =   ParaGlobal.gamma0   ;     
c0    =   ParaGlobal.c0   ;
if strcmp(DataType, 'Positive')
    a_jmean     =   ParaGlobal.a_jmean  ;
end
c_jmean     =   ParaGlobal.c_jmean  ;

%% Initial ParaLocal
c_j     =   cell(Tcurrent+1,1);    Theta = cell(Tcurrent,1);    

if strcmp(DataType, 'Positive')
    a_j   =   a_jmean * ones(1,N);
end
for t=1:Tcurrent+1
    c_j{t}=ones(1,N)*c_jmean(t);
end
p_j = Calculate_pj(c_j,Tcurrent);

for t=Tcurrent:-1:1
    Theta{t}    =   ones(K(t),N)/K(t);
end

%% Initial Registers
Xt_to_t1=cell(Tcurrent,1);

PhiTheta    =   0   ;   
CountNum    =   0   ;   
loglikeset     =   []  ;

%% Sample ParaLocal
for iter    =   1 : (TestBurnin + TestCollection * TestSampleSpace )
    tic
    %%==================================== Upward Pass ===================================
    for t   =   1:Tcurrent
        if t    ==  1 %&& Tcurrent==1
            switch DataType
                case 'Positive'
                    Rate = Phi{1}*Theta{1};
                    Rate = 2*sqrt(a_j(jj)'.*X(iijj).*Rate(iijj));
                    M  = Truncated_bessel_rnd( Rate );
                    a_j = randg(full(sparse(1,jj,M,1,N))+ac) ./ (bc+sum(X,1));
                    Xt = sparse(ii,jj,M,V,N);   X1   =   Xt  ; 
                case 'Binary'
                    Rate = Phi{1}*Theta{1};
                    M = truncated_Poisson_rnd(Rate(iijj));
                    Xt = sparse(ii,jj,M,V,N);
                case 'Count'
                    Xt = sparse(X);
            end
            Xt_to_t1{t}     =   Multrnd_Matrix_mex_fast_v1(Xt,Phi{t},Theta{t});
        else
            Xt_to_t1{t}     =   CRT_Multrnd_Matrix(sparse(Xt_to_t1{t-1}),Phi{t},Theta{t});
        end
    end
    %%==================================== Downward Pass ===================================
    %%====================   Sample Theta  ========================
    if iter >= min(TestBurnin/2,10)
        if Tcurrent > 1
            p_j{2} = betarnd(  sum(Xt_to_t1{1},1)+a0pj   ,   sum(Theta{2},1)+b0pj  );
        else
            p_j{2} = betarnd(  sum(Xt_to_t1{1},1)+a0pj   ,   sum(r_k,1)+b0pj  );
        end
        p_j{2} = min( max(p_j{2},realmin) , 1-realmin);
        c_j{2} = (1-p_j{2})./p_j{2};
        for t   =   3:(Tcurrent+1)
            if t    ==  Tcurrent+1
                c_j{t} = randg(sum(r_k)*ones(1,N)+e0cj) ./ (sum(Theta{t-1},1)+f0cj);
            else
                c_j{t} = randg(sum(Theta{t},1)+e0cj) ./ (sum(Theta{t-1},1)+f0cj);
            end
        end
        p_j_temp = Calculate_pj(c_j,Tcurrent);
        p_j(3:end)=p_j_temp(3:end);
    end
    
    for t  =   Tcurrent:-1:1
        if t    ==  Tcurrent
            shape = r_k;
        else
            shape = Phi{t+1}*Theta{t+1};
        end
        Theta{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1{t})), 1 ./ max(realmin, c_j{t+1}-log(max(1-p_j{t},realmin))) ); %            figure(26),imagesc(Theta{1}),drawnow
        if nnz(isnan(Theta{t})) | nnz(isinf(Theta{t})) | nnz(sum(Theta{t})==0)
            warning(['Theta Nan',num2str(nnz(isnan(Theta{t}))),'_Inf',num2str(nnz(isinf(Theta{t}))),'_ORsum=0]']);
            Theta{t}(isnan(Theta{t}))   =   0   ;
        end
    end
    
    Timetmp     =   toc     ;
    if mod(iter,10)==0
        fprintf('Quatifying Perplexity Layer: %d, iter: %d, TimePerIter: %d seconds. \n',Tcurrent,iter,Timetmp);
    end
    
    %%==================================== Average ===================================
    if (iter > TestBurnin) && (mod(iter-TestBurnin,TestSampleSpace)==0)
        tmp     =   Phi{1}*Theta{1}     ;
        if nnz(isnan(tmp)) | nnz(isinf(tmp))
            tmp     =   Phi{1}*Theta{1}     ;
            if nnz(isnan(tmp)) | nnz(isinf(tmp))
                a = 1;
            end
        end        
        PhiTheta    =   PhiTheta + tmp;
        CountNum    =   CountNum + 1    ;
        EPhiTheta   =   PhiTheta / CountNum     ;
        temp    =   bsxfun(@rdivide, EPhiTheta, max(realmin,sum(EPhiTheta,1)) );
        loglikesettrain(CountNum)   =   sum(X(Yflagtrain).*log(max(realmin,temp(Yflagtrain)))) / sum(X(:));
        loglikeset(CountNum)   =   sum(Xtest(Yflagtest).*log(max(realmin,temp(Yflagtest)))) / sum(Xtest(:));
    end
    
end     %%=======================  One Testing iteration End  ===========================

PerpWordCountSetTrain    =   exp( -loglikesettrain )  ;
PerpWordCountTrain   =   PerpWordCountSetTrain(end)   ;
PerpWordCountSet    =   exp( -loglikeset )  ;
PerpWordCount   =   PerpWordCountSet(end)   ;



