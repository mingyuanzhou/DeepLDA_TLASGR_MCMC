function [ParaGlobal,PerpCun,AccCun,ParaLocal]  =   GBN_Online_CommonMode_CongPerp(X_all,prepar,K,T,trial,...
                                                            DataType,dataname,SuPara,Settings)
% Yulai Cong
% 2017 03 18

%% Settings
eta     =   SuPara.eta    ;
ac      =   SuPara.ac   ;       bc      =   SuPara.bc   ;
a0pj    =   SuPara.a0pj   ;     b0pj    =   SuPara.b0pj   ;
e0cj    =   SuPara.e0cj   ;     f0cj    =   SuPara.f0cj   ;
e0c0    =   SuPara.e0c0    ;    f0c0    =   SuPara.f0c0    ;
a0gamma =   SuPara.a0gamma   ;  b0gamma      =   SuPara.b0gamma   ;

IsDisplay   =   Settings.IsDisplay    ;     
        FigureGap  =   Settings.FigureGap      ;
IsTestPointPerp   =   Settings.IsTestPointPerp    ;  
        PointTestLocation   =   Settings.PointTestLocation  ;
IsTestPerpZhou  =   Settings.IsTestPerpZhou     ;
        PerpZhouBurnin  =   Settings.PerpZhouBurnin     ;
        PerpZhouStep    =   Settings.PerpZhouStep       ;
IsTestAcc   =   Settings.IsTestAcc     ;
        if ~isfield(prepar, 'Y')
            Settings.IsTestAcc    =   0       ;
            IsTestAcc  =   Settings.IsTestAcc     ;
        end
IsSaveData  =   Settings.IsSaveData  ;
        SaveDataLocation    =   Settings.SaveDataLocation   ;

 
IterAll    =   Settings.IterAll   ;    
MiniBatchSZ     =   Settings.MiniBatchSZ     ; 
        BurninMB    =   Settings.BurninMB   ;   
        CollectionMB    =   Settings.CollectionMB      ;
AlgoChosen      =   Settings.AlgoChosen      ;
FurCollapse     =   Settings.FurCollapse    ;  
    epsi0FR     =   Settings.epsi0FR     ;
    tao0FR      =   Settings.tao0FR   ;   kappa0FR  =   Settings.kappa0FR     ;     
    epsipiet    =   (tao0FR + (1:IterAll)) .^ (-kappa0FR)    ;     
    epsipiet    =   epsi0FR * epsipiet / epsipiet(1)    ;
epsi0       =   Settings.epsi0     ;
tao0        =   Settings.tao0   ;   kappa0  =   Settings.kappa0     ;     
epsit       =   (tao0 + (1:IterAll)) .^ (-kappa0)    ;     
epsit       =   epsi0 * epsit / epsit(1)    ;
    etaStr  =   num2str(eta(1))     ;   etaStr(etaStr=='.')   =   []  ;
    MBSizeBurnCol   =   [num2str(MiniBatchSZ),'_',num2str(BurninMB),'_',num2str(CollectionMB)]  ;
    StepSet 	=   [num2str(epsi0),'_',num2str(tao0),'_',num2str(kappa0),'_',num2str(epsi0FR),'_',num2str(tao0FR),'_',num2str(kappa0FR)]  ;   
    StepSet(StepSet=='.')   =   []  ;
    
%% Initialization
if ~strcmp(dataname,'Wiki')
    XOnline     =   X_all(:,prepar.trindx);
    [V,NOnline]     =   size(XOnline);
else
    OriFromChangyouChen    =   Settings.OriFromChangyouChen   ;
    if OriFromChangyouChen
        fid_train   =   fopen('wiki_train.txt', 'r');
        V   =   7702    ;
        NOnline     =   1000000     ;
    else
        XOnline     =   X_all;
        [V,NOnline]     =   size(XOnline);
    end
end
%%============================ Global parameter initialization  ============================
Phi = cell(T,1);   Eta = cell(T,1);
for t = 1:T
    Eta{t}  =   eta(t)   ;
    if t == 1
        Phi{t}  =   0.05 + 0.95 * rand(V,K(1));  
    else
        Phi{t}  =   0.05 + 0.95 * rand(K(t-1),K(t));  
    end
    Phi{t}  =   bsxfun(@rdivide, Phi{t}, max(realmin,sum(Phi{t},1)));
end
r_k     =   1/K(T)*ones(K(T),1);    gamma0  =   1   ;    c0  =   1   ; 

PhiBar = cell(T,1);     PhiHat = cell(T,1);   
for t = 1:T
    PhiBar{t}	=   Phi{t}(1:end-1,:)   ;
    PhiHat{t}   =   Phi{t}          ;
end
%%============================ Local parameter initialization  ============================
Theta   = cell(T,1);        c_j     =   cell(T+1,1);
Xt_to_t1    = cell(T,1);    WSZS    =   cell(T,1);
%%============================ Register initialization  ============================
Ml          =   cell(T+2,1);    % Global
EWSZS       =   cell(T,1);      ESumLnPC    =   cell(T,1);  % Local
EXr     =   []   ;   ESumLnPT1       =   []   ; 
%%============================ Others ============================
PerpCun     =   []  ;   AccCun      =   []  ;
TimeAll     =   0   ;   
RiemannCoefCun  =   cell(1,T+2)   ;
PhiUsageCun     =   cell(1,T)   ;
for t = 1:T
    PhiUsageCun{t}   =   0   ;
end
%%============================ PerpZhou ============================
if IsTestPerpZhou & strcmp(DataType, 'Count')
    NTest   =   size(prepar.XHOtest,2)  ;
    FlagHOTrain     =   prepar.XHOtrain > 0     ;
    FlagHOTest      =   prepar.XHOtest > 0      ;
    Xt_to_t1Test    =   cell(T,1);  ThetaTest   =   cell(T,1);  c_jTest         =   cell(T+1,1);
    PhiThetaTest    =   0   ;   CountNum    =   0   ;
    for t=T:-1:1
        ThetaTest{t}    =   ones(K(t),NTest)/K(t);
    end
    for t=1:(T+1)
        c_jTest{t}=ones(1,NTest);
    end
    p_jTest = Calculate_pj(c_jTest,T);
end

%% Online Iterations
for epochi  =   1:IterAll
    if epochi == 1
        idxall     =   1 : NOnline  ;
    else
        idxall     =   randperm(NOnline)  ;
    end
    MBratio     =   NOnline / MiniBatchSZ     ;    
    N   =   MiniBatchSZ     ;
    
    for MBt     =   1:floor(MBratio)
        MBObserved      =   (epochi-1)*floor(MBratio)+MBt     ;     
        tic
        %% Get minibatch 
        if ~strcmp(dataname,'Wiki')
            X      =   XOnline(:,idxall( (MBt-1)*N + (1:N) ))       ;
        else
            if OriFromChangyouChen 
                X      =   GetBatch(fid_train, N, 'wiki_train.txt')     ;
            else
                X      =   XOnline(:,idxall( (MBt-1)*N + (1:N) ))       ;
            end
        end
        
       %% Initialize ParaLocal    
        if strcmp(DataType, 'Positive') | strcmp(DataType, 'Binary')
            [ii,jj]     =   find(X>eps);    %Treat values smaller than eps as 0
            iijj    =   find(X>eps);
        end
        if strcmp(DataType, 'Positive')
            a_j   =   ones(1,N);
        end
        for t=1:(T+1)
            c_j{t}=ones(1,N);
        end
        p_j = Calculate_pj(c_j,T);
        for t=T:-1:1
            Theta{t}    =   ones(K(t),N) / K(t);
            EWSZS{t}    =   0   ;
            ESumLnPC{t}    =   0   ;
        end
        EXr     =   0   ;   ESumLnPT1       =   0   ;   
        
        %% Sample and Collect local information
        for iter   =   1 : (BurninMB + CollectionMB)
            %%============= Upward Pass ===================
            for t   =   1:T
                if t    ==  1 
                    switch DataType
                        case 'Positive'
                            Rate = Phi{1}*Theta{1};
                            Rate = 2*sqrt(a_j(jj)'.*X(iijj).*Rate(iijj));
                            M  = Truncated_bessel_rnd( Rate );
                            a_j = randg(full(sparse(1,jj,M,1,N))+ac) ./ (bc+sum(X,1));
                            Xt = sparse(ii,jj,M,V,N);  
                        case 'Binary'
                            Rate = Phi{1}*Theta{1};
                            M = truncated_Poisson_rnd(Rate(iijj));
                            Xt = sparse(ii,jj,M,V,N);   
                        case 'Count'
                            Xt = sparse(X);     
                    end
                    [Xt_to_t1{t},WSZS{t}]   =   Multrnd_Matrix_mex_fast(Xt,Phi{t},Theta{t});
                else
                    [Xt_to_t1{t},WSZS{t}]   =   CRT_Multrnd_Matrix(sparse(Xt_to_t1{t-1}),Phi{t},Theta{t});
                end
            end
            Xr = CRT_sum_mex_matrix_v1(sparse(Xt_to_t1{T}'),r_k')';
            %%============= Downward Pass ======================
            if iter >= min(BurninMB/2,10)  
                if T > 1
                    p_j{2} = betarnd(  sum(Xt_to_t1{1},1)+a0pj   ,   sum(Theta{2},1)+b0pj  );
                else
                    p_j{2} = betarnd(  sum(Xt_to_t1{1},1)+a0pj   ,   sum(r_k,1)+b0pj  );
                end
                p_j{2} = min( max(p_j{2},realmin) , 1-realmin);
                c_j{2} = (1-p_j{2})./p_j{2};
                for t   =   3:(T+1)
                    if t    ==  T+1
                        c_j{t} = randg(sum(r_k)*ones(1,N)+e0cj) ./ (sum(Theta{t-1},1)+f0cj);
                    else
                        c_j{t} = randg(sum(Theta{t},1)+e0cj) ./ (sum(Theta{t-1},1)+f0cj);
                    end
                end
                p_j_temp = Calculate_pj(c_j,T);
                p_j(3:end)=p_j_temp(3:end);
            end
            for t  =   T:-1:1
                if t    ==  T
                    shape = r_k;
                else
                    shape = Phi{t+1}*Theta{t+1};
                end
                Theta{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1{t})), 1 ./ max(realmin, c_j{t+1}-logmax(1-p_j{t})) );
                if nnz(isnan(Theta{t})) | nnz(isinf(Theta{t})) | nnz(sum(Theta{t})==0)
                    warning(['Theta Nan',num2str(nnz(isnan(Theta{t}))),'_Inf',num2str(nnz(isinf(Theta{t}))),'_ORsum=0]']);
                    Theta{t}(isnan(Theta{t}))   =   0   ;
                end
            end
            %%============= Collection ======================
            if iter > BurninMB  
                for t  =   T:-1:1
                    if ~FurCollapse
                        EWSZS{t}    =   EWSZS{t} + WSZS{t}    ;
                    else
                        PhiTheta    =   Phi{t} * Theta{t}   ;
                        PhiTheta(PhiTheta<=eps)     =   eps     ;
                        if t == 1
                            EWSZS{t}    =   EWSZS{t} + (X./PhiTheta)*(Theta{t}.')    ;
                        else
                            EWSZS{t}    =   EWSZS{t} + (psi(Xt_to_t1{t-1}+PhiTheta)-psi(PhiTheta))*(Theta{t}.')    ;
                        end
                    end
                    if nnz(isnan(EWSZS{t})) | nnz(isinf(EWSZS{t}))
                        warning(['EWSZS Nan',num2str(nnz(isnan(EWSZS{t}))),'_Inf',num2str(nnz(isinf(EWSZS{t})))]);
                        EWSZS{t}(isnan(EWSZS{t}))   =   0   ;
                    end
                end
                EXr     =   EXr + Xr    ;
                ESumLnPT1     =   ESumLnPT1 + sum(logmax(1-p_j{T+1}))       ;
            end
        end
        
        %% Update global parameters Phi
        %%=================  Update Mk  ===============     
        for t  =   1:T
            PhiUsageCun{t}   =   PhiUsageCun{t} + sum(EWSZS{t},1)  ;
            if ~FurCollapse
                EWSZS{t}    =   MBratio * EWSZS{t}/CollectionMB      ;
            else
                EWSZS{t}    =   MBratio * (Phi{t} .* EWSZS{t}/CollectionMB)    ;
            end
            switch AlgoChosen
                case 1 % SGRLD
                    Ml{t}   =   1   ;
                    RiemannCoefCun{t}(:,MBObserved)   =   Ml{t}(:)  ;
                case 3 % TLFSGR 
                    if t == 1
                        if (MBObserved == 1)
                            Ml{1}     =   mean( sum(EWSZS{1},1) )  ;
                        else
                            Ml{1}     =   (1 - epsipiet(MBObserved)) * Ml{1} ...
                                            + epsipiet(MBObserved) * mean( sum(EWSZS{1},1) )   ;
                        end
                    else
                        Ml{t}   =   mean(Ml{1})   ;
                    end
                    RiemannCoefCun{t}(:,MBObserved)   =   Ml{t}(:)  ;
                case 4 % TLASGR
                    if (MBObserved == 1)
                        Ml{t}     =   sum(EWSZS{t},1)  ;
                    else
                        Ml{t}     =   (1 - epsipiet(MBObserved)) * Ml{t} ...
                                        + epsipiet(MBObserved) * sum(EWSZS{t},1)   ;
                    end
                    RiemannCoefCun{t}(:,MBObserved)   =   Ml{t}(:)  ;                    
                otherwise
                    error('Wrong AlgoChosen!!!')  ;  
            end
            %%=================  Update Global parameters Phi  ===============  
            for iii     =   1:1
                switch AlgoChosen
                    case 1 % SGRLD
                        tmp     =   (EWSZS{t}+Eta{t}) - bsxfun(@times, sum(EWSZS{t}+PhiHat{t},1), Phi{t})  ;
                        tmp1    =   2 * PhiHat{t}  ;
                        PhiHat{t}   =   abs( PhiHat{t} + epsit(MBObserved)*tmp ...
                                        + sqrt(epsit(MBObserved)*tmp1) .* randn(size(PhiHat{t})) )  ;
                        Phi{t}  =   bsxfun(@rdivide, PhiHat{t}, sum(PhiHat{t},1) )  ; %  figure(25),DispDictionaryImagesc(Phi{1});drawnow;
                    case 3 % TLFSGR
                        tmp     =   (EWSZS{t}+Eta{t})   ;
                        tmp     =   bsxfun(@times, 1./Ml{t}, tmp - bsxfun(@times, sum(tmp,1), Phi{t}))  ;
                        tmp1    =   bsxfun(@times, 2./Ml{t}, Phi{t})  ;
                        tmp     =   Phi{t} + epsit(MBObserved)*tmp + sqrt(epsit(MBObserved)*tmp1) .* randn(size(Phi{t}))   ;
                        Phi{t}  =   ProjSimplexSpecial(tmp , Phi{t}, eps)  ; % figure(25),DispDictionaryImagesc(Phi{1});drawnow;
                    case 4 % TLASGR
                        tmp     =   (EWSZS{t}+Eta{t})   ;
                        tmp     =   bsxfun(@times, 1./Ml{t}, tmp - bsxfun(@times, sum(tmp,1), Phi{t}))  ;
                        tmp1    =   bsxfun(@times, 2./Ml{t}, Phi{t})  ;
                        tmp     =   Phi{t} + epsit(MBObserved)*tmp + sqrt(epsit(MBObserved)*tmp1) .* randn(size(Phi{t}))   ;
                        Phi{t}  =   ProjSimplexSpecial(tmp , Phi{t},eps)  ; % figure(25),DispDictionaryImagesc(Phi{1});drawnow;
                    otherwise
                        error('Wrong AlgoChosen!!!')  ;  
                end
            end
            if nnz(imag(Phi{t})~=0) | nnz(isnan(Phi{t})) | nnz(isinf(Phi{t}))
                warning(['Phi Nan',num2str(nnz(isnan(Phi{t}))),'_Inf',num2str(nnz(isinf(Phi{t})))]);
                Phi{t}(isnan(Phi{t}))   =   0   ;
            end
        end
        
        %% Update Global parameters r gamma0 c0 
        EXr     =   MBratio * EXr/CollectionMB     ;
        ESumLnPT1     =   MBratio * ESumLnPT1/CollectionMB     ;
        if (MBObserved == 1)
            Ml{T+1}     =   -ESumLnPT1    ;
        else
            Ml{T+1}     =   (1 - epsipiet(MBObserved)) * Ml{T+1} ...
                            + epsipiet(MBObserved) * (-ESumLnPT1)   ;
        end
        RiemannCoefCun{T+1}(:,MBObserved)   =   Ml{T+1}(:)  ;   

        tmp     =   1/Ml{T+1} * ( (EXr+gamma0/K(T)) - r_k*(c0 - ESumLnPT1) ) ;
        tmp1    =   2/Ml{T+1} * r_k     ;
        r_k     =   abs( r_k + epsit(MBObserved)*tmp + sqrt(epsit(MBObserved)*tmp1) .* randn(size(r_k)) )   ;
      
       %% ==================== Figure ====================
        TimeOneIter     =   toc     ;   
        TimeAll         =   TimeAll + TimeOneIter   ;
        if (mod(MBObserved,5) == 0) | (MBObserved < 5)
            fprintf('Layer: %d, Sweepi: %d, MBObserved: %d, Time: %d, r_k: %d, gamma0: %d, c0: %d\n',T,epochi,MBObserved,TimeOneIter,mean(r_k),gamma0,c0)  ;
        end   
        if IsDisplay  &&  (mod(MBObserved,FigureGap)==0) 
            tmp     =   1   ;
            for t   =   1:T
                tmp     =   tmp * Phi{t}    ;
                if strcmp(DataType,'Binary')
                    figure(90+t),DispDictionaryImshow(1-exp(-tmp));drawnow;                    
                else
                    figure(90+t),DispDictionaryImshow(tmp);drawnow;
                end
            end
        end
        
        %% ==================== Quantify Heldout Perplexity Zhou ====================
        if IsTestPerpZhou & strcmp(DataType, 'Count') & isfield(prepar,'XHOtrain') & isfield(prepar,'XHOtest') %& (MBObserved <= (PerpZhouBurnin + 3000))
            for t   =   1:T
                if t    ==  1 
                    Xt_to_t1Test{t}     =   Multrnd_Matrix_mex_fast_v1(prepar.XHOtrain,Phi{t},ThetaTest{t});
                else
                    Xt_to_t1Test{t}     =   CRT_Multrnd_Matrix(sparse(Xt_to_t1Test{t-1}),Phi{t},ThetaTest{t});
                end
            end
            if T > 1
                p_jTest{2} = betarnd(  sum(Xt_to_t1Test{1},1)+a0pj   ,   sum(ThetaTest{2},1)+b0pj  );
            else
                p_jTest{2} = betarnd(  sum(Xt_to_t1Test{1},1)+a0pj   ,   sum(r_k,1)+b0pj  );
            end
            p_jTest{2} = min( max(p_jTest{2},realmin) , 1-realmin);     c_jTest{2} = (1-p_jTest{2})./p_jTest{2};
            for t   =   3:(T+1)
                if t    ==  T+1
                    c_jTest{t} = randg(sum(r_k)*ones(1,NTest)+e0cj) ./ (sum(ThetaTest{t-1},1)+f0cj);
                else
                    c_jTest{t} = randg(sum(ThetaTest{t},1)+e0cj) ./ (sum(ThetaTest{t-1},1)+f0cj);
                end
            end
            p_j_temp = Calculate_pj(c_jTest,T);     p_jTest(3:end)=p_j_temp(3:end);
            for t  =   T:-1:1
                if t    ==  T
                    shape = r_k;
                else
                    shape = Phi{t+1}*ThetaTest{t+1};
                end
                ThetaTest{t} = bsxfun(@times, randg(bsxfun(@plus,shape,Xt_to_t1Test{t})), 1 ./ max(realmin, c_jTest{t+1}-log(max(1-p_jTest{t},realmin)) ) );     
                if nnz(isnan(ThetaTest{t})) | nnz(isinf(ThetaTest{t})) | nnz(sum(ThetaTest{t})==0)
                    warning(['PerpZhou: Theta Nan',num2str(nnz(isnan(ThetaTest{t}))),'_Inf',num2str(nnz(isinf(ThetaTest{t}))),'_ORsum=0]']);
                    ThetaTest{t}(isnan(ThetaTest{t}))   =   0   ;
                end
            end

            tmp     =   Phi{1}*ThetaTest{1};
            if nnz(isnan(tmp)) | nnz(isinf(tmp))
                tmp     =   Phi{1}*ThetaTest{1};
            end    
            
                   
            temp    =   bsxfun(@rdivide, tmp, max(realmin,sum(tmp,1)) );
            PerpCun.TmpPerpZhouTest(MBObserved)    =   ...
                exp(- sum(prepar.XHOtest(FlagHOTest).*log(max(realmin,temp(FlagHOTest)))) / sum(prepar.XHOtest(:))  );                      

            if (MBObserved > PerpZhouBurnin && mod(MBObserved,PerpZhouStep) == 0)
                PhiThetaTest    =   PhiThetaTest + tmp;
                CountNum    =   CountNum + 1    ;

                temp    =   PhiThetaTest / CountNum     ;
                temp    =   bsxfun(@rdivide, temp, max(realmin,sum(temp,1)) );

                PerpCun.PerpZhouTrain(CountNum)     =   exp(- sum(prepar.XHOtrain(FlagHOTrain).*log(max(realmin,temp(FlagHOTrain)))) / sum(prepar.XHOtrain(:)) );
                PerpCun.PerpZhouTest(CountNum)      =   exp(- sum(prepar.XHOtest(FlagHOTest).*log(max(realmin,temp(FlagHOTest)))) / sum(prepar.XHOtest(:))  );
                PerpCun.PerpZhouMBSeen(CountNum)    =   MBObserved  ;
                PerpCun.PerpZhouDataSeen(CountNum)  =   MBObserved * MiniBatchSZ  ;
                PerpCun.PerpZhouTimeUsed(CountNum)  =   TimeAll     ;

                if (mod(MBObserved,5) == 0)
                    fprintf('PerpZhouVal: MBObserved: %d, HeldoutTrain: %d, HeldoutTest: %d, DataSeen: %d, TimeUsed: %d\n',...
                        MBObserved,full(PerpCun.PerpZhouTrain(end)),full(PerpCun.PerpZhouTest(end)),full(PerpCun.PerpZhouDataSeen(end)),full(PerpCun.PerpZhouTimeUsed(end)))  ;
                end
            end
        end
        
        %% ====================  Quantification Point Perp  ====================
        if IsTestPointPerp
            if strcmp(DataType, 'Positive')
                ParaGlobal.a_jmean  =   median(a_j)     ;
            end
            ParaGlobal.Phi    =   Phi   ;   ParaGlobal.Eta    =   Eta     ;
            ParaGlobal.r_k    =   r_k   ;   ParaGlobal.gamma0    =   gamma0   ;     ParaGlobal.c0    =   c0   ;
            ParaGlobal.c_jmean = zeros(1,T+1);
            for t   =   1:(T+1)
                ParaGlobal.c_jmean(t)  =    median(c_j{t})   ;
            end
            
            if IsTestPointPerp & strcmp(DataType, 'Count') & isfield(prepar,'XHOtrain') & isfield(prepar,'XHOtest')
                tmp     =   find(PointTestLocation == MBObserved)   ;
                if tmp
                    PerpCun.PointPerpMBSeen(tmp)    =   MBObserved  ;
                    PerpCun.PointPerpDataSeen(tmp)  =   MBObserved * MiniBatchSZ  ;
                    PerpCun.PointPerpTimeUsed(tmp)  =   TimeAll     ;
                    
                    [PerpCun.PointPerpVal(tmp),PerpCun.PointPerpValTrain(tmp)]   =   ...
                        QuantGBN_PerpWord(prepar.XHOtrain,prepar.XHOtest,ParaGlobal,K,T,DataType,SuPara,Settings)   ;
                end
            end
        end
       
       %% others        
        if (MBObserved >= IterAll)
            break ;
        end 
        
    end
    if (MBObserved >= IterAll)
        break ;
    end 
end

if strcmp(dataname,'Wiki') & OriFromChangyouChen
    fclose(fid_train)   ;
end






