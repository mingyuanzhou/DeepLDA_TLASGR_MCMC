clear,clc,close all;

%% Data Input
ToBeAnalized    =   1   ;

switch ToBeAnalized
    case 1          %%========= MNIST =================
        TrainSize      =   6e4   ;       TestSize        =   1e4   ;
        load mnist_gray.mat     ;   % Load Data   
        %======== Prepare Train Data  ===========
        ndx = []; mtrain = TrainSize / 10;
        if mtrain < 6000
            for ii = 0:1:9
                tmp = find(train_label==ii);
                ndx = [ndx; tmp(1:mtrain)];
            end
        else
            ndx = [1:60000];
        end
        X       =   train_mnist(:,ndx);
        Xlabel  =   train_label(ndx);
        rng(0,'twister');
        DataType    =   'Positive';
        dataname    =   'MNIST';
        %======== Prepare Test Data   ===========
        ndx = []; mtest = TestSize/10  ;
        if mtest < 1000
            for ii = 0:1:9
                tmp = find(test_label==ii);
                ndx = [ndx; tmp(1:mtest)];
            end
        else
            ndx=[1:10000];
        end
        Xtest       =   test_mnist(:,ndx);
        Xtestlabel  =   test_label(ndx);
        %======== Combine Train and Test ===========
        clear train_mnist train_label test_mnist test_label ;
        X_all   =   [X,Xtest]   ;
        prepar.trindx   =   1:length(Xlabel)    ;
        prepar.teindx   =   length(Xlabel) + (1:length(Xtestlabel))     ;
        prepar.Y        =   [Xlabel(:);Xtestlabel(:)]   ;
        prepar.Y(prepar.Y == 0)     =   10  ; 
        clear X Xlabel  ndx Xtest Xtestlabel;
    case 2         %% ======== 20news from GanZhe  https://github.com/zhegan27  ============
        DataType    =   'Count';
        dataname    =   '20newsGanZhe';
        load('20news_data.mat')     ;        
        X_all   =   [wordsTrain,wordsTest+wordsHeldout]  ;
        prepar.trindx   =   1:size(wordsTrain,2)    ;
        prepar.teindx   =   prepar.trindx(end) + (1:size(wordsTest,2))    ;
        prepar.XHOtrain     =   wordsHeldout    ;
        prepar.XHOtest      =   wordsTest       ;
        prepar.Y    =   [labelsTrain;labelsTest]    ;
        prepar.WO   =   vocabulary  ;
        prepar.labelsToGroup    =   labelsToGroup   ;
        clearvars -EXCEPT X_all prepar DataType dataname ToBeAnalized SystemInRuningLinux SuPara Settings;
    case 3        %%========= RCV1-v2 Count from GanZhe  https://github.com/zhegan27 ======================
        Top2000     =   0   ;   % TrainSize   =   90000   ;
        HeldoutTrainPercent     =   0.8     ; 
        DataType    =   'Count';
        dataname    =   'RCV1_v2';
        load('rcv1_v2_data.mat')     ;   % Load Data  
        if exist('TrainSize','var')
            wordsTrain  =   wordsTrain(:,1:TrainSize)   ;
            labelsTrain     =   labelsTrain(1:TrainSize,:) ;
        end
        prepar.trindx   =   1:size(wordsTrain,2)     ;
        prepar.teindx   =   prepar.trindx(end) + (1:size(wordsTest,2))  ;
        X_all   =   [wordsTrain , wordsTest]     ;
        WO      =   vocabulary       ;
        if 1 
            tmp     =   (sum(X_all,2) >= 5)    ;
            X_all   =   X_all(tmp,:)    ;
            WO      =   WO(tmp,:)    ;
            
            tmp     =   (sum(X_all>0,1) >= 5)    ;
            X_all   =   X_all(:,tmp)    ;
        end
        if Top2000
            tmp     =   sum(X_all,2)   ;
            [~,indx]    =   sort(tmp,'descend')  ;
            X_all   =   X_all(indx(1:2000) , :)  ;  
        end    
        prepar.XHOtrain     =   wordsHeldout    ;   prepar.XHOtest      =   wordsTest    ;
        clearvars -EXCEPT X_all prepar DataType dataname ToBeAnalized SystemInRuningLinux SuPara Settings ;
    case 4        %%========= Wiki Count  ======================
        OriFromChangyouChen     =   0   ;
        Settings.OriFromChangyouChen    =   OriFromChangyouChen     ;
        PreProcess      =   1   ;
        DataType    =   'Count';
        dataname    =   'Wiki';
        load('wiki_test.mat')     ;   % Load Data  
        X_all   =   []  ;
        prepar.WO      =   vocabulary       ;
        prepar.XHOtrain     =   wordsHeldout    ;   prepar.XHOtest      =   wordsTest    ;
        if ~OriFromChangyouChen 
            load('WikiWordDocMatPureTrain.mat')  ;
            if PreProcess
                indxtmp     =   (sum( WikiMat>0 , 1 ) >= 5)     ;     
                X_all       =   WikiMat( : , indxtmp )  ;
            else
                X_all   =   WikiMat     ;
            end
            prepar.trindx   =   1 : size(X_all,2)     ;
            prepar.teindx   =   []  ;
        end        
        clearvars -EXCEPT X_all prepar DataType dataname ToBeAnalized SystemInRuningLinux SuPara Settings;  
    otherwise
        error('Wrong "ToBeAnalized"')
end

%% GBN Settings 
% K = [128]  ;   T   =   length(K)   ;   
K = [128,64]  ;   T   =   length(K)   ;   
% K = [128,64,32]  ;   T   =   length(K)   ;   
        SuPara.ac  =   1   ;   SuPara.bc      =   1   ;
        SuPara.a0pj    =   0.01   ;   SuPara.b0pj    =   0.01   ;
        SuPara.e0cj    =   1   ;   SuPara.f0cj    =   1   ;
        SuPara.e0c0    =   1    ;   SuPara.f0c0    =   1    ;
        SuPara.a0gamma  =   1   ;   SuPara.b0gamma      =   1   ;
SuPara.eta  =   0.05 * ones(1,T)     ;

%% Output Settings
Settings.IsDisplay          =   1   ;     
        Settings.FigureGap     =   5  ;
Settings.IsTestPerpZhou     =   0   ;
        Settings.PerpZhouBurnin     =   2000    ;  
        Settings.PerpZhouStep   =   1   ;
Settings.IsTestPointPerp    =   0   ;   
        Settings.PointTestLocation  =   [50,100:100:20000]  ;
Settings.IsTestAcc      =   0  ;  
        Settings.TestBurnin         =   200     ;      
        Settings.TestCollection     =   200     ;       
        Settings.TestSampleSpace    =   2   ;
        Settings.ProcessMethod      =   [2]   ;  
Settings.IsSaveData     =   0  ;
        Settings.SaveDataLocation  =   [500:500:20000]  ;  

%% Mini-Batch Methods Settings 
Settings.MiniBatchSZ    =   200     ;    
        Settings.BurninMB   =   20*T	;   
        Settings.CollectionMB   =   10*T      ;
switch dataname
    case '20newsGanZhe'
        Settings.IterAll    =   3500   ;	
        Settings.FurCollapse 	=   0  	;
        Settings.IsDisplay   	=   0   ;  
    case 'RCV1_v2'
        Settings.IterAll    =   8000    ;   	
        Settings.PointTestLocation  =   [50,100:100:4000,4500:500:20000]  ;  
        Settings.FurCollapse 	=   1   ;
        Settings.IsDisplay    	=   0   ;  
    case 'Wiki'
        Settings.IterAll    =   5000    ;      
        Settings.PerpZhouBurnin     =   3500    ;
        Settings.PointTestLocation  =   [50,100:100:5000,5200:200:20000]  ;  
        Settings.FurCollapse 	=   1  	;
        Settings.IsDisplay    	=   0   ;
    otherwise
        Settings.IterAll    =   3000    ;
        Settings.FurCollapse 	=   0  	;
end

%% TLASGR MCMC
if 1
    Settings.AlgoChosen     =   4  ;% TLASGR
    Settings.epsi0FR    =   0.5     ;
    Settings.tao0FR     =   20      ;   Settings.kappa0FR   =   0.7     ;   
    Settings.epsi0      =   1     ;
    Settings.tao0       =   20    ;     Settings.kappa0     =   0.7     ;   
    for trial   =    1 : 5
        rng(trial,'twister');
        GBN_Online_CommonMode_CongPerp(X_all,prepar,K,T,trial,DataType,dataname,SuPara,Settings)        ;
    end
end

%% TLFSGR
if 0
    Settings.AlgoChosen     =   3  ;
    Settings.epsi0FR    =   0.5     ;
    Settings.tao0FR     =   20      ;   Settings.kappa0FR   =   0.7     ;   
    Settings.epsi0      =   1     ;
    Settings.tao0       =   20    ;     Settings.kappa0     =   0.7     ;   
    for trial   =    1 : 5
        rng(trial,'twister');
        GBN_Online_CommonMode_CongPerp(X_all,prepar,K,T,trial,DataType,dataname,SuPara,Settings)        ;
    end
end


%% SGRLD
if 0
    Settings.AlgoChosen     =   1  ;% SGRLD
    Settings.epsi0FR    =   0.5     ;
    Settings.tao0FR     =   20      ;   Settings.kappa0FR   =   0.7     ;     
    switch dataname
        case '20newsGanZhe'
            Settings.epsi0    =   0.1     ;
            Settings.tao0     =   20      ;   Settings.kappa0     =   0.9     ;   
        case 'RCV1_v2'
            Settings.epsi0    =   0.001   ; 
            Settings.tao0     =   1000    ;   Settings.kappa0     =   0.6     ;
        case 'Wiki'
            Settings.epsi0    =   0.001   ;
            Settings.tao0     =   1000    ;   Settings.kappa0     =   0.6     ;
        otherwise
            Settings.epsi0    =   0.1     ;
            Settings.tao0     =   20      ;   Settings.kappa0     =   0.9     ;   
    end   
    for trial   =    1 : 5
        rng(trial,'twister');
        GBN_Online_CommonMode_CongPerp(X_all,prepar,K,T,trial,DataType,dataname,SuPara,Settings)        ;
    end
end






