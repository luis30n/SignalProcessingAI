%% INPUT PREPROCESSING
clear all;
tic;
load('random_forest_v2');
x=xsw;
y=ymed;
Nseg = 1;
n_trees=1500;
regresores = string([1:1:81]);
regresores= "R" + regresores;
models=[];
n_regs=40;
n_iters = [5000];
Mdl =[];
val=false;
i=1;
parfor iseg = 1:5000
    dBm = @(x) 10*log10(rms(x).^2/100)+30;
    dBminst = @(x) 10*log10(abs(x).^2/100)+30;
    PAPR = @(x) 20*log10(max(abs(x))/rms(x)); 
    maxdBm = @(x) dBm(x) + PAPR(x);
    scale_dBm = @(x,P) x*10^((P-dBm(x))/20);
    model_PA.tipo = 'GMP';
    model_PA.extension_periodica = 0;
    model_PA.grafica = 0;
    model_PA.h = [];
    model_PA.pe = 0;
    model_PA.type = 'GMP';
    model_PA.extension_periodica = 0;
    model_PA.grafica = 0;
    model_PA.h = [];

    model_PA.Ka = [0:12];
    model_PA.La = 15*ones(size(model_PA.Ka));
    model_PA.Kb = [2 4 6];
    model_PA.Lb = [1 1 1];
    model_PA.Mb = [1 1 1];
    model_PA.Kc = [2 4 6];
    model_PA.Lc = [1 1 1];
    model_PA.Mc = [1 1 1];
    model_PA.dc = 0;
    model_PA.cs = 0;
    model_PA.nmax = 200;

    warning ('off','stats:robustfit:RankDeficient');
    warning ('off','MATLAB:nearlySingularMatrix');    
    orden = [1:81];
    orden = orden(randperm(length(orden)));
    orden = orden(1:n_regs);
    model_PA.s = orden;
    model_PA.calculo = 'shuffle';
    ind = [1:2*length(x)/10];
    %ind = [(iseg-1)*length(x)/Nseg+1:iseg*length(x)/Nseg];
    %disp(fprintf('%d: desde %d a %d',iseg,(iseg-1)*length(x)/Nseg+1,iseg*length(x)/Nseg));
    disp(fprintf('Iteracion %d',iseg));
    x_id = x(ind);
    y_id = y(ind);
    x_va = x(end/2:end);
    y_va = y(end/2:end);

    model_PA = model_gmp_domp_omp(y_id, x_id, model_PA,val)
    
    models=[models;model_PA];
    model_PA=[];
    %si la h no está vacía, el modelo se valida, no identifica uno nuevo.
    %[model_PA] = model_gmp_domp_omp(y_va, x_va, model_PA)
end
   
    n_comb=n_iters;
    A=zeros(n_comb,81); %si no se usa pondremos un 0, si se usa, un uno
    NMSE=[];
    for i = 1:n_comb
        A(i, models(i).s)=1; %el resto cero porque no se usan
        NMSE(i,1)= models(i).nmse;
    end
T = array2table(A,'VariableNames',cellstr(regresores));
NMSE = array2table(abs(NMSE));
T= [T, NMSE];
%% 
X = NMSE;
Y= A;
[n,m]=size(X);
[Xtrain, idtrain]  = datasample(X, round(0.67*n));
Ytrain= Y(idtrain);
idtest       = 1:n;
idtest(idtrain) = [];
X_test         = X(idtest);
Y_test = Y(idtest);

imp = manual_importances(T(1:81),T)
[B,I] = maxk(imp, length(imp));
regresores_order= "Regresor " + models(1).Rmat(I);
O= table(regresores_order, B', 'VariableNames',{'RegresoresOrder', 'Importance'});
orden = I;
configuration;
model_PA.s = orden(1:n_regs);
model_PA.calculo = 'shuffle';
val = true;
model_PA = model_gmp_domp_omp(y_id, x_id, model_PA, true);
toc