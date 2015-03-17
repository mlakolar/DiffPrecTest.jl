{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf190
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \
                            %size%\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   \
n1=100;\
n2=100;\
rho1=0.5;\
p=200;\
rep=1000;\
\
for i = 1:p\
  I (i,i)=1;\
end\
for i = 1:p\
    D(i,i) = unifrnd(0.5,2.5);\
end\
\
% create Sigma1\
for i=1:p\
    Omega1(i,i)=1;\
end\
for i=1:p-1\
    Omega1(i,i+1)=0.6;\
    Omega1(i+1,i)=0.6;\
end\
for i=1:p-2\
    Omega1(i,i+2)=0.3;\
    Omega1(i+2,i)=0.3;\
end\
Omega1 = D^(1/2)*Omega1*D^(1/2);\
Sigma1=Omega1^(-1);\
\
%create Omega2\
for i=1:p\
    Omega2(i,i)=1;\
end\
for k=1:p/10\
    for j=10*(k-1)+2:10*(k-1)+10\
        Omega2(10*(k-1)+1,j)=0.5;\
        Omega2(j,10*(k-1)+1)=0.5;\
    end\
end\
de = abs(min(eig(Omega2)))+0.05;\
Omega2 = (Omega2 + de*I)/(1+de);\
Omega2 = D^(1/2)*Omega2*D^(1/2);\
Sigma2 = Omega2^(-1);\
\
%model 3\
for i = 1:p\
  Sigma4 (i,i)=1;\
end\
for i = 1:(p-1)\
  for j = (i+1):p\
    Sigma4 (i,j) = binornd (1,0.05)*0.8;\
    Sigma4 (j,i) = Sigma4 (i,j);\
  end\
end\
de = abs (min (eig (Sigma4)))+0.05;\
Sigma4 = (Sigma4 + de*I)/(1+de);\
Omega4=Sigma4;\
Omega4 = D^(1/2)*Omega4*D^(1/2);\
\
\
Sigma4=Omega4^(-1);\
\
\
%model 4 in paper\
KK=2;\
\
for i = 1:p\
  Sigma5 (i,i) = 1;\
end\
for k = 1:(p/KK)\
  for i = ((k-1)*KK+1):(k*KK-1)\
    for j = (i+1):(k*KK)\
      Sigma5 (i,j)=0.8;\
      Sigma5 (j,i)=Sigma5 (i,j);\
    end\
  end\
end\
de = abs (min (eig (Sigma5)))+0.05;\
Sigma5 = (Sigma5 + de*I)/(1+de);\
Omega5=Sigma5^(-1);\
Omega5 = D^(1/2)*Omega5*D^(1/2);\
Sigma5=Omega5^(-1);\
\
\
\
for i=1:n1\
    e1(i)=1;\
end\
for i=1:n2\
    e2(i)=1;\
end\
SSS=Sigma1^(1/2);\
for l=1:rep\
    x = randn (n1,p)*SSS;\
    y = randn (n2,p)*SSS;\
    \
    Shatmimix=cov(x);\
    sigmax=diag(Shatmimix);\
\
    Shatmimiy=cov(y);\
    sigmay=diag(Shatmimiy);\
    for i = 1:p\
        newx=x';\
        newy=y';\
        newx(i,:)=[];\
        newy(i,:)=[];\
        bx(:,i)=lasso(newx',x(:,i),'Lambda',2*sqrt(sigmax(i)*log(p)/(n1)));\
        by(:,i)=lasso(newy',y(:,i),'Lambda',2*sqrt(sigmay(i)*log(p)/(n2)));\
     \
        %calculate \\xi\
        c1(:,i) = x(:,i)-mean(x(:,i))'*e1'- (newx-mean(newx')'*e1)'*bx(:,i);\
        c2(:,i) = y(:,i)-mean(y(:,i))'*e2'-(newy-mean(newy')'*e2)'*by(:,i);\
    end\
    R1=c1'*c1/n1;\
    R2=c2'*c2/n2;\
    s1=mean(c1.^2);\
    s2=mean(c2.^2);\
    for i=1:p-1\
        for j=i+1:p\
            T1(i,j)=R1(i,j)+s1(1,i)*bx(i,j)+s1(1,j)*bx(j-1,i);\
            T2(i,j)=R2(i,j)+s2(1,i)*by(i,j)+s2(1,j)*by(j-1,i);\
            diff(i,j)=(T1(i,j)/(R1(i,i)*R1(j,j))-T2(i,j)/(R2(i,i)*R2(j,j)))^2/(1/(R1(i,i)*R1(j,j))/n1*(1+bx(i,j)^2*R1(i,i)/R1(j,j))+1/(R2(i,i)*R2(j,j))/n2*(1+by(i,j)^2*R2(i,i)/R2(j,j)));\
        end\
    end\
    M(l)=max(max(diff));\
\
end\
\
cri = -2*log (-(8*3.14159)^(1/2)*log (0.95));\
cri = 4*log (p)-log (log (p)) + cri;\
\
sum (M>cri)/rep\
\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \
                            %power%\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   \
\
Sigma1=Sigma1;\
Omega1=Omega1;\
r = randperm (p-1);\
rd = r (1:8);\
rd = sort (rd);\
VT = Omega1;\
Sigma=Omega1;\
VT (rd (1),rd (5))=Sigma (rd (1),rd (5))+(randi(2)-1.5)*2*unifrnd (1/2,1)*max(diag(Sigma))*sqrt(2*log(p)/n1);\
VT (rd (2),rd (6))=Sigma (rd (2),rd (6))+(randi(2)-1.5)*2*unifrnd (1/2,1)*max(diag(Sigma))*sqrt(2*log(p)/n1);\
VT (rd (3),rd (7))=Sigma (rd (3),rd (7))+(randi(2)-1.5)*2*unifrnd (1/2,1)*max(diag(Sigma))*sqrt(2*log(p)/n1);\
VT (rd (4),rd (8))=Sigma (rd (4),rd (8))+(randi(2)-1.5)*2*unifrnd (1/2,1)*max(diag(Sigma))*sqrt(2*log(p)/n1);\
VT (rd (5),rd (1))=VT (rd (1),rd (5));\
VT (rd (6),rd (2))=VT (rd (2),rd (6));\
VT (rd (7),rd (3))=VT (rd (3),rd (7));\
VT (rd (8),rd (4))=VT (rd (4),rd (8));\
de = abs (min (min (eig (VT)),min (eig (Sigma))))+0.05;\
Omega1 = (Omega1 + de*I);\
VT = (VT+de*I);\
Sigma1=Omega1^(-1);\
VTT=VT^(-1);\
SSS=Sigma1^(1/2);\
VVV=VTT^(1/2);\
\
for l=1:rep\
    x = randn (n1,p)*SSS;\
    y = randn (n2,p)*VVV;\
    \
    Shatmimix=cov(x);\
    sigmax=diag(Shatmimix);\
\
    Shatmimiy=cov(y);\
    sigmay=diag(Shatmimiy);\
    for i = 1:p\
        newx=x';\
        newy=y';\
        newx(i,:)=[];\
        newy(i,:)=[];\
        bx(:,i)=lasso(newx',x(:,i),'Lambda',2*sqrt(sigmax(i)*log(p)/(n1)));\
        by(:,i)=lasso(newy',y(:,i),'Lambda',2*sqrt(sigmay(i)*log(p)/(n2)));\
     \
        %calculate \\xi\
        c1(:,i) = x(:,i)-mean(x(:,i))'*e1'- (newx-mean(newx')'*e1)'*bx(:,i);\
        c2(:,i) = y(:,i)-mean(y(:,i))'*e2'-(newy-mean(newy')'*e2)'*by(:,i);\
    end\
    R1=c1'*c1/n1;\
    R2=c2'*c2/n2;\
    s1=mean(c1.^2);\
    s2=mean(c2.^2);\
    for i=1:p-1\
        for j=i+1:p\
            T1(i,j)=R1(i,j)+s1(1,i)*bx(i,j)+s1(1,j)*bx(j-1,i);\
            T2(i,j)=R2(i,j)+s2(1,i)*by(i,j)+s2(1,j)*by(j-1,i);\
            diff(i,j)=(T1(i,j)/(R1(i,i)*R1(j,j))-T2(i,j)/(R2(i,i)*R2(j,j)))^2/(1/(R1(i,i)*R1(j,j))/n1*(1+bx(i,j)^2*R1(i,i)/R1(j,j))+1/(R2(i,i)*R2(j,j))/n2*(1+by(i,j)^2*R2(i,i)/R2(j,j)));\
        end\
    end\
    M(l)=max(max(diff));\
\
end\
\
cri = -2*log (-(8*3.14159)^(1/2)*log (0.95));\
cri = 4*log (p)-log (log (p)) + cri;\
\
sum (M>cri)/rep\
\
\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \
                            %FDR%\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   \
\
\
SSS=I;\
VVV=Sigma1^(1/2);\
VV2=Sigma2^(1/2);\
OOO=Omega1-Omega2;\
for l=1:rep\
    x = randn (n1,p)*VVV;\
    y = randn (n2,p)*VV2;\
    \
    Shatmimix=cov(x);\
    sigmax=diag(Shatmimix);\
\
    Shatmimiy=cov(y);\
    sigmay=diag(Shatmimiy);\
    for ii=1:40\
        for i = 1:p\
            newx=x';\
            newy=y';\
            newx(i,:)=[];\
            newy(i,:)=[];\
            bx(:,i)=lasso(newx',x(:,i),'Lambda',2*ii/40*sqrt(sigmax(i)*log(p)/(n1)));\
            by(:,i)=lasso(newy',y(:,i),'Lambda',2*ii/40*sqrt(sigmay(i)*log(p)/(n2)));\
     \
            %calculate \\xi\
            c1(:,i) = x(:,i)-mean(x(:,i))'*e1'- (newx-mean(newx')'*e1)'*bx(:,i);\
            c2(:,i) = y(:,i)-mean(y(:,i))'*e2'-(newy-mean(newy')'*e2)'*by(:,i);\
        end\
        R1=c1'*c1/n1;\
        R2=c2'*c2/n2;\
        s1=mean(c1.^2);\
        s2=mean(c2.^2);\
        for i=1:p-1\
            for j=i+1:p\
                T1(i,j)=R1(i,j)+s1(1,i)*bx(i,j)+s1(1,j)*bx(j-1,i);\
                T2(i,j)=R2(i,j)+s2(1,i)*by(i,j)+s2(1,j)*by(j-1,i);\
                diff(i,j)=(T1(i,j)/(R1(i,i)*R1(j,j))-T2(i,j)/(R2(i,i)*R2(j,j)))/sqrt((1/(R1(i,i)*R1(j,j))/n1*(1+bx(i,j)^2*R1(i,i)/R1(j,j))+1/(R2(i,i)*R2(j,j))/n2*(1+by(i,j)^2*R2(i,i)/R2(j,j))));\
            end\
        end\
      \
        z=0:0.01:floor(2*sqrt(log(p))+1);\
        s=size(z);\
        le=s(2);\
        for k=1:le\
            rr(k)=(p^2-p)/2*(2-2*normcdf(z(k)))/max(sum(sum(abs(diff)>=z(k))),1);\
        end\
        t=min(find((rr-alpha)<=0));\
        ff(l,ii)=sum(sum(abs(diff(OOO(1:(p-1),:)==0))>=z(t)))/max(sum(sum(abs(diff)>=z(t))),1);\
        sss(ii)=0;\
        for kk=1:10\
            sss(ii)=((sum(sum(abs(diff)>=norminv(1-kk/floor(10/(1-normcdf(sqrt(log(p)))))))))/(kk/floor(10/(1-normcdf(sqrt(log(p)))))*(p^2-p))-1)^2+sss(ii);\
        end\
        fp(l,ii)=(max(sum(sum(abs(diff)>=z(t))),1)-sum(sum(abs(diff(OOO(1:(p-1),:)==0))>=z(t))))/(sum(sum(OOO(1:(p-1),:)~=0))-p+1)*2;\
    end\
    iihat=min(find(sss==min(sss(1:40))));\
    fdp(l)=ff(l,iihat)\
    fpo(l)=fp(l,iihat)\
end\
\
sum(fdp)/rep\
sum(fpo)/rep\
\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \
                            %real data%\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   \
clear\
%% Import the data %%\
filename = ' case.csv';\
largesurv = csvread(filename,1,1);\
filename = ' control.csv';\
smallsurv = csvread(filename,1,1);\
p=size(largesurv,1);\
x=largesurv;\
y=smallsurv;\
n1=size(largesurv,2);\
n2=size(smallsurv,2);\
alpha=0.1;\
for i=1:n1\
    e1(i)=1;\
end\
for i=1:n2\
    e2(i)=1;\
end\
\
\
clear largesurv;\
clear smallsurv;\
x=x';\
y=y';\
sigmax=diag(cov(x));\
sigmay=diag(cov(y));\
  \
\
%% ii: looping for the cross validation, ii also affects the tuning parameter in lasso %%\
\
for ii=1:40\
    for i = 1:p\
        newx=x';\
        newy=y';\
        newx(i,:)=[];\
        newy(i,:)=[];\
        bx(:,i)=lasso(newx',x(:,i),'Lambda',2*ii/40*sqrt(sigmax(i)*log(p)/(n1)));\
        by(:,i)=lasso(newy',y(:,i),'Lambda',2*ii/40*sqrt(sigmay(i)*log(p)/(n2)));\
        %calculate epsilon (residuals)%\
        c1(:,i) = x(:,i)-mean(x(:,i))'*e1'- (newx-mean(newx')'*e1)'*bx(:,i);  \
        c2(:,i) = y(:,i)-mean(y(:,i))'*e2'-(newy-mean(newy')'*e2)'*by(:,i);\
    end\
    \
    % covariances R1 and R2 %\
    R1=c1'*c1/n1;     \
    R2=c2'*c2/n2;  \
    s1=mean(c1.^2);\
    s2=mean(c2.^2);\
    % calculate Tij and Wij\
    for i=1:p-1\
        for j=i+1:p\
            T1(i,j)=R1(i,j)+s1(1,i)*bx(i,j)+s1(1,j)*bx(j-1,i);\
            T2(i,j)=R2(i,j)+s2(1,i)*by(i,j)+s2(1,j)*by(j-1,i);\
            diff(i,j)=(T1(i,j)/(R1(i,i)*R1(j,j))-T2(i,j)/(R2(i,i)*R2(j,j)))/sqrt(1/(R1(i,i)*R1(j,j))/n1*(1+bx(i,j)^2*R1(i,i)/R1(j,j))+1/(R2(i,i)*R2(j,j))/n2*(1+by(i,j)^2*R2(i,i)/R2(j,j)));\
        end\
    end\
    \
    % Calculate t_hat for current tuning parameter\
    z=0:0.01:floor(2*sqrt(log(p))+1);\
    s=size(z);\
    le=s(2);\
    for k=1:le\
        rr(k)=(p^2-p)/2*(2-2*normcdf(z(k)))/max(sum(sum(abs(diff)>=z(k))),1);\
    end\
    t=min(find((rr-alpha)<=0));\
    \
    % Find s_hat use the criteria as described in the algorithm in simulation section \
    sss(ii)=0;\
    for kk=1:10\
        sss(ii)=((sum(sum(abs(diff)>=norminv(1-kk/floor(10/(1-normcdf(sqrt(log(p)))))))))/(kk/floor(10/(1-normcdf(sqrt(log(p)))))*(p^2-p))-1)^2+sss(ii);\
    end\
    [r,c]=find(abs(diff)>=z(t));\
end\
\
% obtain the tuning parameter\
iihat=min(find(sss==min(sss(1:40))))\
\
%% [r,c] corresponds to iihat are the pairs of indices we have selected. %%\
%% To calculate the p-value of the global test, fix ii=40 in the above procedure, then apply the following code. %%\
\
% Calculate M_n\
M = max(max(diff.^2));\
% Calculate p-value\
p-value = 1-exp(-1/sqrt(8*3.14159)*exp(-(M-4*log(p)+log(log(p)))/2))\
}