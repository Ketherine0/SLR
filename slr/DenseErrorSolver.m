function [X,L] = DenseErrorSolver( Y,A,alpha,delta,global_max_iter,lasso_max_iter)

% Objective:  min_{X,L} ||X||_1 + alpha||L||_* + delta/2||E||_F^2 st: Y = AX+L+E
% Augmented Lagrangian Formulation : ||X||_1 + alpha||L||_* + delta/2||E||_F^2 + <Lambda,Y - AX- L - E> + (beta/2)||Y - AX - L- E ||_F^2
% Solution via ADMM:
% Initializations: X=0?, L=0?, E=0?, Lambda=ones?, beta=?
% Iterations:
% (1) Solve L_{k+1}= argmin L : alpha||L||_* + (beta/2)||Y- AX_k - L - E +(1/beta)Lambda_k||_F^2 
%                  = argmin L : ||L||_* + (beta/2alpha)||Y- AX_k - L - E +(1/beta)Lambda_k||_F^2 
% -> L_{k+1}=D_{(alpha/beta)}(Y - AX_k -E + (1/beta)Lambda_k)= US_{(alpha/beta)}(Sigma)V^t, for the SVD of Y-Ax_k -E -(1/beta)Lambda_k.
% (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - AX_k - L -E +(1/beta)Lambda_k||_F^2 
% -> Lasso problem. Should be solved using ADMM
% (2.5) Solve E_{k+1} = argmin E : delta/2||E||_F^2 + <Lambda,Y - AX- L - E> + (beta/2)||Y - AX - L- E ||_F^2
% -> E_{k+1} = (Lambda + beta(Y - AX - L))/(beta + delta)
% (3) Lambda_{k+1}=Lambda_k+ beta(Y - Ax_k - L - E)

[M,K]= size(Y);
[~,N]=size(A);

X=zeros(N,K);
L=zeros(M,K);
E=zeros(M,K);
Lambda=ones(M,K);

At = A';
AtA = At*A ;
tau = eigs(AtA,1,'lm');
tauInv = 1/tau;

beta=(20*M*K)/sum(sum(abs(Y)));
betaInv = 1/beta ;
betaTauInv = betaInv*tauInv ;

converged_main = 0 ;
%maxIter = 200 ;
maxIter=global_max_iter;
iter = 0;
tolX = 1e-6 ;
tolL = 1e-6 ;

while ~converged_main 
   iter= iter+1;
   fprintf('Global Iteration %d of %d \n',iter,maxIter);
   X_old=X;
   L_old=L;   
   E_old =E;
   %% (1) Solve L_{k+1}= argmin L : alpha||L||_* + (beta/2alpha)||Y-AX_k- L + (1/beta)Lambda_k||_F^2 
   tempM=Y - A*X - E + (1/beta)*Lambda;
   [U,S,V]=svd(tempM);
   S=shrink(S,(alpha/beta));
   L=U*S*V';
   
   %% (2) Solve X_{k+1}= argmin X : ||X||_1 + (beta/2)||Y - AX_k - L + (1/beta)Lambda_k||_F^2 
   %% Reduced to x_{k+1}= argmin x : ||x||_1 + (beta/2)||b-Ax||_F^2 per column
   %% Reduced to x_{k+1}= argmin x : (2/beta)||x||_1 + ||b-Ax||_F^2 per column
   b=(Y - L - E + (1/beta)*Lambda);
   Atb=At*b;
   for c=1:K
       X(:,c)=FastIllinoisSolver(AtA,Atb(:,c),X(:,c),tauInv,betaTauInv,lasso_max_iter);
   end
   %% (2.5) Solve E_{k+1} = argmin E : delta/2||E||_F^2 + <Lambda,Y - AX- L - E> + (beta/2)||Y - AX - L- E ||_F^2
   %% -> E_{k+1} = (Lambda + beta(Y - AX - L))/(beta + delta)
   E=(Lambda + beta*(Y - A*X - L))/(beta + delta);
   
   %% (3) Lambda_{k+1}=Lambda_k+ beta(Y-Ax_k-L-E)
   Lambda = Lambda + beta*(Y-A*X-L-E);
   
   %% Stopping Criteria
   if (((norm(X_old - X) < tolX*norm(X_old)) && (norm(L_old - L) < tolL*norm(L_old))) || iter > maxIter)
   converged_main = 1 ;
   end
   
   %% Print Error
   
   energy=sum(sum(abs(X)))+alpha*sum(diag(S))+ (delta/2)*(sum(sum(E.*E)));
   fprintf('L1 norm of X = %f \n', sum(sum(abs(X))));
   fprintf('Nuclear norm of L = %f \n', sum(diag(S)));
   fprintf('Frobenius norm of E = %f \n', sum(sum(E.*E)));
   fprintf('EnergyFunction = %f \n', energy);
   constraint_error=sum(sum((Y - A*X - L - E).*(Y - A*X - L - E)));
   fprintf('Constraint Error = %f \n', constraint_error);
end
     
end

