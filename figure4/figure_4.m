%% problem setup
clear

%%

rng(16)

% System and problem parameters
Astar=1.2; Bstar=0.9; n=1; m=1;
r=0.2; % uncertainty
A0=1.1; B0=1; % 0.2 <= A <=1.3, 0.8 <= B <=1.2
Q=1; R=1;

x_max =10; u_max =10;
Bmax=B0+r;

K=-1; rho=0.5;
w_max=0.1;
eta_max = 0.01;



T0=2000;
W = 5; % MPC horizon
T= T0+W;
update_ind = 100:100:T0;

% target trajectory
aa= (1:T)/20;
target = 8*sin(aa)';





%% RMPC parameters
% construct matrix M used in RMPC
J = zeros(W);
for i=2:W
    J(i,i-1)=1;
end


% construct matrix M used in RMPC
AK0=A0+B0*K;
M = zeros(W);
for i=1:W
    for j =1:i
        M(i,j) = AK0^(i-j)*B0;
    end
end

% construct a_vec used in RMPC
a_vec = zeros(W,1);
for i=1:W
    a_vec(i)=AK0^(i);
end


%% generate different random noises
w_traj = w_max*(2*rand(T+1,1)-1); % random disturbance
eta_traj=eta_max*(sign(2*rand(T+1,1)-1));  % bernoulli {-1,1} exploration noise

alpha =  (0:0.05:pi)';
n_direction = length(alpha)

r = 0.2;
r_SM = r;
r_LS = r;

%% Set Membership

x_sm_traj = zeros(T+1,1);
x_sm_traj(1) = 5;
u_sm_traj=zeros(T,1);
z_sm_traj=zeros(2,T);

r_SM_list = [];
for t=1: T0
    current_target = target(t+1:t+W);
    
    %
    u_sm_traj(t)  = rmpc(x_sm_traj(t,:), B0, K, eta_traj(t), current_target, W, r_SM, x_max, u_max, w_max, J, M, a_vec);
    %
    
    x_sm_traj(t+1)=Astar* x_sm_traj(t)+ Bstar*u_sm_traj(t)+w_traj(t);
    z_sm_traj(:,t)= [x_sm_traj(t); u_sm_traj(t)];
    
    if any(update_ind == t)
        [G_gamma, h_gamma] = set_member(x_sm_traj(1:t+1,:), u_sm_traj(1:t,:), w_traj(1:t,:), w_max, t);
        r_SM = SM_diameter(G_gamma,h_gamma, alpha);
        r_SM_list = [r_SM_list,r_SM];
        r_SM = min(r, r_SM);
        
        
    end
end

%% Least Square

x_ls_traj = zeros(T+1,1);
x_ls_traj(1) = 5;
u_ls_traj=zeros(T,1);
z_ls_traj=zeros(2,T);

S= Astar^2+Bstar^2;

r_LS_list = [];
diam_index = [];
for t=1: T0
    current_target = target(t+1:t+W);
    
    %
    u_ls_traj(t)  = rmpc(x_ls_traj(t,:), B0, K, eta_traj(t), current_target, W, r_LS, x_max, u_max, w_max, J, M, a_vec);
    %
    
    x_ls_traj(t+1)=Astar* x_ls_traj(t)+ Bstar*u_ls_traj(t)+w_traj(t);
    z_ls_traj(:,t)= [x_ls_traj(t); u_ls_traj(t)];
    
    if any(update_ind == t)
        [~, diam_lse] = least_square(x_ls_traj(1:t+1,:), u_ls_traj(1:t,:), t, w_max);
        r_LS = min([1.5*diam_dean(t), diam_lse]);
        r_LS_list = [r_LS_list, r_LS];
        r_LS = min(r, r_LS);
        
        diam_index = [diam_index, t];
    end
end

%% ground truth RMPC

x_traj = zeros(T+1,1);
x_traj(1) = 5;
u_traj=zeros(T,1);
z_traj=zeros(2,T);

for t=1: T0
    current_target = target(t+1:t+W);
    
    %
    u_traj(t)  = rmpc(x_traj(t,:), B0, K, eta_traj(t), current_target, W, 0.01, x_max, u_max, w_max, J, M, a_vec);
    %
    
    x_traj(t+1)=Astar* x_traj(t)+ Bstar*u_traj(t)+w_traj(t);
    z_traj(:,t)= [x_traj(t); u_traj(t)];
    
    
end




%% Figure 4: plotting every data point in a continuous line and every 18 data point for marker

k = 1200;
sp = 1:1:k;
marker_space = 1:18:k;


ax1 = gca;
ax1.FontSize = 22;
figure(1)
hold on
set(gca,'fontname','Times')  % Set it to times
plot(sp, x_traj(sp),'v-','LineWidth',3,'MarkerSize',8, 'MarkerIndices', marker_space)
plot(sp,x_ls_traj(sp), '*-','LineWidth',2,  'MarkerSize',8, 'MarkerIndices', marker_space)
plot(sp,x_sm_traj(sp),'LineWidth',3)

leg = legend('OPT','LSE','SME')
set(leg,'FontSize',20)




xl=xlabel('T')
set(xl,'FontSize',20);
set(xl,'FontName', 'Times')



yl=ylabel('State')
set(yl,'FontSize',20);
set(yl,'FontName', 'Times')
box on


%% Uncertainty functions: SM + LS

function [point_lse, diam_lse] = least_square(x_list, u_list, T, w_max)
% lambda is LSE's regularizer
% S is (a, b)'s upper bound
% delta is failure prob
% n is dim(x)

n=1;
S = 1.2;
delta =0.1;
lambda =0.01;


L=w_max;

% generate z_list= (x0,u0; x1, u1;...)
z_list = zeros(T,2);
z_list(:,1)=x_list(1:end-1);
z_list(:,2)=u_list;
z_list;


V = lambda*eye(2) + z_list'*z_list;
point_lse = inv(V)*z_list'*x_list(2:end);

% confidence region
beta_inner = sqrt(det(V))/sqrt(det(lambda*eye(2)));
% 2*log(beta_inner/delta);
beta_inner_1 = sqrt(2*log(beta_inner/delta))*L*n;
beta = (beta_inner_1+sqrt(lambda)*S)^2;

% so the region is gamma=(theta-thetahat)
% gamma' V gamma <=beta
% do orthogonal transform: V= eigvectors D eigvectors'

[eigvectors, eigvalues]=eig(V);
% eigvectors'*eigvectors;

% so ellips is gamma_1^2/(beta/eigvalue_1) + gamma_2^2/(beta/eigvalue_2)=1
ellips_1= beta/eigvalues(1,1);
ellips_2= beta/eigvalues(2,2);

diam_lse = 2* max(sqrt(ellips_1), sqrt(ellips_2));

end

function diam = diam_dean(T)
% Using Dean 2018 paper for the 2d uncertainty set with better bounds
rho = 0.8;
Cu = 1;
CK = sqrt(2)*Cu;
x0_norm = 5;


B_norm = 0.9;
d= 1;
n=1;
sigma_w = 0.1;
sigma_eta = 0.1;
delta = 0.05;

diam = sigma_w*Cu/sigma_eta * sqrt((d+n)/T) * sqrt(log(d*Cu^2/delta...
    + sigma_w^2*rho^2*Cu^2 *CK^2/ (sigma_eta^2 * delta * (1-rho^2))...
    * ( 1 + B_norm + x0_norm/sigma_w / sqrt(T))));
end

function [G_gamma, h_gamma] = set_member(x_list, u_list, w_list, w_max,T)
% set membership estimation
% estimate a, b
% theta =(a; b)
% gamma =(a-a* ; b-b*)
% zt=(xt; ut)

% zt' gamma-wt<wmax, so zt' gamma<wmax+wt
% zt'gamma -wt>-wmax, so (-zt)'gamma< wmax -wt
%

% generate z_list= (x0,u0; x1, u1;...)
z_list = zeros(T,2);
z_list(:,1)=x_list(1:end-1);
z_list(:,2)=u_list;
z_list;

% LHS matrix G
G_gamma=[z_list;-z_list];

% rhl vector
h_gamma = [w_max+ w_list; w_max - w_list];
end


function diam_SM = SM_diameter(G,h, alpha)
% require: Ggamma <= h must contain 0 as interior point
% generate samples on uniform sphere, one way is just cos(alpha),
% sin(alpha)

direction_x=cos(alpha);
direction_y=sin(alpha);
direction = [direction_x, direction_y];

% number of directions.
% plot(alpha, direction_x, alpha, direction_y)

% for every direction, figure out the maximum value and the minimum value
% on this direction, i.e., max r s.t. G*direction(i)*r <=h, and
% min r s.t. G*direction(i)*r <=h
% this can be solved by linear programming

boundary_points = zeros(length(alpha),2);
r_max_list = zeros(length(alpha),1);
r_min_list = zeros(length(alpha),1);

% trouble: if we don't re-normalize to center 0, some directions are
% infeasible
options =  optimset('Display','off');
for i = 1: length(alpha)
    % max r=min -r s.t. G*direction(i)*r <=h
    f_1 = -1;
    A_ineq = G*(direction(i,:)');
    r_solution = linprog(f_1,A_ineq, h,[],[],[],[],[],options);
    r_max_list(i)=r_solution;
    boundary_points(2*i-1,:)=r_max_list(i)*(direction(i,:)');
    % min r s.t. G*direction(i)*r <=h
    f_2=1;
    r_solution = linprog(f_2,A_ineq, h,[],[],[],[],[],options);
    r_min_list(i)=r_solution;
    boundary_points(2*i,:)=r_min_list(i)*direction(i,:);
end



boundary_points;
r_max_list;
r_min_list;

% diameter of the SM:
% pick any two boundary points, and solve the maximum difference
diam_SM =max(r_max_list-r_min_list); % maximum distance on all directions

% 2n_direction choose 2
boundary_points_index = 1: 2*length(alpha);
possible_pairs = nchoosek(boundary_points_index,2);
diam_x1 = boundary_points(possible_pairs(:,1),:);
diam_x2 = boundary_points(possible_pairs(:,2),:);

diam_SM=max(max(sqrt(diag((diam_x1-diam_x2)*(diam_x1-diam_x2)'))),diam_SM);


end

%% RMPC functions
function ut = rmpc(xt, B0, K, eta_t,current_target, W, r, x_max, u_max, w_max, J, M, a_vec)

v_vec_current = RMPC(xt, B0, current_target, W, r, x_max, u_max, w_max,J, M, a_vec);

v0_current = v_vec_current(1);

ut = K*xt + v0_current + eta_t;
end


function v_vec = RMPC(xt, B0, current_target, W, r, x_max, u_max, w_max, J, M, a_vec)
% RMPC optimization at time t

eta_max=0.1;
rho = 0.4; % decay rate of the closed loop under K


Bmax = B0+r; % worst-case B matrix

% construct tube
d = zeros(W+1, 1);
d(2)= w_max + Bmax*eta_max + 2*r*x_max + 2*r*u_max;  % this is very loose bound for tube size of state
for i =3:W+1
    d(i)= d(i-1)+rho^(i-2)*d(2);
end



e1=zeros(W,1); e1(1)=1;

x_tighten = x_max*ones(W,1)-d(2:end); % tighten x and u's constraints
u_tighten = u_max*ones(W,1)-d(1:end-1);



H= M'*M + (eye(W)+ J*M)'*(eye(W)+ J*M);
f= (a_vec*xt-current_target)'*M+ (J*a_vec*xt-xt*e1)'*(J*M+eye(W));

A_ineq = [M; -M; J*M+eye(W); -J*M-eye(W)];
b_ineq = cat(1, x_tighten - a_vec*xt, x_tighten + a_vec*xt, u_tighten - xt* e1 -J*a_vec*xt,u_tighten +xt* e1 +J*a_vec*xt);

options =  optimset('Display','off');

Aeq=[]; beq=[];lb=[];ub=[];x0=[];

[v_vec,fval,exitflag,output,lambda] = quadprog(H,f',A_ineq,b_ineq,Aeq,beq,lb,ub,x0,options);
end



