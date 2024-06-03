
%% generate T_list(end) trajectories and plot Figure 1a
clear all
rng(4)

n=2;
A_star = 0.8;
B_star = 1;
T_list = [5,10:20:350]'

delta = 0.1;
sigma = 1;
n_x = 1;
n_u = 1;
n_z = n_x+n_u;




T_max = T_list(end);


% generate disturbances by TrunGaussian(0.5,[-wmax, wmax])
sigma_w=0.5;
w_max = 2*sigma_w;
range_w =[-w_max, w_max];
[w_list_max,~,~] = TruncatedGaussian(sigma_w, range_w, [T_max,1]);

% generate state trajectory
x0 = 0;
x_list_max = zeros(T_max+1,1); % [x0, x1,..., xT+1]
x_list_max(1,:)=x0;

% generate ut by iid TrunGaussian(0.5,[-wmax, wmax])
sigma_u=0.5;
u_max = 2* sigma_u;
range_u =[-u_max, u_max];
% u_list_max = (2*rand(T_max,1)-1)*u_max;
[u_list_max,~,~] = TruncatedGaussian(sigma_u, range_u, [T_max,1]);


% generate a trajectory
for t = 1: T_max
    x_list_max(t+1)= A_star* x_list_max(t)+B_star*u_list_max(t)+w_list_max(t);
end

% generate sampling directions
alpha =  (0:0.05:pi)';
n_direction = length(alpha)

% generate LSE confidence region's parameters
n=1;
delta =0.1;
S= sqrt(A_star^2+B_star^2);
lambda =0.1;



boundary_SM = zeros(2*length(alpha),2,length(T_list));
diameter_SM = zeros(length(T_list),1);


% LSE computation using Yingying's implementation
diameter_LSE = zeros(length(T_list), 1);
beta_LSE =zeros(length(T_list),1);
V_LSE = zeros(2,2,length(T_list));
estimator_LSE = zeros(length(T_list),2);


% LSE computation using 2011 paper
old_beta_LSE =zeros(length(T_list),1);
old_V_LSE = zeros(2,2,length(T_list));
old_estimator_LSE = zeros(length(T_list),2);
old_diameter_LSE = zeros(length(T_list), 1);



for i = 1: length(T_list)
    T = T_list(i)
    
    % SM
    x_list = x_list_max(1:T+1,:);
    u_list = u_list_max(1:T,:);
    w_list = w_list_max(1:T,:);
    z_list = [x_list(1:T,:) u_list];
    [G_gamma, h_gamma] = set_member(x_list, u_list, w_list, w_max,T);
    
    [diam_SM, boundary_points] = SM_diameter(G_gamma,h_gamma, alpha);
    diameter_SM(i)=diam_SM;
    boundary_SM(:,:,i)=boundary_points;
    
    %     % old LSE
    [old_point_lse, old_diam_lse, old_beta, old_V] = least_square(x_list, u_list, T, w_max, lambda, S,delta,n);
    old_diameter_LSE(i)=old_diam_lse; %%%%
    old_beta_LSE(i) = old_beta;
    old_V_LSE(:,:,i)=old_V;
    old_estimator_LSE(i,:)=old_point_lse;
    
    % new LSE
    V_LSE(:,:,i) = eye(n_x+n_u);
    [estimator_LSE(i,:), diameter_LSE(i)] = new_LSE_diam_fxn(x_list,z_list,n_x,n_u, n_z, delta,sigma);
    beta_LSE(i) = diameter_LSE(i);
    
    
end
beta_LSE = beta_LSE .^2;



% Get the minimum of the two
new_geq_old = diameter_LSE >= old_diam_lse;



%% Figure 1a

figure(2);
fig =semilogy(T_list, diameter_SM, T_list, min(diameter_LSE, old_diameter_LSE),  'linewidth',3)
xlim([0 350])
legend('SME', 'LSE')

xlabel('T','FontSize', 22)
ylabel('Diameter','FontSize', 22)
ax = gca;
ax.FontSize = 22;
set(gca,'fontname','Times')


%% true_T = 5 / T = 1  (Figure 1b)

% Get figure parameters

T = 1;

if new_geq_old(T) == true
    
    A_LS = old_estimator_LSE(T,1);
    B_LS = old_estimator_LSE(T,2);
    
    
    M = old_V_LSE(:,:,T)
    b = old_beta_LSE(T);
    
    
else
    
    A_LS = estimator_LSE(T,1);
    B_LS = estimator_LSE(T,2);
    
    
    M = V_LSE(:,:,T)
    b = beta_LSE(T);
    
    
end



figure(1)
hold on
% Contouring LSE
syms x y
eq = [x-A_LS,y-B_LS] * M * [x-A_LS,y-B_LS]' ==b;
z = fimplicit(eq,'Color', [0 0.4470 0.7410],'HandleVisibility', 'off');


% Filled plot of LSE
[a,b,theta] = get_ellipse_param(M,b);
plot_ellipse(a/2,b/2,A_LS,B_LS,theta,[0 0.4470 0.7410])


% set membership
A = 0.8;
B = 1;
% load('boundary_SM.mat')
points = boundary_SM(:,:,T);
k = convhull(points(:,1),points(:,2));
pgon = polyshape(points(k,1)+A,points(k,2)+B);
plot(pgon,'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha',1,'EdgeColor', [0.9290 0.6940 0.1250])


% True point
plot(A,B,'rPentagram', 'markerSize', 10, 'LineWidth', 1.5)
ax = gca;
ax.FontSize = 22;
legend('LSE','SME', 'True')

box on

xlabel('A')
ylabel('B')

hold off

%% true_T = 250 / T = 14 (Figure 1c)
T = 14;
if new_geq_old(T) == true
    
    A_LS = old_estimator_LSE(T,1);
    B_LS = old_estimator_LSE(T,2);
    
    
    M = old_V_LSE(:,:,T)
    b = old_beta_LSE(T);
    
    
else
    
    A_LS = estimator_LSE(T,1);
    B_LS = estimator_LSE(T,2);
    
    
    M = V_LSE(:,:,T)
    b = beta_LSE(T);
    
    
end


figure
hold on
% Contouring LSE
syms x y
eq = [x-A_LS,y-B_LS] * M * [x-A_LS,y-B_LS]' ==b;
z = fimplicit(eq,'Color', [0 0.4470 0.7410],'HandleVisibility', 'off');


% Filled plot of LSE
[a,b,theta] = get_ellipse_param(M,b);


plot_ellipse(a/2,b/2,A_LS,B_LS,theta,[0 0.4470 0.7410])


% set membership
A = 0.8;
B = 1;
% load('boundary_SM.mat')
points = boundary_SM(:,:,T);
k = convhull(points(:,1),points(:,2));
pgon = polyshape(points(k,1)+A,points(k,2)+B);
plot(pgon,'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha',1,'EdgeColor', [0.9290 0.6940 0.1250])


% True point
plot(A,B,'rPentagram', 'markerSize', 8, 'LineWidth', 1.5)
ax = gca;
ax.FontSize = 22;
legend('LSE','SME', 'True')


xlim([0.5,1.1])
ylim([0.4,1.7])
box on

xlabel('A')
ylabel('B')

hold off

%% functions

function [a,b,theta] = get_ellipse_param(M,b)
v1 = M(1,1);
v2 = M(1,2);
v3 = M(2,2);

[P, E] = eig(M);

theta = acos(P(1,1)) * 180 / pi;


% current ellipse is Ax^2 + By^2 = C where A = eig1 B = eig2, C = b
% then put into the standard ellipse form (1/a^2) x^2 + (1/b^2) y^2 = 1 and
% then a is the semi-major axis, b is the semi-minor axis,
% and width = 2a, height = 2b

semi_axis_major = sqrt(b/E(1,1));
semi_axis_minor = sqrt(b/E(2,2));

a = 2* semi_axis_major;
b = 2* semi_axis_minor;



end




function plot_ellipse(a,b,cx,cy,angle,color)
%a: width in pixels
%b: height in pixels
%cx: horizontal center
%cy: vertical center
%angle: orientation ellipse in degrees
%color: color code (e.g., 'r' or [0.4 0.5 0.1])
angle=angle/180*pi;
r=0:0.1:2*pi+0.1;
p=[(a*cos(r))' (b*sin(r))'];
alpha=[cos(angle) -sin(angle)
    sin(angle) cos(angle)];

p1=p*alpha;

patch(cx+p1(:,1),cy+p1(:,2),color,'EdgeColor',color);

end

%% functions

function [point_lse, diam_lse, beta,V] = least_square(x_list, u_list, T, w_max, lambda, S,delta,n)
% lambda is LSE's regularizer
% S is (a, b)'s upper bound
% delta is failure prob
% n is dim(x)

L=w_max;



% generate  z_list= (x0,u0; x1, u1;...)
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

% So SM's set is G*theta<=h


end
function [diam_SM boundary_points] = SM_diameter(G,h, alpha)
% require: Ggamma <= h must contain 0 as interior point
% generate samples on uniform sphere, one way is just cos(alpha),
% sin(alpha)
G;
h;



direction_x=cos(alpha);
direction_y=sin(alpha);
direction = [direction_x, direction_y];

% number of directions.
% plot(alpha, direction_x, alpha, direction_y)

% for every direction, figure out the maximum value and the minimum value
% on this direction, i.e., max r s.t. G*direction(i)*r <=h, and
% min r s.t. G*direction(i)*r <=h
% this can be solved by linear programming

boundary_points = zeros(2*length(alpha),2);
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

function [theta_new_LSE, diam_new_LSE] = new_LSE_diam_fxn(x_list,z_list,n_x,n_u, n_z, delta,sigma)
% z_list is data of xt ut in each row, inverse of math formulation
% x_list is x1,..., xT+1 in each row

Gamma_T = z_list'*z_list; % Gamma_T=sum_t zt zt'

x_list_end = x_list(2:end,:); % x2,...,xT+1

theta_new_LSE = x_list_end'*z_list *pinv(Gamma_T);




% compute parameters: lambda1, lambda2, P, upsilon


% we have U_eig_gauss*Alpha_gauss*U_eig_gauss'=Gamma_T_gauss
% and U_eig_gauss^-1=U_eig_gauss' orthogonal
[U_eig, Alpha_matrix] = eig(Gamma_T);
alpha_list = diag(Alpha_matrix);

diam_LSE_new_options = zeros(n_z-1,1);
for p = 1 : n_z-1
    
    
    % p = n_z-1; % I need to iterate over 1:n_z-1, to take minimum
    
    P0 = diag([ones(p,1);zeros(n_z-p,1)]);
    lambda_1 = min(alpha_list(1:p));
    lambda_2 = min(alpha_list(p+1:end));
    
    upsilon = lambda_1;
    
    P_matrix = U_eig*P0*U_eig';
    
    
    % generate v1,..., vn_z, kappa1, kappa2
    
    [Q_factor, R_factor] =qr(P_matrix); % check if R's last n_z-p diagonals are 0 or close to 0
    
    % Q_factor_gauss=(q1,..., q_nz)=(v1,...,v_nz)
    
    kappa_1_options = zeros(p,1);
    kappa_2_options = zeros(n_z-p,1);
    
    for i = 1:p
        kappa_1_options(i)=Q_factor(:,i)'*Gamma_T*Q_factor(:,i)/lambda_1;
    end
    
    for i = 1:n_z-p
        kappa_2_options(i)=Q_factor(:,i+p)'*Gamma_T*Q_factor(:,i+p)/lambda_2;
    end
    
    
    kappa_1=max(kappa_1_options);
    kappa_2=max(kappa_2_options);
    
    diam_LSE_new_sq = 12*n_x*p*kappa_1*log(3*n_x*n_z*kappa_1/delta)/lambda_1 ...
        + 48*n_x*(n_z-p)*kappa_2*log(3*n_x*n_z*kappa_2/delta)/lambda_2;
    diam_LSE_new_options(p) = sqrt( diam_LSE_new_sq)*sigma;
end
diam_new_LSE = min(diam_LSE_new_options);




end
















