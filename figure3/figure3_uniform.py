import numpy as np
import cvxpy as cp
import tqdm
import os
import datetime as dt
import pickle
from math import erf, sqrt
# from ICML_helper import stabilize, generate_truncGauss
from minimax_tilting_sampler import TruncatedMVN
from scipy.spatial.distance import pdist

def compute_uniform_variance(a, b):
    return sqrt((b-a)**2/12)


def sample_sphere(d,N=100):
    # dimension d and makes 10 samples
    x = np.random.normal(size=(N, d))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x


def get_eta_theta(x_list, nx, u_list = None):
    if u_list is not None:
        X = np.vstack([np.kron(np.eye(nx), xt) for xt in x_list[:-1]])
        U = np.vstack([np.kron(np.eye(nx),ut ) for ut in u_list])
        G = np.hstack((X,U))
        nu = u_list.shape[1]
        d = nx*(nx+nu)
    else:
        G = np.vstack([np.kron(np.eye(nx), xt) for xt in x_list[:-1]])
        d = nx**2 #nx*(nx+nu)
        
    theta = cp.Variable((d,1))
    eta = cp.Variable(1)
         
    h_ub = np.vstack([x_.reshape((nx,1))  for x_ in x_list[1:]]) + eta
    h_lb = np.vstack([x_.reshape((nx,1))  for x_ in x_list[1:]]) - eta

    constraints = [eta>=0]
    constraints += [G@theta <= h_ub]
    constraints += [G@theta >= h_lb]

    prob = cp.Problem(cp.Minimize(eta), constraints)
    prob.solve(verbose=False, solver = cp.MOSEK)
    if prob.status != 'optimal':
      print('SOLVER FAILED, solution status: {}'.format(prob.status))
      # print('Optimization solved susccessfully!')
    eta_hat = eta.value
    theta_hat = theta.value

    return eta_hat.flatten()


def get_diam(x,u,nx,nu,W,lb,ub, prev_bound=None, Z=None):
    '''compute the diameter of points of box surrounding points'''
    # theta = [A11,A12,A21,A22,B1,B2]
    # https://arxiv.org/pdf/1905.11877.pdf Algorithm 3.
    # d: dimension of the parameter space
    # r: Radius of a ball that contains the parameter polytope
    # epsilon: Steiner point approximation accuracy
    # x_list,u_list: state and control trajectory
    # W,lb,ub: disturbance bound, lower and upper bound on the parameter space.
    G, h_lb, h_ub = get_params_for_diameter(x, u,nx, W)
    d = nx*(nx)
    # mat = np.eye(nx)
    if Z is None:
        n_samp = 15*d
        Z = sample_sphere(d,N=n_samp) # Z is a N by d matrix with N iid sampled R^d gaussian vector
    # print(Z.shape)
    # Z = np.vstack([np.eye(d),-np.eye(d)])
    if prev_bound is not None:
        P, lowerbound = support(Z,d,G, h_lb, h_ub ,W,lb,ub, prev_bound)
    else:
        P, lowerbound = support(Z,d,G, h_lb, h_ub ,W,lb,ub)
        
    pairs = pdist(P,metric="euclidean")
    return max(pairs), lowerbound, Z



def cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2))) 


def phi(x):
    return 1/(2*np.pi) * np.exp(-0.5*x**2)


def compute_truncGaussian_variance(beta, alpha, sig=1.):
    term1 = (beta * phi(beta) - alpha*phi(alpha)) / (cdf(beta) - cdf(alpha))
    term2 = ((phi(alpha) - phi(beta))/(cdf(beta) - cdf(alpha))) ** 2
    v = sig**2 * ( 1 - term1 - term2)
    return sqrt(v)
    
    
def stabilize(A,max_eig=0.8):
    eigs,_ = np.linalg.eig(A)
    spec_rad_A = max(np.abs(eigs))
    A = A / spec_rad_A * max_eig
    return A


def generate_truncGauss(nx, W, T, seed):
    # adapted from github package: https://github.com/brunzema/truncated-mvn-sampler
    
    # zero mean with random cov
    np.random.seed(seed)
    mu = np.zeros(nx)
    # cov = 0.5 - np.random.rand(nx ** 2).reshape((nx, nx))
    # cov = np.triu(cov)
    # cov += cov.T - np.diag(cov.diagonal())
    # cov = np.dot(cov, cov)
    cov = np.eye(nx)

    # constraints
    lb = np.zeros_like(mu) - W
    ub = np.zeros_like(mu) + W

    # create truncated normal and sample from it
    tmvn_w = TruncatedMVN(mu, cov, lb, ub, seed = seed)
    wts = tmvn_w.sample(T).T
    
    ## Sanity Check
    ## With fixed input seed, this function samples the same vectors
    ## wts is returned as (T,nx) shape so wts[n] returns w[n]
    ## Plotting empirical distribution sampled from tmvn
    # plt.hist(wts[:,0], color='lightgreen', ec='black', bins=80)
    
    return wts



def generate_trajectory(true_theta, specs, savedir='out_unspecified'):
    # assume that control action will be generated from truncated Gaussian, if B matrix is part of the dynamics
    specs['true_theta'] = true_theta
    distribution = specs['distribution']
    T = specs['T']
    W = specs['W']
    Nx = specs['Nx']
    seed = specs['seed']
    np.random.seed(seed)
    A = true_theta['A']
    B, u = None, None
    if len(true_theta) > 1:
        B = true_theta['B']
        Nu = B.shape[1]
        u = generate_truncGauss(Nu, W, T, seed)
        
    if distribution == "truncated_gaussian":
        w = generate_truncGauss(Nx, W, T, seed)
    elif distribution == "uniform":
        w = np.random.uniform(low = -W, high = W, size=(T,Nx))
    elif distribution == "beta":
        w = np.random.beta(a=0.5, b=0.5, size=(T,Nx))*2*W - W
    else:
        print('Distribution not supported.')
        
    x = [None]*(T+1)
    x[0] = np.random.normal(size=(Nx,))
    for t in range(T):
        x[t+1] = A @ x[t] + w[t]
        if len(true_theta) > 1:
            x[t+1] = A @ x[t] + B @ u[t] + w[t]
    trajectory = dict(A=A, B=B, x=np.array(x), u=u, w_list=w, W=W, specs=specs)
    
    # save data   
    if savedir != '':
        os.makedirs(savedir, exist_ok=True)
    tz = dt.timezone(dt.timedelta(hours=-8))  # PST
    start_time = dt.datetime.now(tz)
    filename = os.path.join(savedir, f'trajectory_{distribution}_')
    filename += f'Nx={Nx}_'
    if len(true_theta) > 1:
        filename += f'Nu={Nu}_'
    filename += f'T={T}_'
    filename += f'W={W}_'
    filename += f'seed={seed}_'
    filename += start_time.strftime('%Y%m%d_%H%M%S')
    # with open(f'{filename}.pkl', 'wb') as f:
    #     pickle.dump(file=f, obj=trajectory)
    return trajectory


def support(Z, d, G, h_lb, h_ub , W, lb, ub, prev_bound = None):
    y = cp.Variable((d,1))
    constraints = [y <= ub*np.ones((d,1))]
    constraints += [y >= lb*np.ones((d,1))]
    constraints += [G@y <= h_ub]
    constraints += [G@y >= h_lb]
    if prev_bound is not None:
        lower = prev_bound['lower']
        prev_Z = prev_bound['Z']
        constraints +=[prev_Z @ y <= lower.reshape((prev_Z.shape[0],1))]
        
    y_values = np.zeros(shape=(Z.shape[0], d))
    lowerbound = np.zeros((Z.shape[0],))
    for i, zi in enumerate(Z):
        prob = cp.Problem(cp.Maximize(y.T @ zi), constraints)
        prob.solve(cp.MOSEK)
        y_values[i] = y.value.flatten()
        lowerbound[i] = prob.value
    return y_values, lowerbound


def get_box_size(x, u, nx, nu, W, lb, ub, prev_bound=None): 
    '''compute the diameter of points of box surrounding points'''
    # theta = [A11,A12,A21,A22,B1,B2]
    # https://arxiv.org/pdf/1905.11877.pdf Algorithm 3.
    # d: dimension of the parameter space
    # r: Radius of a ball that contains the parameter polytope
    # epsilon: Steiner point approximation accuracy
    # x_list,u_list: state and control trajectory
    # W,lb,ub: disturbance bound, lower and upper bound on the parameter space.
    G, h_lb, h_ub = get_params_for_diameter(x, u,nx, W)
    d = nx*(nx)
    Z = np.vstack([np.eye(d),-np.eye(d)])
    if prev_bound is not None:
        P = support(Z,d,G, h_lb, h_ub ,W,lb,ub, prev_bound)
    else:
        P = support(Z,d,G, h_lb, h_ub ,W,lb,ub)
    box_dims = abs(np.diag(P[:d,:d]) - np.diag(P[d:,:d]))
    diam = np.linalg.norm(box_dims,ord=2)
    return diam, P


def get_params_for_diameter(x, u, nx, W):
    G = np.vstack([np.kron(np.eye(nx), xt) for xt in x[:-1]])
    h_ub = np.vstack([x_.reshape((nx,1))  for x_ in x[1:]]) + W
    h_lb = np.vstack([x_.reshape((nx,1))  for x_ in x[1:]]) - W
    return G,h_lb, h_ub


def get_SM_diam_no_skip(x, u, w, specs, savedir='out_unspecified', max_diam=100, w_hat = None, skip_to_end = False, optional_name = None):
    # Diameter using box approximation of the membership set using (eq 3.5) from "Robust Adaptive Model Predictive Control: Performance and Parameter Estimation" by Lu et al, 2021.
    # x: x[0]...x[T]
    # u: u[0]...u[T-1]
    # savedir: the directory to which the data will be save
    # max_diam: some rough initialization bound on the diameter for the parameters before the SM set becomes compact with enough data
    
    T = specs['T']
    W = specs['W']
    if 'lb' in specs.keys():
        w_lb = specs['lb']
        w_ub = specs['ub']
    else:
        w_lb = -W
        w_ub = W
    if u is not None:
        z = np.hstack([x[:T], u])
        Nu = u.shape[1]
    else:
        z = x
    Nx = x.shape[1]
    d = z.shape[1] * Nx # (Nx+Nu) * Nx parameters
    M =  np.vstack([np.eye(d),-np.eye(d)]) # Matrix corresponding to upper and lower bound of each unknown parameter
    O = np.vstack([np.eye(Nx),-np.eye(Nx)])
    
    # initialization
    diam_SM = np.zeros((T+1,))
    old_SM = np.zeros((T+1,))
    diam_SM[d-1] = max_diam
    mu_t_list = [None] * (T+1)
    mu_t_list[d-1] = np.ones(shape=(2*d,)) * max_diam
    
    if skip_to_end:
        if w_hat is not None:
            W = w_hat[-1]
        diam_SM, _ , _ = get_diam(x, None, Nx, 0, W, -100, 100)
        print('----- Seed = {}, Nx = {}, SM diam = {}, W = {}-----'.format(specs['seed'], Nx, diam_SM, W))
    else:
        for t in tqdm.tqdm(range(d, T+1)): 
            if w_hat is not None:
                W = w_hat[t]
            diam, Y,_ = get_diam(x[:t], None, Nx, 0, W, -100, 100)
            diam_SM[t] = diam
            print(diam)

    # save data   
    if savedir != '':
        os.makedirs(savedir, exist_ok=True)
    distribution = specs['distribution']
    tz = dt.timezone(dt.timedelta(hours=-8))  # PST
    start_time = dt.datetime.now(tz)
    

    if w_hat is not None:
        if optional_name is not None:
            filename = os.path.join(savedir, f'SM-UCB_{optional_name}_diameter_{distribution}_')
        else:
            filename = os.path.join(savedir, f'SM-UCB_diameter_{distribution}_')
    else:
        filename = os.path.join(savedir, f'SM_diameter_{distribution}_')
    filename += f'Nx={Nx}_'
    if u is not None:
        Nu = u[0].shape[0]
        filename += f'Nu={Nu}_'
    filename += f'T={T}_'
    filename += f'W={W}_'
    filename += f'seed={seed}_'
    filename += start_time.strftime('%Y%m%d_%H%M%S')
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(file=f, obj=dict(diam_SM=diam_SM, x=np.array(x), u=u, w_list=w, W=W, specs=specs, w_hat=w_hat))
        
    return diam_SM



def get_SM_diam(x, u, w, specs, L, savedir='out_unspecified', max_diam=100,  skip = False):
    # Diameter using box approximation of the membership set using (eq 3.5) from "Robust Adaptive Model Predictive Control: Performance and Parameter Estimation" by Lu et al, 2021.
    # x: x[0]...x[T]
    # u: u[0]...u[T-1]
    # L: window size for block recursive update, the notation for L is Nu in the reference
    # savedir: the directory to which the data will be save
    # max_diam: some rough initialization bound on the diameter for the parameters before the SM set becomes compact with enough data
    
    T = specs['T']
    W = specs['W']
    if 'lb' in specs.keys():
        w_lb = specs['lb']
        w_ub = specs['ub']
    else:
        w_lb = -W
        w_ub = W
    if u is not None:
        z = np.hstack([x[:T], u])
    else:
        z = x
    Nx = x.shape[1]
    d = z.shape[1] * Nx # (Nx+Nu) * Nx parameters
    M =  np.vstack([np.eye(d),-np.eye(d)]) # Matrix corresponding to upper and lower bound of each unknown parameter
    O = np.vstack([np.eye(Nx),-np.eye(Nx)])
    
    # initialization
    diam_SM = np.zeros((T+1,))
    diam_SM[L-1] = max_diam
    mu_t_list = [None] * (T+1)
    mu_t_list[L-1] = np.ones(shape=(2*d,)) * max_diam
    if skip:
        batch_size = L # Step size to save time
        specs['batch_size'] = batch_size
    else:
        batch_size = 1
    
    Z = sample_sphere(d, 20*d)
    for t in tqdm.tqdm(range(L, T+1, batch_size)): 
        # Using up to x[t], computing P_t, q_t, approximating Theta_t in paper
        # Construct constraints based on the last L data points starting at t = L,...,T (t=L:using z[0]...z[L-1];...; t=T:using z[T-L]...z[T-1] )
        # Using z[t-1], z[t-2], ... z[t-L], constructing P[t],..., P[t-L+1] in reverse order
        '''
        P_t = np.vstack(np.kron(O, z[t-k]) for k in reversed(range(1,L+1))) # k = 1,...L
        # Using x[t], z[t-1], ... z[t-L+1], constructing P[t],..., P[t-L+1]in reverse order 
        q_t = np.hstack([np.hstack([x[t-k] - w_lb, w_ub - x[t-k]]) for k in reversed(range(L))])

        mu_t_list[t] = [None] * (2*d) # bound on all (signed) direction at time t

        for i in range(2*d):
             # Compute the bound on i^th (signed) direction at time t
            theta_i = cp.Variable((d, ))
            constraints = [ P_t @ theta_i  <=  q_t]
            
            if t == L:
                step_size_copy = 1
            else:
                step_size_copy = batch_size
            constraints += [M @ theta_i <= mu_t_list[t-step_size_copy]]
            prob = cp.Problem(cp.Maximize(M[i] @ theta_i), constraints)
            prob.solve()
            mu_t_list[t][i] = prob.value
        diam_SM[t] = np.linalg.norm(np.array(mu_t_list[t][:d]) - np.array(mu_t_list[t][d:]), ord=2)
        '''
        if t == L:
            diam, lowerbound, Z = get_diam(x[t-L:t], None, Nx, 0, W, lb=-100, ub=100, Z=Z)
        else:    
            diam, lowerbound, Z = get_diam(x[t-L:t], None, Nx, 0, W, lb=-100, ub=100, prev_bound = dict(Z=Z, lower=lowerbound), Z=Z)     
        # box_dims = abs(np.diag(Y[:d,:]) - np.diag(Y[d:,:]))
        # diam = np.linalg.norm(box_dims,ord=2)
        diam_SM[t] = diam
        print(diam)
        
        
    # save data   
    if savedir != '':
        os.makedirs(savedir, exist_ok=True)
    specs['window_size'] = L
    distribution = specs['distribution']
    tz = dt.timezone(dt.timedelta(hours=-8))  # PST
    start_time = dt.datetime.now(tz)
    
    filename = os.path.join(savedir, f'SM_diameter_{distribution}_')
    filename += f'Nx={Nx}_'
    if u is not None:
        Nu = u[0].shape[0]
        filename += f'Nu={Nu}_'
    filename += f'T={T}_'
    filename += f'W={W}_'
    filename += f'seed={seed}_'
    filename += f'Window={L}_'
    filename += f'BatchSize={batch_size}_'
    filename += start_time.strftime('%Y%m%d_%H%M%S')
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(file=f, obj=dict(diam_SM=diam_SM, x=np.array(x), u=u, w_list=w, W=W, specs=specs))
    return diam_SM






def get_LSE_diam_old(x, u, w, specs, delta=0.1, sigma = 1.,lam = 0.1, savedir='out_unspecified', variance_scale = 1.0, skip = True):
    # Diameter computation using AY 11': https://proceedings.mlr.press/v19/abbasi-yadkori11a/abbasi-yadkori11a.pdf  
    # lam: weight of regularizer for LS: min_A ||x_t+1-Ax_t||_2^2 + lambda_ ||A||_2^2
    # delta: confidence bound
    # sigma: subGaussian parameter (variance proxy)
    T = specs['T']
    W = specs['W']
    
    x = x / variance_scale
    # u = u / variance_scale
    Nx = x.shape[1]
    Nz = x.shape[1] 
    
    A = specs['true_theta']['A']
    S = np.sqrt(np.trace((A).T @ A) )
    
    # get z[0],...z[T-1]
    if not skip:
        diam = [None] * (T+1)
        for t in range(Nz+1,T+1):
            if u is not None:
                z = np.hstack([x[:t], u[:t]]) # becomes  T by (Nx + Nu)
                Nu = u[0].shape[0]
            else:
                z = x[:T]            
            Vt = lam * np.eye(Nz) + z.T @ z
            print(t)
            Bt = get_Bt(delta, Vt, lam, Nx, sigma, S)
            diam[t] = ellipsoid_volume_diam(Vt, Bt)
            
    else: 
        if u is not None:
            z = np.hstack([x[:T], u]) # becomes  T by (Nx + Nu)
            Nu = u[0].shape[0]
        else:
            z = x[:T]
        Vt = lam * np.eye(Nz) + z.T @ z
        Bt = get_Bt(delta, Vt, lam, Nx, sigma, S)
        diam = ellipsoid_volume_diam(Vt, Bt)
        diam = np.array(diam)

    # save data   
    if savedir != '' and savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    distribution = specs['distribution']
    tz = dt.timezone(dt.timedelta(hours=-8))  # PST
    start_time = dt.datetime.now(tz)
    
    if savedir is not None:
        filename = os.path.join(savedir, f'old_LSE_diameter_{distribution}_')
        filename += f'Nx={Nx}_'
        if u is not None:
            Nu = u[0].shape[0]
            filename += f'Nu={Nu}_'
        filename += f'T={T}_'
        filename += f'W={W}_'
        filename += f'seed={seed}_'
        filename += f'delta={delta}_'
        filename += f'subGaussianL={sigma}_'
        filename += start_time.strftime('%Y%m%d_%H%M%S')
    
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(file=f, obj=dict(diam_LSE=diam, x=np.array(x), u=u, w_list=w, W=W, specs=specs))    
    return diam


def ellipsoid_volume_diam(Vt ,Bt):
    '''Compute diameter of ellipsoid as defined by Ct in equation (3) in AY11'''
    # diagonalize Vt
    w,eig_vec     = np.linalg.eigh(Vt)
    # print(Bt,w)
    axes_   = np.sqrt(w**-1*Bt)
    diam_   = 2.*np.max(axes_) # axes are radii
    return diam_


def get_Bt(delta,Vt,lambda_,nx,L,S):
    '''Compute beta_t from equation 3 in AY'''
    return (nx*L*np.sqrt( 2*np.log(np.linalg.det(Vt)**0.5 * np.linalg.det(lambda_*np.eye(nx))**0.5 /delta) ) + lambda_**0.5*S)**2



def get_LS_diam(x, u, w, specs, delta=0.1, sigma = 1., savedir='out_unspecified'):
    # Least square confidence bound computed with Lemma E.3 from "Naive Exploration is Optimal for Online LQR" by Simchowitz & Foster 2021
    # delta: confidence so that with 1-delta confidence \theta^* is in the ellipsoid
    # sigma: subGaussian parameter (variance proxy) with sigma_gauss =1; sigma_uniform = 4/3; sigma_beta = 2

    T = specs['T']
    W = specs['W']
    
    # get z[0],...z[T-1]
    if u is not None:
        z = np.hstack([x[:T], u]) # becomes  T by (Nx + Nu)
        Nu = u[0].shape[0]
    else:
        z = x[:T]
    Nx = x.shape[1]
    Nz = z.shape[1]
    
    diam_LS = np.zeros((T+1,)) # diam_LS[t] corresponds to data x[t], z[t-1]
    theta_LS = [None] * (T+1)
    for t in range(Nz,T+1):
        Gamma_t = z[:t].T @ z[:t] # Gamma_t = zt@zt' summing up to t-1
        if np.linalg.matrix_rank(Gamma_t) < Nz:
            print('Data not exciting enough for full rank!')
        theta_LS[t] = x[1:t+1].T @ z[:t] @ np.linalg.pinv(Gamma_t) 
        
        # Compute the least sqaures confidence region diameter
        # Let Gamma_T = U @ S @ U' be the SVD of Gamma_t  
        [U, s, Vh] = np.linalg.svd(Gamma_t)
        
        # p is the subspace dimension
        diam_LS_options = np.zeros((Nz-1,))
        for p in range(1, Nz): # p = 1:Nz-1
            P0 = np.diag(np.hstack([np.ones((p,)), np.zeros((Nz-p,))]))
            lambda_1 = min(s[:p]) # smallest eigenvalue of Gamma_T in the first p coordinate
            lambda_2 = min(s[p:]) # smallest eigenvalue of Gamma_T in the last Nz-p coordinate
            P = U @ P0 @ Vh
            [Q,_] = np.linalg.qr(P)
            
            # generate v_1,..., v_Nz, kappa1, kappa2
            kappa_1_options = np.zeros((p,))
            kappa_2_options = np.zeros((Nz-p,))
            for i in range(p): # i = 0,...p-1
                kappa_1_options[i]= Q[:,i] @ Gamma_t @ Q[:,i] / lambda_1
            for i in range(p, Nz): # i = p,...Nz-1
                kappa_2_options[i-p]= Q[:,i] @ Gamma_t @ Q[:,i] / lambda_2
            kappa_1 = max(kappa_1_options)
            kappa_2 = max(kappa_2_options)
            
            diam_LS_new_sq = 12*Nx*p*kappa_1*np.log(3*Nx*Nz*kappa_1/delta)/lambda_1 + 48*Nx*(Nz-p)*kappa_2*np.log(3*Nx*Nz*kappa_2/delta)/lambda_2;
            diam_LS_options[p-1] = np.sqrt(diam_LS_new_sq)*sigma
        diam_LS[t] = 2 * min(diam_LS_options)
    
    
    # save data       
    if savedir != '' and savedir is not None:
        os.makedirs(savedir, exist_ok=True)
    specs['subGaussian'] = sigma 
    specs['delta'] = delta
    distribution = specs['distribution']
    tz = dt.timezone(dt.timedelta(hours=-8))  # PST
    start_time = dt.datetime.now(tz)
    
    if savedir is not None:
        filename = os.path.join(savedir, f'LSE_diameter_{distribution}_')
        filename += f'Nx={Nx}_'
        if u is not None:
            Nu = u[0].shape[0]
            filename += f'Nu={Nu}_'
        filename += f'T={T}_'
        filename += f'W={W}_'
        filename += f'seed={seed}_'
        filename += f'delta={delta}_'
        filename += f'subGaussian={sigma}_'
        filename += start_time.strftime('%Y%m%d_%H%M%S')
    
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(file=f, obj=dict(diam_LSE=diam_LS, theta_LS=theta_LS, x=np.array(x), u=u, w_list=w, W=W, specs=specs))
    return diam_LS, theta_LS





if __name__ == '__main__':
    savedir = 'ICML_dimension_uniform_eig=0.9'
    T       = 1000 
    W       = 2. 
    distribution = 'uniform'
    variance_scale = compute_uniform_variance(a=-W, b=W)
    sigma = dict(truncated_gaussian=1., uniform=4/3, beta=2. )    
    
    
    for seed in [2023, 202, 1001, 1, 22, 2024, 102, 2002,2, 101 ]:
        for N in [5,10,15,20,25]:
            
            w_bar_list = np.zeros((T+1,))
            w_hat_list = np.zeros((T+1,))
            
            np.random.seed(seed)
            A = stabilize(np.random.rand(N,N),max_eig=0.9)
            Nx = A.shape[1]
            Nz = Nx
            true_theta = dict(A=A)
            run_specs = dict(T=T, W=W, Nx=N, seed=seed, distribution=distribution)
            trajectory = generate_trajectory(true_theta=true_theta, specs=run_specs, savedir=savedir)
        
            SM_traj = get_SM_diam_no_skip(x=trajectory['x'], u=trajectory['u'], w=trajectory['w_list'], specs=trajectory['specs'], savedir=savedir, skip_to_end = True)
            
            LS_traj, LS_theta = get_LS_diam(x=trajectory['x'], u=trajectory['u'], w=trajectory['w_list'], specs=trajectory['specs'], delta=0.1, sigma = sigma[distribution], savedir=None)
            
            LSE_old_diam = get_LSE_diam_old(x=trajectory['x'], u=trajectory['u'],w=trajectory['w_list'], specs=trajectory['specs'], delta=0.1, sigma = sigma[distribution],lam = 0.1, variance_scale = variance_scale, savedir=None)
            
            
            if savedir != '':
                os.makedirs(savedir, exist_ok=True)

            filename = os.path.join(savedir, f'LSE_diameter_{distribution}_')
            filename += f'Nx={Nx}_'
            filename += f'T={T}_'
            filename += f'W={W}_'
            filename += f'seed={seed}_'
            filename += f'subGaussian={sigma}_'
            if savedir is not None:
                with open(f'{filename}.pkl', 'wb') as f:
                    pickle.dump(file=f, obj=dict(diam_LSE=min(LSE_old_diam, LS_traj[-1]), x=trajectory['x'], u=trajectory['u'], w=trajectory['w_list'], specs=trajectory['specs']))
            
            
            
            
            
            
            