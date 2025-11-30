import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, List
import numpy as np
import math
import dask.array as da
from functools import partial

# ============== Optimization Helpers ==============

def _optimal_fft_size(n: int) -> int:
    """Find size with only small prime factors (2, 3, 5) for fast FFT."""
    target = n
    while True:
        m = target
        for p in [2, 3, 5]:
            while m % p == 0:
                m //= p
        if m == 1:
            return target
        target += 1


def _ensure_float32(arr):
    """Ensure array is float32 for GPU efficiency."""
    return jnp.asarray(arr, dtype=jnp.float32)


@jax.jit
def fft(arr:jnp.array) -> jnp.array:
    return jnp.fft.rfftn(arr)

@jax.jit
def fft_mul_conj(A:jnp.array, B:jnp.array) -> jnp.array:
    """
    C = conj(A)*B
    """
    return jnp.conj(A) * B

@jax.jit
def fft_convolve_cc_f2(A:jnp.array, B:jnp.array) -> jnp.array:
    return jnp.fft.irfftn(A*B)

@jax.jit
def fft_convolve_cc(A:jnp.array, B:jnp.array) -> jnp.array:
    M, N, P = A.shape
    C = A*B
    MNP = M*N*P
    return jnp.fft.ifft(C)/MNP

@jax.jit
def fft_convolve_cc_conj_f2(A:jnp.array, B:jnp.array) -> jnp.array:
    return jnp.fft.irfftn(jnp.conj(A)*B)

def get_error(y: jnp.array, g: jnp.array, metric:str='IDIV') -> float:
    if metric == 'MSE':
        error = get_fMSE(y, g)
    elif metric == 'IDIV':
        error = get_fIdiv_jit(y, g)
    return error

def get_fMSE(y:jnp.array, g: jnp.array) -> float:
    """
    Get mean squared error between input y and guess g, 
    on domain of g
    """
    M, N, P = g.shape
    y_subset = y[:M, :N, :P]
    err = jnp.power(y_subset - g,2).sum()
    return err / (M*N*P) # get average error

@jax.jit
def get_fIdiv_jit(y:jnp.array, g:jnp.array) -> float:
    M, N, P = g.shape
    y_subset = y[:M, :N, :P].ravel()
    g_flat = g.ravel()
    #pos_idx = jnp.argwhere((y_subset > 0)*(g_flat>0)).ravel()
    #I = jnp.sum(g_flat[pos_idx]*jnp.log(g_flat[pos_idx]/y_subset[pos_idx]) - (g_flat[pos_idx]-y_subset[pos_idx]))
    I = jnp.sum(g_flat*jnp.log(g_flat/y_subset) - (g_flat-y_subset))
    return I/(M*N*P)

def get_fIdiv(y:jnp.array, g:jnp.array) -> float:
    M, N, P = g.shape
    y_subset = y[:M, :N, :P].ravel()
    g_flat = g.ravel()
    pos_idx = jnp.argwhere((y_subset > 0)*(g_flat>0)).ravel()
    I = jnp.sum(g_flat[pos_idx]*jnp.log(g_flat[pos_idx]/y_subset[pos_idx]) - (g_flat[pos_idx]-y_subset[pos_idx]))
    return I/(M*N*P)

def psf_autocrop(psf:np.array, im:np.array, border_quality:int=1, xycropfactor:float=0.001) -> jnp.array:
    M,N,P = im.shape
    pM, pN, pP = psf.shape
    psf = psf_autocrop_by_image(psf, im)
    # crop the PSF by removing outer planes that have litter information
    # but only if PSF is larger than the image in some dimension
    if border_quality > 0:
        psf = psf_autocrop_xy(psf, xycropfactor=xycropfactor)
    return psf

def psf_autocrop_by_image(psf:np.array, im:np.array, border_quality:int=1) -> jnp.array:
    m, n, p = psf.shape
    M, N, P = im.shape
    if border_quality == 0:
        mopt = M
        nopt = N
        popt = P
    else:
        mopt = (M-1)*2 + 1
        nopt = (N-1)*2 + 1
        popt = (P-1)*2 + 1
    if p < popt:
        print("PSF is smaller than image, no cropping")
        return psf
    if (p%2) == 0:
        # raise error that PSF should have odd number of slices
        return None
    if (m > mopt) or (n > nopt) or (p > popt):
        # initialize with everything set to bound
        m0 = n0 = p0 = 0
        m1, n1, p1 = m-1, n-1, p-1
        if m > mopt:
            m0 = (m-mopt)//2
            m1 -= (m-mopt)//2
        if n > nopt:
            n0 = (n-nopt)//2
            n1 -= (n-nopt)//2
        if p > popt:
            p0 = (p-popt)//2
            p1 -= (p-popt)//2
        psf_cropped = psf[m0:m1+1, n0:n1+1, p0:p1+1]
        print(f"PSF Z-crop: {list(psf.shape)} -> {list(psf_cropped.shape)}")
        return psf_cropped
    # no cropping applied
    return psf

def psf_autocrop_xy(psf:np.array, xycropfactor:float=0.001) -> jnp.array:
    m, n, p = psf.shape
    # find the y-z plane with the largest sum
    #psf_arr = np.array(psf)
    sum_over_plane = np.sum(psf, axis=(1,2))
    maxsum = np.max(sum_over_plane)
    #maxsum = 0
    #for xx in range(m):
    #    sum_val = 0
    #    for yy in range(n):
    #        for zz in range(p):
    #            sum_val += psf_arr[xx, yy, zz]
    #    maxsum = max(sum_val, maxsum)
    first = -1
    sum_val = 0

    while sum_val < xycropfactor * maxsum:
        first += 1
        sum_val = 0
        for yy in range(n):
            for zz in range(p):
                sum_val += psf[first, yy, zz]
    if first < 1:
        print(f"No XY crop, shape is {list(psf.shape)}")
        return psf
    else:
        psf_cropped = psf[first:m-first, first:n-first, :]
        print(f"PSF XY crop: {list(psf.shape)} -> {list(psf_cropped.shape)}")
        return psf_cropped

def psf_autocrop_center_z(psf:jnp.array) -> jnp.array:
    m, n, p = psf.shape
    midm = (m-1)//2
    midn = (n-1)//2
    midp = (p-1)//2
    maxm, maxn, maxp = np.array(max_idx(psf)).astype(int)
    if midp == maxp:
        print(f"No cropping of PSF in Z, size is {list(psf.shape)}")
        return psf
    else:
        m0 = n0 = 0
        m1, n1 = m-1, n-1
        # identify the max number of planes in either direction
        p0 = maxp
        p1 = maxp
        # start at the max plane
        # this will select as many planes as possible in both directions
        # while keeping the stack symmetric in z
        while (p0 > 1) and (p1+2 < p):
        #while (p0 >= 0) and (p1 < p):
            p0 -= 1
            p1 += 1
        print(f"Brightest at plane {maxp}")
        print(f'Selecting z planes at {p0}:{p1}')
        psf_cropped = psf[m0:m1+1, n0:n1+1, p0:p1+1]
        print(f"Cropping PSF in Z: {list(psf.shape)} -> {list(psf_cropped.shape)}")
        return psf_cropped



def initial_guess(M:int, N:int, P:int, wM:int, wN:int, wP:int) -> jnp.array:
    """
    Create initial guess: the fft of an image that is 1 in MNP and 0 outside
     * M, N, P is the dimension of the microscopic image
    """
    one = jnp.zeros((wM, wN, wP))
    one = one.at[:M,:N,:P].set(1)
    return fft(one)

def gaussian_kernel_1d(sigma:float) -> jnp.array:
    n = 1 # guarantee at least 1
    while math.erf((n+1)/sigma) < (1.0-1e-8):
        n += 1
    N = 2*n + 1
    K = jnp.zeros(N)
    mid = int((N-1)/2)
    s2 = sigma**2
    for kk in range(N):
        x = kk-mid
        K[kk] = math.exp(-0.5*(x**2)/s2)
    return K/K.sum()

def gaussian_kernel_3d(sigma_x:float, sigma_y:float, sigma_z:float) -> jnp.array:
    n_x = 1 # guarantee at least 1
    while math.erf((n_x+1)/sigma_x) < (1.0-1e-8):
        n_x += 1
    n_y = 1
    while math.erf((n_y+1)/sigma_y) < (1.0-1e-8):
        n_y += 1
    n_z = 1
    while math.erf((n_z+1)/sigma_z) < (1.0-1e-8):
        n_z += 1
 
    N_x = 2*n_x + 1
    N_y = 2*n_y + 1
    N_z = 2*n_z + 1
    mid_x = float((N_x-1)/2)
    mid_y = float((N_y-1)/2)
    mid_z = float((N_z-1)/2)
    x = jnp.arange(0, n_x).float() - mid_x
    y = jnp.arange(0, n_y).float() - mid_y
    z = jnp.arange(0, n_z).float() - mid_z
    x,y,z = jnp.meshgrid([x,y,z])
    gauss_x = jnp.exp(-0.5 * (x/sigma_x)**2)
    gauss_y = jnp.exp(-0.5 * (y/sigma_y)**2)
    gauss_z = jnp.exp(-0.5 * (z/sigma_z)**2)
    K = gauss_x * gauss_y * gauss_z
    return K/jnp.sum(K)

def max_idx(arr:jnp.array) -> jnp.array:
    return jnp.argwhere(arr == arr.max()).ravel()

def circshift(arr:jnp.array,shifts:jnp.array) -> jnp.array:
    # shift each axis in turn
    #for i in range(len(shifts)):
    #    arr = jnp.roll(arr, int(shifts[i]), axis=i)
    shifts = tuple(shifts)
    return jnp.roll(arr, shifts, (0,1,2))

def insert(T:jnp.array, F:jnp.array) -> jnp.array:
    """
    Insert [f1 x f2 x f3] into T [t1 x t2 x t3] in the 'upper left' corner
    """
    # convert back
    #T = np.array(T)
    #F = np.array(F) 
    if T.ndim == 2:
        F1, F2 = F.shape
        T = T.at[:F1, :F2].set(F)
    elif T.ndim == 3:
        F1, F2, F3 = F.shape
        T = T.at[:F1, :F2, :F3].set(F)
    return T
    #return jax.ops.index_update(T, jax.ops.index[:F.shape[0], :F.shape[1], :F.shape[2]], F)

def gsmooth_aniso(arr:jnp.array, lsigma:float, asigma:float, padding='constant') -> jnp.array:
    M, N, P = arr.shape
    K = gaussian_kernel_3d(lsigma, lsigma, asigma)

    kx, ky, kz = K.shape
    arr_padded = jnp.pad(arr, (kx,kx,ky,ky,kz,kz), mode=padding)
    arr_f = jnp.fft.fftn(arr_padded)
 
    temp = jnp.zeros_like(arr_padded) 
    temp = insert(temp, K)
    maxi = tuple([-int(i) for i in max_idx(temp)])
    #print(maxi)
    
    temp = circshift(temp, maxi)
    k_f = jnp.fft.fftn(temp)
    result = jnp.real(jnp.fft.ifftn(arr_f * k_f))
    return result[kx:(M+kx), ky:(N+ky), kz:(P+kz)]

def gsmooth(arr:jnp.array, gsigma:float) -> jnp.array:
    """
    Gaussian smooth array
    """
    return gsmooth_aniso(arr, gsigma, gsigma)

def get_midpoint(arr:np.array) -> Tuple[int, int, int]:
    m,n,p = arr.shape
    return int((m-1)/2), int((n-1)/2), int((p-1)/2)

def prefilter(im:jnp.array, psf:jnp.array, psigma:float=0) -> Tuple[jnp.array, jnp.array]:
    if psigma <= 0:
        return im, psf
    else:
        return gsmooth(im, psigma), gsmooth(psf, psigma)

def compute_tile_positions(im:jnp.array, max_size:int, overlap:int) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                                                            List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Return a list of the x,y indices of the tiles
    """
    M,N,P = im.shape
    tile_pos_with_overlap = []
    tile_pos_without_overlap = []
    n_tiles_x = np.ceil(float(M)/float(max_size)).astype(int)
    n_tiles_y = np.ceil(float(N)/float(max_size)).astype(int)
    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            tile_start_x = max(0, i*max_size-overlap)
            tile_stop_x = min((i+1)*max_size+overlap, M)
            tile_start_y = max(0, j*max_size-overlap)
            tile_stop_y = min((j+1)*max_size+overlap, N)
            tile_pos_with_overlap.append(((tile_start_x, tile_stop_x), (tile_start_y, tile_stop_y)))

            tile_start_x = max(0, i*max_size)
            tile_stop_x = min((i+1)*max_size, M)
            tile_start_y = max(0, j*max_size)
            tile_stop_y = min((j+1)*max_size, N)
            tile_pos_without_overlap.append(((tile_start_x, tile_stop_x), (tile_start_y, tile_stop_y)))
    return tile_pos_with_overlap, tile_pos_without_overlap

def run_dw_tiled_dask(im:np.ndarray, psf:np.ndarray, 
    tile_factor:int=4,
    n_iter:int=10, alphamax:float=10, bg:Optional[float]=None, 
    relax:int=0, psigma:int=0, border_quality:int=1,
    positivity:bool=True,method:str='shb', err_thresh:Optional[float]=None) -> np.ndarray:
    """
    Run tiled deconvolution on an image using single-threaded JAX functions, parallelized with Dask. 
    Each block is processed independently.
    im is a Nx x Ny x Nz numpy array
    psf is a Nx x Ny x Nz numpy array 
    """
    M, N, P = im.shape
    tile_max_size = M // tile_factor
    im_dask = da.from_array(jnp.array(im), chunks=(tile_max_size, tile_max_size, P))
    psf = jnp.array(psf)

    if im.min() < 0:
        im -= im.min()
    if im.max() < 1000:
        im *= 1000/im.max()

    # normalize PSF
    psf /= psf.sum()
    psf = psf_autocrop(psf, im)

    if relax > 0:
        mid_x, mid_y, mid_z = get_midpoint(psf)
        psf = psf.at[mid_x, mid_y, mid_z].add(relax)
        psf /= psf.sum()

    tile_padding = psf.shape[0] // 2
    # split image into N tiles
       #print(curr_tile.shape, psf.shape)
    decon_fn = partial(decon, psf=psf, psigma=psigma, n_iter=n_iter, alphamax=alphamax, bg=bg, border_quality=border_quality, 
                    positivity=positivity, method=method, err_thresh=err_thresh)
    decon_img = im_dask.map_overlap(decon_fn, depth=(tile_padding, tile_padding, 0), boundary='reflect', meta=im)
    return np.array(decon_img.compute())

def run_dw_tiled(im:jnp.array, psf:jnp.array, 
    tile_max_size:int=256, tile_padding:int=40,
    n_iter:int=10, alphamax:float=10, bg:Optional[float]=None, 
    relax:int=0, psigma:int=0, border_quality:int=1,
    positivity:bool=True,method:str='shb' ) -> jnp.array:
    """
    Run DW on tiles independently, without parallelization. 
    """
    M, N, P = im.shape

    if im.min() < 0:
        im -= im.min()
    if im.max() < 1000:
        im *= 1000/im.max()

    # normalize PSF
    psf /= psf.sum()
    psf = psf_autocrop(psf, im)

    if relax > 0:
        mid_x, mid_y, mid_z = get_midpoint(psf)
        psf = psf.at[mid_x, mid_y, mid_z].add(relax)
        psf /= psf.sum()

    # split image into N tiles
    pos_with_overlap, pos_without_overlap = compute_tile_positions(im, tile_max_size, tile_padding) 
    decon_img = jnp.zeros_like(im)
    for i in range(len(pos_with_overlap)):
        # get tile with padding
        (min_x_overlap, max_x_overlap), (min_y_overlap, max_y_overlap) = pos_with_overlap[i]
        #print(min_x_overlap, max_x_overlap, min_y_overlap, max_y_overlap)
        curr_tile = im[min_x_overlap:max_x_overlap, min_y_overlap:max_y_overlap, :]
        # run decon on tile with padding
        #print(curr_tile.shape, psf.shape)
        res = decon(curr_tile, psf, psigma=psigma, n_iter=n_iter, alphamax=alphamax, bg=bg, border_quality=border_quality, 
                    positivity=positivity, method=method)
        # get rid of padding
        (min_x, max_x), (min_y, max_y) = pos_without_overlap[i]
        if min_x == 0:
            crop_x_min = 0
        else:
            crop_x_min = tile_padding
        if max_x == M:
            crop_x_max = res.shape[0]
        else:
            crop_x_max = res.shape[0]-tile_padding
        if min_y == 0:
            crop_y_min = 0
        else:
            crop_y_min = tile_padding
        if max_y == N:
            crop_y_max = res.shape[1]
        else:
            crop_y_max = res.shape[1]-tile_padding
        res_cropped = res[crop_x_min:crop_x_max, crop_y_min:crop_y_max, :]
        #print(res.shape, res_cropped.shape, tile_padding, min_x, max_x, min_y, max_y)
        decon_img = decon_img.at[min_x:max_x, min_y:max_y, :].set(res_cropped)
    return decon_img

def run_dw(im:np.array, psf:np.array,
    n_iter:int=10, alphamax:float=10, bg:Optional[float]=None,
    relax:int=0, psigma:int=0, border_quality:int=1,
    positivity:bool=True,method:str='shb_jit',verbose:bool=True, err_thresh:Optional[float]=0.01,
    optimize_fft_size:bool=True) -> jnp.array:
    M, N, P = im.shape

    if im.min() < 0:
        im -= im.min()
    if im.max() < 1000:
        im *= 1000/im.max()
        
    # normalize PSF
    psf /= psf.sum()
    psf = psf_autocrop(psf, im)

    if relax > 0:
        mid_x, mid_y, mid_z = get_midpoint(psf)
        psf[mid_x, mid_y, mid_z] += relax
        psf /= psf.sum()
    
    psf /= psf.sum()
    im = jnp.array(im)
    psf = jnp.array(psf)
    im, psf = prefilter(im, psf, psigma) 
    return decon(im, psf, psigma, n_iter, alphamax, bg, border_quality, positivity, method, verbose=verbose, err_thresh=err_thresh, optimize_fft_size=optimize_fft_size)

def decon(im:jnp.array, psf:jnp.array, psigma:int=3, n_iter:int=10, alphamax:float=10,
          bg:Optional[float]=None, border_quality:int=1, positivity:bool=True, method:str='shb_jit',err_thresh:Optional[float]=None,
          verbose:bool=False, optimize_fft_size:bool=True) -> jnp.array:
    # auto compute background
    if bg is None:
        bg = im.min()
        if bg < 1e-2:
            bg = 1e-2

    M, N, P = im.shape
    pM, pN, pP = psf.shape 
    wM = M + pM - 1
    wN = N + pN - 1
    wP = P + pP - 1

    if border_quality == 1:
        wM = M + (pM + 1)//2
        wN = N + (pN + 1)//2
        wP = P + (pP + 1)//2

    elif border_quality == 0:
        wM = max(M, pM)
        wN = max(N, pN)
        wP = max(P, pP)

    # Optimize FFT sizes for faster computation (use only small prime factors 2,3,5)
    if optimize_fft_size:
        wM = _optimal_fft_size(int(wM))
        wN = _optimal_fft_size(int(wN))
        wP = _optimal_fft_size(int(wP))

    Z = jnp.zeros((wM, wN, wP))

    # insert the PSF into the larger image
    Z = insert(Z, psf)

    # shift PSF so midpoint is at (0,0,0)
    Z = circshift(Z, -max_idx(Z))

    # PSF FFT
    cK = fft(Z)
    del Z
    sigma = 0.01
    if border_quality > 0:
        F_one = initial_guess(M, N, P, wM, wN, wP)

        W = jnp.fft.irfftn(fft_mul_conj(cK, F_one))
        #W = jnp.where(W > sigma, 1/W, 0)
        idx = W>sigma
        #
        W = W.at[idx].divide(W[idx])  # W[idx] = 1/W[idx]
        W = W.at[~idx].set(0)
    else:
        # No border correction weights when border_quality == 0
        W = jnp.ones((wM, wN, wP), dtype=im.dtype)

    sumg = im.sum() 
    # x is initial guess, initially previous iteration xp is set to be the same
    x = jnp.ones((wM, wN, wP)) * sumg/(wM*wN*wP)
    xp = x
    prev_err = 1e6
    for i in range(n_iter):
        if method == 'shb':
            # Eq. 10 in SHB paper
            alpha = (i-1.0)/(i+2.0)
            if alpha < 0:
                alpha = 0
            if alpha > alphamax:
                alpha = alphamax
            
            p = x + alpha*(x - xp) 
            p = p.at[jnp.where(p < bg)].set(bg)

            # optionally smooth image
            if psigma > 0:
                x = gsmooth(x, psigma)

            xp_temp, err = iter_shb(im, cK, p, W)
            # swap to update
            xp = x
            x = xp_temp

        elif method == 'shb_jit':
            # Eq. 10 in SHB paper
            alpha = (i-1.0)/(i+2.0)
            if alpha < 0:
                alpha = 0
            if alpha > alphamax:
                alpha = alphamax
            
            # optionally smooth image
            if psigma > 0:
                x = gsmooth(x, psigma)

            xp_temp, err = iter_shb_jit(im, cK, x, xp, W, bg, alpha)
            # swap to update
            xp = x
            x = xp_temp

        elif method == 'rl':
            x, err = iter_rl(im, cK, xp, bg, W)
            xp = x
        if verbose: 
            print(f"Iter: {i}, Err: {float(err):2.2f}, Delta: {float(prev_err-err):2.2f}")

        if err_thresh is not None:
            if prev_err-err < err_thresh:
                break

        prev_err = err
        if positivity and bg > 0:
            # this isn't in RL function
            x = jnp.where(x < bg, bg, x) #x.at[x < bg].set(bg)

    x = xp
    # crop to corner subregion
    return x[:M,:N,:P]

@jax.jit
def iter_rl(im: jnp.array, fftPSF:jnp.array, f:jnp.array, bg:float, W:Optional[jnp.array]=None) -> jnp.array:
    M, N, P = im.shape
    wM, wN, wP = f.shape
    F = fft(f)
    y = jnp.fft.irfftn(fftPSF * F)#fft_convolve_cc_f2(fftPSF, F)
    error = get_error(y, im)
    # crop down to size of image
    y_subset = y[:M,:N,:P]
    y_subset = jnp.where(y_subset > 0, im/y_subset, bg)
    #idx = y_subset > 0
    #y_subset = y_subset.at[idx].set(im[idx]/y_subset[idx])
    #y_subset = y_subset.at[~idx].set(bg)
    # set everything outside of image to 1e-6
    #y[M:, N:, P:] = 1e-6
    y = jnp.ones_like(y)*1e-6
    # set everything within image
    y = y.at[:M, :N, :P].set(y_subset)
    # convolve with PSF for next iteration
    F_sn = fft(y)
    x = jnp.fft.irfftn(jnp.conj(fftPSF) * F_sn)#fft_convolve_cc_conj_f2(fftPSF, F_sn)
    #if W is None:
    #    x *= f
    #else:
    x *= f * W
    return x, error

@jax.jit
def iter_shb_jit(im:jnp.array, cK:jnp.array, x: jnp.array, xp: jnp.array, W:jnp.array, bg:float, alpha:float) -> Tuple[jnp.array, float]:
    """Iteration of SHB

    Args:
        im (jnp.array): Input image
        cK (jnp.array): fft(psf) 
        x (jnp.array): initial guess
        xp (jnp.array): current guess
        alpha (float): step size
        bg (jnp.array): background value
        W (jnp.array): Bertero weights

    Returns:
        jnp.array: _description_
    """
    M, N, P = im.shape
    # current guess, based on update from previous
    # p^k in Eq. 7 o SHB paper
    # estimate gradient based on scaled difference from previous round
    p = x + (x - xp)*alpha
    # set pixels less than background to background
    pK = jnp.where(p < bg, bg, p) #p.at[jnp.where(p < bg)].set(bg)
    pK_F = fft(pK)
    # convolve with PSF
    y = fft_convolve_cc_f2(cK, pK_F)
    error = get_error(y, im)
    mindiv = 1e-6 # smallest allowed divisor
    y_subset = y[:M, :N, :P]
    y_subset  = jnp.where(jnp.abs(y_subset) < mindiv, jnp.copysign(mindiv, y_subset), y_subset)
    #y_subset = y_subset.at[jnp.where(jnp.abs(y_subset) < mindiv)].set(jnp.sign(y_subset[jnp.abs(y_subset) < mindiv])*mindiv)
    y_subset = im/y_subset
    y = jnp.zeros_like(y)
    y = y.at[:M, :N, :P].set(y_subset)
    #y[M:, N:, P:] = 0
    #else:
    #    x *= pK
    x = update_x_shb(y, cK, pK, W)
    return x, error

@jax.jit
def iter_shb(im:jnp.array, cK:jnp.array, pK: jnp.array, W:jnp.array ) -> Tuple[jnp.array, float]:
    """Iteration of SHB

    Args:
        im (jnp.array): Input image
        cK (jnp.array): fft(psf) 
        x (jnp.array): initial guess
        xp (jnp.array): current guess
        alpha (float): step size
        bg (jnp.array): background value
        W (jnp.array): Bertero weights

    Returns:
        jnp.array: _description_
    """
    M, N, P = im.shape
    # current guess, based on update from previous
    # p^k in Eq. 7 o SHB paper
    # estimate gradient based on scaled difference from previous round

    # set pixels less than background to background
    #pK = jnp.where(p < bg, bg, p) #p.at[jnp.where(p < bg)].set(bg)
    pK_F = fft(pK)
    # convolve with PSF
    y = fft_convolve_cc_f2(cK, pK_F)
    error = get_error(y, im)
    mindiv = 1e-6 # smallest allowed divisor
    y_subset = y[:M, :N, :P]
    y_subset  = jnp.where(jnp.abs(y_subset) < mindiv, jnp.copysign(mindiv, y_subset), y_subset)
    #y_subset = y_subset.at[jnp.where(jnp.abs(y_subset) < mindiv)].set(jnp.sign(y_subset[jnp.abs(y_subset) < mindiv])*mindiv)
    y_subset = im/y_subset
    y = jnp.zeros_like(y)
    y = y.at[:M, :N, :P].set(y_subset)
    #y[M:, N:, P:] = 0
    #else:
    #    x *= pK
    x = update_x_shb(y, cK, pK, W)
    return x, error

@jax.jit
def update_x_shb(y:jnp.array, cK:jnp.array, pK:jnp.array, W:jnp.array) -> jnp.array:
    Y = fft(y)
    # convolve with PSF
    x = fft_convolve_cc_conj_f2(cK, Y)
    #if W is not None:
    x *= pK * W
    return x


# ============== Optimized JAX Implementations ==============

def _make_decon_fori(M: int, N: int, P: int, wM: int, wN: int, wP: int):
    """
    Factory function to create a JIT-compiled deconvolution loop with fixed dimensions.

    This approach uses closure over static dimensions to avoid dynamic indexing issues.
    """
    wshape = (wM, wN, wP)
    mindiv = 1e-6

    @jax.jit
    def _decon_fori_inner(im: jnp.ndarray, cK: jnp.ndarray, x_init: jnp.ndarray,
                          W: jnp.ndarray, bg: float, n_iter: int, alphamax: float) -> jnp.ndarray:
        """
        Fully JIT-compiled SHB deconvolution loop using jax.lax.fori_loop.

        This eliminates Python loop overhead by compiling the entire iteration
        sequence into a single XLA program.
        """

        def body_fn(i, carry):
            x, xp = carry

            # Compute momentum coefficient (Eq. 10 in SHB paper)
            i_f = i.astype(jnp.float32)
            alpha = jnp.clip((i_f - 1.0) / (i_f + 2.0), 0.0, alphamax)

            # Momentum estimate
            p = x + alpha * (x - xp)
            p = jnp.maximum(p, bg)

            # Forward model: convolve with PSF
            pK_F = jnp.fft.rfftn(p, s=wshape)
            y = jnp.fft.irfftn(cK * pK_F, s=wshape)

            # Compute ratio in image domain (use static slice)
            y_obs = lax.dynamic_slice(y, (0, 0, 0), (M, N, P))
            y_safe = jnp.where(jnp.abs(y_obs) < mindiv,
                              jnp.copysign(mindiv, y_obs), y_obs)
            ratio = im / y_safe

            # Embed ratio in work volume
            y_full = jnp.zeros(wshape, dtype=im.dtype)
            y_full = lax.dynamic_update_slice(y_full, ratio, (0, 0, 0))

            # Back-convolve with conjugate PSF
            Y = jnp.fft.rfftn(y_full, s=wshape)
            x_new = jnp.fft.irfftn(jnp.conj(cK) * Y, s=wshape)

            # Apply weights
            x_new = x_new * p * W

            # Positivity constraint
            x_new = jnp.maximum(x_new, bg)

            return (x_new, x)

        init_carry = (x_init, x_init)
        final_x, final_xp = lax.fori_loop(0, n_iter, body_fn, init_carry)

        # IMPORTANT: Original decon() returns xp (second-to-last iteration result)
        # at line 559: "x = xp"
        return final_xp

    return _decon_fori_inner


# Cache for compiled deconvolution functions
_decon_fori_cache = {}


def decon_fast(im: jnp.ndarray, psf: jnp.ndarray, psigma: int = 3,
               n_iter: int = 10, alphamax: float = 10.0,
               bg: Optional[float] = None, border_quality: int = 1,
               positivity: bool = True) -> jnp.ndarray:
    """
    Fast SHB deconvolution using jax.lax.fori_loop.

    This is a drop-in replacement for decon() that compiles the entire
    iteration loop into a single XLA program, eliminating Python overhead.

    The setup (PSF preparation, weights, etc.) is identical to decon() to
    ensure numerical parity. Only the iteration loop is optimized.

    Args:
        im: Input image (M, N, P)
        psf: Point spread function (already normalized)
        psigma: Smoothing parameter (must be 0 for fori_loop version)
        n_iter: Number of iterations
        alphamax: Maximum momentum parameter for SHB
        bg: Background value. If None, computed from image minimum.
        border_quality: Border handling (0=none, 1=half, 2=full)
        positivity: Whether to enforce positivity constraint

    Returns:
        Deconvolved image (M, N, P)
    """
    if psigma > 0:
        raise ValueError("decon_fast does not support psigma > 0. Use decon() instead.")

    # Auto compute background (same as decon)
    if bg is None:
        bg = im.min()
        if bg < 1e-2:
            bg = 1e-2
    bg = float(bg)

    M, N, P = im.shape
    pM, pN, pP = psf.shape

    # Compute work shape (EXACTLY as in decon)
    wM = M + pM - 1
    wN = N + pN - 1
    wP = P + pP - 1

    if border_quality == 1:
        wM = int(M + (pM + 1) / 2)
        wN = int(N + (pN + 1) / 2)
        wP = int(P + (pP + 1) / 2)
    elif border_quality == 0:
        wM = max(M, pM)
        wN = max(N, pN)
        wP = max(P, pP)

    # Convert to int for JAX
    wM, wN, wP = int(wM), int(wN), int(wP)
    wshape = (wM, wN, wP)

    # Prepare PSF (EXACTLY as in decon using insert and circshift)
    Z = jnp.zeros(wshape)
    Z = insert(Z, psf)
    Z = circshift(Z, -max_idx(Z))

    # PSF FFT
    cK = fft(Z)

    # Compute Bertero weights (EXACTLY as in decon)
    sigma = 0.01
    if border_quality > 0:
        F_one = initial_guess(M, N, P, wM, wN, wP)
        W = jnp.fft.irfftn(fft_mul_conj(cK, F_one))
        idx = W > sigma
        W = W.at[idx].divide(W[idx])
        W = W.at[~idx].set(0)
    else:
        W = jnp.ones(wshape)

    # Initial guess (EXACTLY as in decon)
    sumg = im.sum()
    x_init = jnp.ones(wshape) * sumg / (wM * wN * wP)

    # Get or create compiled function for these dimensions
    cache_key = (M, N, P, wM, wN, wP)
    if cache_key not in _decon_fori_cache:
        _decon_fori_cache[cache_key] = _make_decon_fori(M, N, P, wM, wN, wP)

    _decon_fori_fn = _decon_fori_cache[cache_key]

    # Run optimized deconvolution
    x = _decon_fori_fn(im, cK, x_init, W, bg, n_iter, alphamax)

    # Crop to original size (same as decon)
    return x[:M, :N, :P]


# ============== Batch Processing with vmap ==============

def batch_deconvolve(images: jnp.ndarray, psf: jnp.ndarray,
                     n_iter: int = 10, alphamax: float = 10.0,
                     bg: Optional[float] = None, border_quality: int = 1) -> jnp.ndarray:
    """
    Vectorized deconvolution of multiple images with the same PSF.

    Uses jax.vmap to process multiple images in parallel, achieving
    significant speedup on GPU when processing batches.

    NOTE: PSF should already be preprocessed (normalized, autocropped) before
    calling this function.

    Args:
        images: Stack of 3D images, shape (batch_size, M, N, P)
        psf: Point spread function (shared across all images, already normalized)
        n_iter: Number of iterations
        alphamax: Maximum momentum parameter for SHB
        bg: Background value. If None, computed per-image from minimum.
        border_quality: Border handling (0=none, 1=half, 2=full)

    Returns:
        Deconvolved images, shape (batch_size, M, N, P)
    """
    batch_size, M, N, P = images.shape
    pM, pN, pP = psf.shape

    # Compute work shape (EXACTLY as in decon)
    wM = M + pM - 1
    wN = N + pN - 1
    wP = P + pP - 1

    if border_quality == 1:
        wM = int(M + (pM + 1) / 2)
        wN = int(N + (pN + 1) / 2)
        wP = int(P + (pP + 1) / 2)
    elif border_quality == 0:
        wM = max(M, pM)
        wN = max(N, pN)
        wP = max(P, pP)

    wM, wN, wP = int(wM), int(wN), int(wP)
    wshape = (wM, wN, wP)

    # Prepare PSF (EXACTLY as in decon using insert and circshift)
    Z = jnp.zeros(wshape)
    Z = insert(Z, psf)
    Z = circshift(Z, -max_idx(Z))

    # PSF FFT
    cK = fft(Z)

    # Compute Bertero weights (EXACTLY as in decon)
    sigma = 0.01
    if border_quality > 0:
        F_one = initial_guess(M, N, P, wM, wN, wP)
        W = jnp.fft.irfftn(fft_mul_conj(cK, F_one))
        idx = W > sigma
        W = W.at[idx].divide(W[idx])
        W = W.at[~idx].set(0)
    else:
        W = jnp.ones(wshape)

    # Compute background per image if not provided
    if bg is None:
        bg_arr = jnp.maximum(images.min(axis=(1, 2, 3)), 1e-2)
    else:
        bg_arr = jnp.full(batch_size, float(bg))

    # Create vmappable deconvolution function with all constants captured
    mindiv = 1e-6

    def single_decon(im_single, bg_val):
        """Deconvolve a single image - vmappable."""

        def body_fn(i, carry):
            x, xp = carry

            # Compute momentum coefficient (EXACTLY as in decon shb_jit path)
            i_f = i.astype(jnp.float32)
            alpha = jnp.clip((i_f - 1.0) / (i_f + 2.0), 0.0, alphamax)

            # Momentum estimate
            p = x + alpha * (x - xp)
            p = jnp.maximum(p, bg_val)

            # Forward model
            pK_F = jnp.fft.rfftn(p, s=wshape)
            y = jnp.fft.irfftn(cK * pK_F, s=wshape)

            # Compute ratio
            y_obs = lax.dynamic_slice(y, (0, 0, 0), (M, N, P))
            y_safe = jnp.where(jnp.abs(y_obs) < mindiv,
                              jnp.copysign(mindiv, y_obs), y_obs)
            ratio_slice = im_single / y_safe

            # Embed ratio in work volume
            y_full = jnp.zeros(wshape, dtype=im_single.dtype)
            y_full = lax.dynamic_update_slice(y_full, ratio_slice, (0, 0, 0))

            # Back-convolve
            Y = jnp.fft.rfftn(y_full, s=wshape)
            x_new = jnp.fft.irfftn(jnp.conj(cK) * Y, s=wshape)
            x_new = x_new * p * W

            # Positivity
            x_new = jnp.maximum(x_new, bg_val)

            return (x_new, x)

        sumg = im_single.sum()
        x_init = jnp.ones(wshape) * sumg / (wM * wN * wP)
        init_carry = (x_init, x_init)
        final_x, final_xp = lax.fori_loop(0, n_iter, body_fn, init_carry)
        # Return xp to match original decon() behavior
        return final_xp[:M, :N, :P]

    # Apply vmap over batch dimension
    batched_decon = jax.jit(jax.vmap(single_decon, in_axes=(0, 0)))

    return batched_decon(images, bg_arr)


# ============== Parallel Tiled Deconvolution ==============

def run_dw_tiled_parallel(im: jnp.ndarray, psf: jnp.ndarray,
                          tile_max_size: int = 256, tile_padding: int = 40,
                          n_iter: int = 10, alphamax: float = 10.0,
                          bg: Optional[float] = None, border_quality: int = 1,
                          relax: int = 0, psigma: int = 0,
                          positivity: bool = True) -> jnp.ndarray:
    """
    Parallel tiled deconvolution using vmap for efficient processing of large images.

    This function splits a large image into tiles, processes them all in parallel
    using batch_deconvolve with vmap, then reassembles the result. This is much
    faster than sequential tiled processing for large images.

    Args:
        im: Input 3D image (M, N, P)
        psf: Point spread function
        tile_max_size: Maximum tile size in X and Y
        tile_padding: Overlap padding to avoid edge artifacts
        n_iter: Number of iterations
        alphamax: Maximum momentum for SHB
        bg: Background value (None = auto-detect)
        border_quality: Border handling quality (0, 1, or 2)
        relax: Relaxation parameter for PSF
        psigma: Gaussian smoothing sigma for PSF
        positivity: Enforce positivity constraint

    Returns:
        Deconvolved image (M, N, P)
    """
    M, N, P = im.shape

    # Preprocess image (same as run_dw_tiled)
    if im.min() < 0:
        im = im - im.min()
    if im.max() < 1000:
        im = im * (1000 / im.max())

    # Normalize and preprocess PSF
    psf = psf / psf.sum()
    psf = psf_autocrop(psf, im)

    if relax > 0:
        mid_x, mid_y, mid_z = get_midpoint(psf)
        psf = psf.at[mid_x, mid_y, mid_z].add(relax)
        psf = psf / psf.sum()

    # Apply prefilter if requested
    im, psf = prefilter(im, psf, psigma)

    # Compute tile positions
    pos_with_overlap, pos_without_overlap = compute_tile_positions(im, tile_max_size, tile_padding)
    n_tiles = len(pos_with_overlap)

    if n_tiles == 1:
        # Single tile - just use regular decon
        return decon(im, psf, psigma=0, n_iter=n_iter, alphamax=alphamax,
                    bg=bg, border_quality=border_quality, positivity=positivity,
                    method='shb_jit', verbose=False)

    # Extract all tiles and find maximum tile dimensions
    tiles = []
    tile_shapes = []
    for i in range(n_tiles):
        (min_x_overlap, max_x_overlap), (min_y_overlap, max_y_overlap) = pos_with_overlap[i]
        tile = im[min_x_overlap:max_x_overlap, min_y_overlap:max_y_overlap, :]
        tiles.append(tile)
        tile_shapes.append(tile.shape)

    # Find maximum tile size for padding
    max_tile_x = max(s[0] for s in tile_shapes)
    max_tile_y = max(s[1] for s in tile_shapes)

    # Pad all tiles to uniform size
    padded_tiles = []
    for tile in tiles:
        pad_x = max_tile_x - tile.shape[0]
        pad_y = max_tile_y - tile.shape[1]
        if pad_x > 0 or pad_y > 0:
            padded = jnp.pad(tile, ((0, pad_x), (0, pad_y), (0, 0)), mode='reflect')
        else:
            padded = tile
        padded_tiles.append(padded)

    # Stack into batch
    tile_batch = jnp.stack(padded_tiles, axis=0)  # (n_tiles, max_tile_x, max_tile_y, P)

    # Batch deconvolve all tiles in parallel
    decon_batch = batch_deconvolve(
        tile_batch, psf,
        n_iter=n_iter, alphamax=alphamax,
        bg=bg, border_quality=border_quality
    )

    # Reassemble tiles into output image
    decon_img = jnp.zeros_like(im)

    for i in range(n_tiles):
        # Get the original tile shape
        orig_x, orig_y, orig_z = tile_shapes[i]

        # Crop deconvolved tile to original size (remove padding)
        res = decon_batch[i, :orig_x, :orig_y, :]

        # Compute crop boundaries for overlap removal
        (min_x_overlap, max_x_overlap), (min_y_overlap, max_y_overlap) = pos_with_overlap[i]
        (min_x, max_x), (min_y, max_y) = pos_without_overlap[i]

        if min_x == 0:
            crop_x_min = 0
        else:
            crop_x_min = tile_padding
        if max_x == M:
            crop_x_max = res.shape[0]
        else:
            crop_x_max = res.shape[0] - tile_padding
        if min_y == 0:
            crop_y_min = 0
        else:
            crop_y_min = tile_padding
        if max_y == N:
            crop_y_max = res.shape[1]
        else:
            crop_y_max = res.shape[1] - tile_padding

        res_cropped = res[crop_x_min:crop_x_max, crop_y_min:crop_y_max, :]
        decon_img = decon_img.at[min_x:max_x, min_y:max_y, :].set(res_cropped)

    return decon_img
