import numpy as np
from scipy import linalg


from winnie.plot import animate_quick_implot, quick_implot, mpl, plt
from winnie.utils import (
    dist_to_pt,
    ang_size_to_px_size,
    ang_size_to_proj_sep,
    px_size_to_ang_size,
    median_filter_sequence
    )
import vip_hci as vip
from vip_hci.fm import ScatteredLightDisk, Phase_function
from astropy import convolution
import stpsf
from IPython.display import clear_output
import astropy.units as u

# Module-level counter used by obj_fn's progress display (see 'count' parameter).
# 'global counter' inside obj_fn binds to this module's namespace, not the caller's,
# so setting 'counter = 0' in a notebook has no effect here; use 'ffm.counter = 0' instead.
counter = 0


## FM functions adapted so rings share same inc, PA and scattering parameters (but not necessarily the same r0, h0, ain, aout, or gamma)

# Now, let's a) generalize the GRaTer model to a composite image formed from the super-position of N individual ring-like components, and b) set up an objective function to optimize a disk model for the data.

# (we don't actually need more than 1 ring here — I just had the code already and thought it might be helpful for other applications)

def grater_2hg_disk_model(r0, h0, ain, aout, pa, incl, g1, g2, wg1,
                          e=0., omega=0., gamma=2., beta=1., xdo=0, ydo=0,
                          cent=np.array([160.,160.]), distance=10., nx=320, ny=320,
                          pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                          halfNbSlices=25, polar=False, flux_max=1.0):
    """
    A simple ring-like disk morphology based on Augereau et al. (1999) and assuming a linear combo of two H-G SPFs as the scattering phase function.

    r0: fiducial radius in au
    h0: technically h0/r0 — the ratio of scale height to fiducial radius at the fiducial radius
    ain: radial density power law exponent interior to r0
    aout: radial density power law exponent exterior to r0
    pa: disk position angle in degrees
    incl: disk inclination in degrees
    g1: 1st Henyey-Greenstein asymmetry parameter
    g2: 2nd Henyey-Greenstein asymmetry parameter
    wg1: Weight for the SPF term with asymmetry parameter g1 (value in range 0-1); wg2 is 1-wg1
    e: eccentricity
    omega: argument of pericenter in degrees
    gamma: vertical density exponent (gamma = 2 for gaussian)
    beta: disk radial flaring exponent (beta = 1 for linear)

    cent: the pixel position for the center of the disk (generally the location of the star in the data)
    distance: distance to the target in parsecs
    nx: number of x-axis pixels for the image
    ny: number of y-axis pixels for the image
    pxscale: the pixel scale for the data; either a float (must be arcsec/pixel) or astropy units (any units that can be cast to arcsec/pixel)
    accuracy: the numerical accuracy for the model; pixels with density below this value will be set to zero
    rmax_accuracy: if accuracy is None, the model's accuracy is set such that non-zero values are achieved to this separation (in au)
    halfNbSlices: the number of planar slices to compute above and below the disk midplane
    polar: if True, a simple bell-shaped polarization curve is used to generate a polarized intensity image
    flux_max: if not None, normalize the model image so that this is the maximum value.
    """

    def dont_actually_check_inclination():
        """
        VIP's ScatteredLightDisk will normally decrease disk inclination if too high.
        This will cause issues for optimization, so we'd rather live with the artifacts / etc.
        Instead, we just tell VIP that the incl is good whenever it checks.
        """
        return None

    if accuracy is None:
        if rmax_accuracy is None:
            im_corner_dists = np.hypot(*(np.array([[0, 0], [0, ny-1], [nx-1, 0], [nx-1, ny-1]])-cent).T)
            rmax_accuracy = ang_size_to_proj_sep(px_size_to_ang_size(np.max(im_corner_dists), pxscale), distance).value
        accuracy = (rmax_accuracy/r0)**(aout)

    spf = {'name':'DoubleHG', 'g': [g1, g2], 'weight': wg1, 'polar': polar}
    dens = {'name': '2PowerLaws', 'ain': ain, 'aout': aout, 'a': r0, 'ksi0':h0*r0,
            'e': e, 'gamma': gamma, 'beta': beta, 'accuracy': accuracy}

    vip_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=(pxscale << u.arcsec/u.pixel).value,
                                  pa=pa-180., omega=omega, xdo=xdo, ydo=ydo,
                                  density_dico=dens, spf_dico=spf, flux_max=flux_max,
                                  xs=cent[0], ys=cent[1])

    vip_disk.dust_density.accuracy = accuracy
    vip_disk.dust_density.dust_distribution_calc.rmax = vip_disk.dust_density.dust_distribution_calc.a*accuracy**(1/vip_disk.dust_density.dust_distribution_calc.aout)
    vip_disk.check_inclination = dont_actually_check_inclination
    disk = vip_disk.compute_scattered_light(halfNbSlices=halfNbSlices)
    return fix_star_cross(disk, cent)


class DustEllipticalDistributionGaussian:
    """
    Gaussian ring dust density distribution, written as a drop-in replacement for VIP's
    DustEllipticalDistribution2PowerLaws (VIP's ScatteredLightDisk only uses density_cylindrical,
    rmax, zmax and itiltthreshold from it).

    The radial profile is a Gaussian in semi-major axis, a_eff = r*(1-e*cos(theta))/(1-e^2), centred on r0
    with the given FWHM. The vertical profile is the same as VIP's: exp(-(|z|/H)^gamma), with
    H = ksi0*(a_eff/r0)^beta.

    If surface_density is True (default), the Gaussian describes the surface density, i.e. the midplane
    density is divided by H(r)/ksi0 so that Sigma(r) ∝ rho_mid(r)*H(r) is Gaussian. If False, the
    midplane volume density is Gaussian (and Sigma(r) is a Gaussian times (r/r0)^beta).
    """

    def __init__(self, accuracy=1.e-3, density_dico={}):
        self.accuracy = accuracy
        self.set_density_distribution(density_dico)

    def set_density_distribution(self, density_dico):
        self.a = float(density_dico.get('a', 60.))
        self.fwhm = float(density_dico.get('fwhm', 10.))
        self.sigma = self.fwhm/(2.*np.sqrt(2.*np.log(2.)))
        self.e = float(np.clip(density_dico.get('e', 0.), 0., 0.99))
        self.ksi0 = float(density_dico.get('ksi0', 1.))
        self.gamma = float(density_dico.get('gamma', 2.))
        self.beta = float(density_dico.get('beta', 1.))
        self.surface_density = bool(density_dico.get('surface_density', True))
        self.dens_at_r0 = float(density_dico.get('dens_at_r0', 1.))

        # Number of sigmas at which the Gaussian drops below 'accuracy'
        nsig = np.sqrt(-2.*np.log(self.accuracy))
        self.amin = max(self.a - nsig*self.sigma, 0.)
        self.amax = self.a + nsig*self.sigma
        # maximum distance from the star (apocentre of the outer edge) for the integration, AU
        self.rmax = self.amax*(1.+self.e)
        # maximum height for the integration, AU (evaluated at amax so the flared outer edge isn't truncated)
        self.zmax = self.ksi0*(self.amax/self.a)**self.beta*(-np.log(self.accuracy))**(1./self.gamma)
        self.itiltthreshold = np.rad2deg(np.arctan(self.rmax/self.zmax))

    def density_cylindrical(self, r, costheta, z):
        """Return the particle volume density at r, theta, z."""
        a_eff = r*(1.-self.e*costheta)/(1.-self.e**2)
        radial_term = self.dens_at_r0*np.exp(-0.5*((a_eff-self.a)/self.sigma)**2)
        # avoid dividing by zero for points passing (almost) through the star
        h_ratio = np.power(np.maximum(a_eff, 1e-3*self.a)/self.a, self.beta)
        if self.surface_density:
            radial_term = radial_term/h_ratio
        vertical_term = np.exp(-np.power(np.abs(z)/(self.ksi0*h_ratio), self.gamma))
        return radial_term*vertical_term

    def density_cartesian(self, x, y, z):
        r = np.sqrt(x**2+y**2)
        costheta = 0 if r == 0 else x/r
        return self.density_cylindrical(r, costheta, z)

    def print_info(self, pxInAu=None):
        print('Type of structure: Gaussian ring')
        print('Centre (semi-major axis): {0:.2f} au, FWHM: {1:.2f} au'.format(self.a, self.fwhm))
        print('Eccentricity: {0:.2f}'.format(self.e))
        print('Scale height: {0:.2f} au at {1:.2f} au'.format(self.ksi0, self.a))
        print('Vertical profile index: {0:.2f}, flaring coefficient: {1:.2f}'.format(self.gamma, self.beta))
        print('Gaussian in surface density: {0}'.format(self.surface_density))
        print('Maximum radius for integration: {0:.2f} au'.format(self.rmax))
        print('Maximum height for integration: {0:.2f} au'.format(self.zmax))


def grater_gauss_2hg_disk_model(r0, fwhm, h0, pa, incl, g1, g2, wg1,
                                e=0., omega=0., gamma=2., beta=1., xdo=0, ydo=0,
                                cent=np.array([160.,160.]), distance=10., nx=320, ny=320,
                                pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                                halfNbSlices=25, polar=False, flux_max=1.0, surface_density=True):
    """
    Same as grater_2hg_disk_model, but with a Gaussian radial density profile instead of two power laws.

    r0: ring centre (semi-major axis) in au
    fwhm: ring full width at half maximum in au
    h0: technically h0/r0 — the ratio of scale height to radius at r0
    surface_density: if True, the surface density is Gaussian; if False, the midplane volume density is Gaussian
    accuracy: the density is truncated where the Gaussian drops below this fraction of its peak (default 1e-3, i.e. ~3.7 sigma)
    rmax_accuracy: if not None, the integration is additionally truncated at this separation (in au)

    All other parameters as in grater_2hg_disk_model.
    """
    if accuracy is None:
        accuracy = 1e-3

    spf = {'name':'DoubleHG', 'g': [g1, g2], 'weight': wg1, 'polar': polar}
    dens = {'a': r0, 'fwhm': fwhm, 'ksi0': h0*r0, 'e': e, 'gamma': gamma, 'beta': beta,
            'surface_density': surface_density}

    # VIP only implements '2PowerLaws', so build the disk with a placeholder density and swap in the Gaussian one
    vip_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=(pxscale << u.arcsec/u.pixel).value,
                                  pa=pa-180., omega=omega, xdo=xdo, ydo=ydo,
                                  density_dico={'name': '2PowerLaws'}, spf_dico=spf, flux_max=flux_max,
                                  xs=cent[0], ys=cent[1])

    gauss_dens = DustEllipticalDistributionGaussian(accuracy, dens)
    if rmax_accuracy is not None:
        gauss_dens.rmax = min(gauss_dens.rmax, rmax_accuracy)
    vip_disk.dust_density.accuracy = accuracy
    vip_disk.dust_density.type = 'Gaussian'
    vip_disk.dust_density.dust_distribution_calc = gauss_dens
    vip_disk.check_inclination = lambda: None # see dont_actually_check_inclination in grater_2hg_disk_model
    disk = vip_disk.compute_scattered_light(halfNbSlices=halfNbSlices)
    return fix_star_cross(disk, cent)


def fix_star_cross(disk, cent):
    # For disks with material coincident with the stellar position along the line of sight, VIP returns an image with a plus-sign shaped region of zeros at the star's position
    # The below is a quick work-around to correct this. Very likely makes zero difference following PSF-convolution
    if cent[0]%1 == 0:
        jvals = (np.array([-1,0,1])+cent[0]).astype(int)
    else:
        jvals = np.array([np.floor(cent[0]), np.ceil(cent[0])]).astype(int)
        
    if cent[1]%1 == 0:
        ivals = (np.array([-1,0,1])+cent[1]).astype(int)
    else:
        ivals = np.array([np.floor(cent[1]), np.ceil(cent[1])]).astype(int)
        
    for j in jvals:
        disk[int(cent[1]),j] = np.nan
    for i in ivals:
        disk[i,int(cent[0])] = np.nan
        
    disk = convolution.interpolate_replace_nans(disk, np.ones((3,3)))
    return disk


def grater_nring_2hg_disk_model(cent=np.array([160.,160.]),
                                 distance=10., nx=320, ny=320,
                                 pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                                 polar=False, halfNbSlices=25, shared_params=None, ring_model='2PowerLaws',
                                 **disk_params):
    """
    ring_model: '2PowerLaws' (each ring needs r0_i, ain_i, aout_i) or 'Gaussian' (each ring needs r0_i, fwhm_i).
    """
    if ring_model == '2PowerLaws':
        ring_fn = grater_2hg_disk_model
    elif ring_model == 'Gaussian':
        ring_fn = grater_gauss_2hg_disk_model
    else:
        raise ValueError(f"Unknown ring_model '{ring_model}'; use '2PowerLaws' or 'Gaussian'.")

    composite_image = np.zeros((ny,nx))
    all_rings_finished = False
    i=1
    while not all_rings_finished:
        ring_params = {}
        suffix = f'_{i}'
        for pkey in disk_params:
            if pkey.endswith(suffix):
                ring_params[pkey.replace(suffix, '')] = disk_params[pkey]

        if len(ring_params) == 0:
            all_rings_finished = True
            
        else:
            # add PA, inc, h0, gamma and scattering parameters to ring_params if not already there (i.e., if not specified separately for each ring, assume they are shared across rings)
            for shared_pkey in shared_params:
                if shared_pkey not in ring_params:
                    ring_params[shared_pkey] = disk_params[shared_pkey]
            # obtain the flux scaling for this ring (default to 1 if not specified) and add the appropriately scaled ring model to the composite image
            F_i = ring_params.pop('F', 1)
            composite_image += F_i * ring_fn(**ring_params, cent=cent,
                                                            distance=distance, nx=nx, ny=ny,
                                                            pxscale=pxscale, accuracy=accuracy, 
                                                            rmax_accuracy=rmax_accuracy,
                                                            polar=polar, halfNbSlices=halfNbSlices, flux_max=1.0 )
        i+=1
    return composite_image



def lnprob_Nrings(parameters, rdi_reduc, roi, distance, pix_sr, crop_size=None, h0=0.05, gamma=2.0, err=None, wdb=None, shared_params=None, prior_ranges=None, halfNbSlices=2, return_soln=False, plot_model=False, print_chisq=False, backscattering=False, ring_model='2PowerLaws'):
    """
    parameters: [F, pa, incl, g1, (g2, wg1 if backscattering), r0_1, <ring params>_1, then for each extra ring: F_i, r0_i, <ring params>_i]
        where <ring params> are (ain, aout) for ring_model='2PowerLaws' or (fwhm,) for ring_model='Gaussian'.

    raw_model:
        your idealized / unblurred disk model image oriented north-up.

    raw_model_pxscale:
        the pixelscale used for raw_model in arcsec/px (or any equivalent astropy units)

    raw_model_center:
        the position where the star would be in the model image (zero indexed and as [x,y])
    """
    if wdb is None:
        wdb = winnie.SpaceRDI(from_fits=rdi_reduc.filename)
    # else:
    #     wdbi=copy.deepcopy(wdb)  # Create a copy of the SpaceRDI object to avoid modifying the original one
    if err is None:
        err = 1
    
    if crop_size is not None:
        wdb.set_crop(crop_size)
    # y1,y2,x1,x2 = wdb._crop_indices

    ### extract common parameters
    Fi, pai, incli, g1i = parameters[0:4]
    if backscattering:
        g2i, wg1i = parameters[4:6]
        Nshared=6
    else:
        g2i=0
        wg1i=1.0
        Nshared=4

    # per-ring radial parameters (after r0): ring 1 has [r0, *ring_keys], each extra ring has [F, r0, *ring_keys]
    if ring_model == '2PowerLaws':
        ring_keys = ['ain', 'aout']
    elif ring_model == 'Gaussian':
        ring_keys = ['fwhm']
    else:
        raise ValueError(f"Unknown ring_model '{ring_model}'; use '2PowerLaws' or 'Gaussian'.")
    Nring1 = 1 + len(ring_keys)
    Nextra = 2 + len(ring_keys)

    pdict = {
        'flux': Fi, # mJy
        'pa': pai,
        'incl': incli,
        'h0': h0,
        'gamma': gamma,
        'g1': g1i,
        'g2': g2i,
        'wg1': wg1i,
        'F_1': 1.0
    }
    for key, val in zip(['r0']+ring_keys, parameters[Nshared:Nshared+Nring1]):
        pdict[f'{key}_1'] = val

    # check how many extra rings are specified in the parameters by looking at the length of the parameters list
    Nring_extra = (len(parameters) - (Nshared + Nring1)) / Nextra
    if not Nring_extra.is_integer():
        raise ValueError(f"The number of parameters for extra rings is not a multiple of {Nextra}. Each extra ring should have {Nextra} parameters: {', '.join(['F', 'r0']+ring_keys)}.")
    Nring_extra = int(Nring_extra)
    # populate pdict with the extra ring parameters
    for i in range(Nring_extra):
        start = Nshared + Nring1 + i*Nextra
        for key, val in zip(['F', 'r0']+ring_keys, parameters[start:start+Nextra]):
            pdict[f'{key}_{i+2}'] = val

    # apply priors
    if priors(pdict, prior_ranges=prior_ranges):
        return -np.inf  # Return negative infinity if any parameter is outside its prior range

    # require rings to be ordered by increasing r0 (avoids label switching between rings)
    r0s = [pdict[f'r0_{i+1}'] for i in range(Nring_extra+1)]
    if np.any(np.diff(r0s) <= 0):
        return -np.inf
    
    # Generate the raw model oriented north-up
    # We generate a detector-sampled raw model, and then subpixelate it before PSF convolution
    # You could alter this to generate an oversampled model quite easily, but in my experience the
    # only difference is runtime.
    raw_model = grater_nring_2hg_disk_model(**pdict, cent=wdb.c_star, distance=distance,
                                            nx=wdb.nx, ny=wdb.ny, pxscale=wdb.pxscale,
                                            rmax_accuracy=None, halfNbSlices=halfNbSlices, shared_params=shared_params,
                                            ring_model=ring_model)
    # normalize raw_model to have an integrated flux of F_i
    
    # print raw flux
    # print(f"Raw model flux: {np.sum(raw_model)*pix_sr*1e6:.3e} Jy")

    raw_model /= np.sum(raw_model)*pix_sr
    raw_model *= (Fi/1.0e9)  # Convert F_i from mJy to MJy for normalization

    # print raw flux after normalization
    # print(f"Raw model flux after normalization: {np.sum(raw_model)*pix_sr*1e6:.3e} Jy")

    

    # Convolve the model using the prepared SpaceConvolution object then set it as the CSS model in the SpaceRDI object
    wdb.set_circumstellar_model(raw_model=raw_model, raw_model_pxscale=wdb.pxscale, raw_model_center=wdb.c_star)

    # Run RDI on the model to get the forward-modeled result
    fmrdi_reduc = wdb.run_rdi(forward_model=True, save_products=False, collapse_rolls=True, derotate=rdi_reduc.derotated, correct_distortion=rdi_reduc.distortion_corrected)


    if plot_model:
        # plot data, model and residuals
        pim = np.array([rdi_reduc.rolls[0], raw_model, fmrdi_reduc.rolls[0], rdi_reduc.rolls[0]-fmrdi_reduc.rolls[0]])
        fig, (ax1,ax2, ax3, ax4) = quick_implot(np.where(roi, pim, np.nan), clim_perc=99., cmap='RdBu_r', show=False, 
                                   show_ticks=True, 
                                   extent=rdi_reduc.extent)  
        # titles
        ax1.set_title('Data')
        ax2.set_title('Raw Model')
        ax3.set_title('Forward Model')
        ax4.set_title('Residuals')

        

    chisq = np.sum(((rdi_reduc.rolls[0]-fmrdi_reduc.rolls[0])/err)[roi]**2)

    if print_chisq:
        # print Chisq and reduced chi-squared metric for current parameters
        dof = np.sum(roi) - len(parameters)
        print(f"Chi2: {chisq:.1f}, dof: {dof}, reduced Chi2: {chisq/dof:.3f}")

    if return_soln:

        return raw_model, fmrdi_reduc, wdb

    # return the log of the likelihood, which is -0.5*chisq (up to an additive constant that we can ignore for optimization or MCMC sampling)
    return -0.5 * chisq


def promote_spacerdi_to_float64(wdb):
    """
    Converts the science/reference image and error cubes of a winnie SpaceRDI object to float64 (JWST data are
    float32) and refreshes the cropped views. Call right after wdb.load_concat(...).

    Why: RDI solves (R R^T) c = R t for the reference coefficients. For our data, R R^T has a condition number of
    ~1e9, beyond what float32 (~7 significant digits) can resolve, so float32 RDI results depend on rounding details
    of the BLAS/LAPACK library: OpenBLAS happens to give results close to the float64 solution, while Apple
    Accelerate gives residuals several times too large. In float64 the result is stable with any library.
    """
    for attr in ['_imcube_sci', '_imcube_ref', '_errcube_sci', '_errcube_ref', '_imcube_css']:
        arr = getattr(wdb, attr, None)
        if arr is not None and arr.dtype != np.float64:
            setattr(wdb, attr, arr.astype(np.float64))
    # refresh the cropped aliases (imcube_sci, imcube_ref, ...) so they point at the float64 arrays
    if wdb.cropped_shape is None:
        wdb.set_crop()
    else:
        wdb.set_crop(list(wdb.cropped_shape))


## Faster forward-modeled RDI for repeated likelihood calls (e.g. MCMC)
# In forward-model mode winnie computes, for each optimization zone, coefficients c = (R R^T)^-1 R m, where R holds the
# reference images within the zone and m the convolved model, and then subtracts c . R_sub within the subtraction zone.
# R, R R^T and its LU factorization do not depend on the model, so they can be computed once and reused for every call.

def precompute_fm_rdi(wdb):
    """
    Precomputes the model-independent parts of forward-modeled RDI for the current crop, zones and RDI settings of
    the winnie SpaceRDI object 'wdb'. The returned dictionary is used by fm_rdi_cached / lnprob_Nrings_cached, and must
    be recomputed whenever the crop, zones or RDI presets of wdb change.

    Only supports the settings that are valid for forward modeling without smoothing (e.g. rdi_presets);
    raises a ValueError for HPFRDI (opt_smoothing_fn) or fixed coefficients (coeffs_in).
    """
    settings = {**wdb.rdi_settings}
    for key in ['opt_smoothing_fn', 'coeffs_in', 'hcube_css']:
        if settings.get(key, None) is not None:
            raise ValueError(f"precompute_fm_rdi does not support the RDI setting '{key}'; use wdb.run_rdi(forward_model=True) instead.")
    zero_nans = settings.get('zero_nans', False)
    ref_mask = settings.get('ref_mask', None)

    # always solve in float64: R R^T is too ill-conditioned for float32 (see promote_spacerdi_to_float64)
    imcube_ref = np.asarray(wdb.imcube_ref, dtype=np.float64) # (nT_ref, ny, nx), cropped
    nT_ref = imcube_ref.shape[0]
    regions = []
    for Ri, (opt_i, sub_i) in enumerate(zip(wdb.optzones, wdb.subzones)):
        use = np.ones(nT_ref, dtype=bool) if ref_mask is None else np.asarray(ref_mask[Ri], dtype=bool)
        refarrs = imcube_ref[:, opt_i]
        if zero_nans:
            refarrs = np.nan_to_num(refarrs)
        refarrs = refarrs[use]
        regions.append(dict(opt=opt_i.copy(),
                            sub=sub_i.copy(),
                            use=use,
                            refarrs=refarrs, # (nT_ref_used, nP_opt)
                            lu_piv=linalg.lu_factor(refarrs @ refarrs.T, check_finite=False),
                            ref_sub=imcube_ref[:, sub_i].copy())) # (nT_ref, nP_sub), never zeroed (as in winnie's reconstruction)

    return dict(regions=regions, zero_nans=zero_nans, crop_indices=tuple(wdb._crop_indices),
                shape=imcube_ref.shape, optzones=wdb.optzones.copy(), subzones=wdb.subzones.copy())


def fm_rdi_cached(wdb, fm_cache, derotate=False, correct_distortion=False, collapse_rolls=True):
    """
    Equivalent to wdb.run_rdi(forward_model=True, save_products=False, ...) for the circumstellar model currently set in
    wdb (wdb.set_circumstellar_model), but using the reference-only quantities precomputed by precompute_fm_rdi.
    Returns a winnie SpaceReduction object.
    """
    if (tuple(wdb._crop_indices) != fm_cache['crop_indices'] or wdb.imcube_ref.shape != fm_cache['shape']
            or not np.array_equal(wdb.optzones, fm_cache['optzones']) or not np.array_equal(wdb.subzones, fm_cache['subzones'])):
        raise ValueError("fm_cache does not match the current crop/zones of wdb; rerun precompute_fm_rdi(wdb).")
    for key in ['opt_smoothing_fn', 'coeffs_in', 'hcube_css']:
        if wdb.rdi_settings.get(key, None) is not None:
            raise ValueError(f"The RDI presets of wdb now include '{key}' (e.g. HPFRDI or MCRDI), which fm_cache does not "
                             "cover; switch back with wdb.rdi_presets(), or use wdb.run_rdi(forward_model=True).")
    if wdb.imcube_css is None:
        raise ValueError("Set a circumstellar model with wdb.set_circumstellar_model before calling fm_rdi_cached.")

    hcube = np.asarray(wdb.imcube_css, dtype=np.float64) # (nT, ny, nx)
    nT = hcube.shape[0]
    psf_model = np.full(hcube.shape, np.nan, dtype=hcube.dtype)
    for reg in fm_cache['regions']:
        tararrs = hcube[:, reg['opt']]
        if fm_cache['zero_nans']:
            tararrs = np.nan_to_num(tararrs)
        coeffs = np.zeros((nT, len(reg['use'])), dtype=hcube.dtype)
        coeffs[:, reg['use']] = linalg.lu_solve(reg['lu_piv'], reg['refarrs'] @ tararrs.T, check_finite=False).T
        psf_model[:, reg['sub']] = coeffs @ reg['ref_sub']

    residuals = hcube - psf_model
    return wdb.imcube_to_products(residuals, derotate=derotate, correct_distortion=correct_distortion,
                                  collapse_rolls=collapse_rolls, output_ext=wdb.output_ext+'_fwdmod',
                                  reduc_label=f'FM {wdb.reduc_label}', save_products=False)


def nring_parameters_to_pdict(parameters, h0=0.05, gamma=2.0, backscattering=False, ring_model='2PowerLaws'):
    """
    Converts the parameter vector used by lnprob_Nrings / lnprob_Nrings_cached into a dictionary of disk parameters
    (see lnprob_Nrings for the parameter order). Returns (pdict, Nrings).
    """
    Fi, pai, incli, g1i = parameters[0:4]
    if backscattering:
        g2i, wg1i = parameters[4:6]
        Nshared = 6
    else:
        g2i = 0
        wg1i = 1.0
        Nshared = 4

    if ring_model == '2PowerLaws':
        ring_keys = ['ain', 'aout']
    elif ring_model == 'Gaussian':
        ring_keys = ['fwhm']
    else:
        raise ValueError(f"Unknown ring_model '{ring_model}'; use '2PowerLaws' or 'Gaussian'.")
    Nring1 = 1 + len(ring_keys)
    Nextra = 2 + len(ring_keys)

    pdict = {'flux': Fi, 'pa': pai, 'incl': incli, 'h0': h0, 'gamma': gamma,
             'g1': g1i, 'g2': g2i, 'wg1': wg1i, 'F_1': 1.0}
    for key, val in zip(['r0']+ring_keys, parameters[Nshared:Nshared+Nring1]):
        pdict[f'{key}_1'] = val

    Nring_extra = (len(parameters) - (Nshared + Nring1)) / Nextra
    if not Nring_extra.is_integer():
        raise ValueError(f"The number of parameters for extra rings is not a multiple of {Nextra}. Each extra ring should have {Nextra} parameters: {', '.join(['F', 'r0']+ring_keys)}.")
    Nring_extra = int(Nring_extra)
    for i in range(Nring_extra):
        start = Nshared + Nring1 + i*Nextra
        for key, val in zip(['F', 'r0']+ring_keys, parameters[start:start+Nextra]):
            pdict[f'{key}_{i+2}'] = val
    return pdict, Nring_extra+1


def lnprob_Nrings_cached(parameters, rdi_reduc, roi, distance, pix_sr, crop_size=None, h0=0.05, gamma=2.0, err=None, wdb=None,
                         shared_params=None, prior_ranges=None, halfNbSlices=2, return_soln=False, plot_model=False,
                         print_chisq=False, backscattering=False, ring_model='2PowerLaws', fm_cache=None):
    """
    Same as lnprob_Nrings, but the forward-modeled RDI uses the reference-only quantities precomputed with
    fm_cache = precompute_fm_rdi(wdb) (computed after setting the crop, zones and RDI presets used for the fit).
    """
    if wdb is None or fm_cache is None:
        raise ValueError("lnprob_Nrings_cached requires wdb and fm_cache = precompute_fm_rdi(wdb).")
    if err is None:
        err = 1
    if crop_size is not None:
        wdb.set_crop(crop_size)

    pdict, Nrings = nring_parameters_to_pdict(parameters, h0=h0, gamma=gamma, backscattering=backscattering, ring_model=ring_model)
    Fi = pdict['flux']

    # apply priors
    if priors(pdict, prior_ranges=prior_ranges):
        return -np.inf

    # require rings to be ordered by increasing r0 (avoids label switching between rings)
    r0s = [pdict[f'r0_{i+1}'] for i in range(Nrings)]
    if np.any(np.diff(r0s) <= 0):
        return -np.inf

    raw_model = grater_nring_2hg_disk_model(**pdict, cent=wdb.c_star, distance=distance,
                                            nx=wdb.nx, ny=wdb.ny, pxscale=wdb.pxscale,
                                            rmax_accuracy=None, halfNbSlices=halfNbSlices, shared_params=shared_params,
                                            ring_model=ring_model)
    # normalize raw_model to have an integrated flux of F_i (mJy -> MJy)
    raw_model /= np.sum(raw_model)*pix_sr
    raw_model *= (Fi/1.0e9)

    wdb.set_circumstellar_model(raw_model=raw_model, raw_model_pxscale=wdb.pxscale, raw_model_center=wdb.c_star)
    fmrdi_reduc = fm_rdi_cached(wdb, fm_cache, derotate=rdi_reduc.derotated, correct_distortion=rdi_reduc.distortion_corrected)

    if plot_model:
        pim = np.array([rdi_reduc.rolls[0], raw_model, fmrdi_reduc.rolls[0], rdi_reduc.rolls[0]-fmrdi_reduc.rolls[0]])
        fig, (ax1,ax2, ax3, ax4) = quick_implot(np.where(roi, pim, np.nan), clim_perc=99., cmap='RdBu_r', show=False,
                                   show_ticks=True, extent=rdi_reduc.extent)
        ax1.set_title('Data')
        ax2.set_title('Raw Model')
        ax3.set_title('Forward Model')
        ax4.set_title('Residuals')

    chisq = np.sum(((rdi_reduc.rolls[0]-fmrdi_reduc.rolls[0])/err)[roi]**2)

    if print_chisq:
        dof = np.sum(roi) - len(parameters)
        print(f"Chi2: {chisq:.1f}, dof: {dof}, reduced Chi2: {chisq/dof:.3f}")

    if return_soln:
        return raw_model, fmrdi_reduc, wdb

    return -0.5 * chisq


## Parallel sampling (e.g. emcee with a pool of worker processes)
# Each worker must have its own copy of the SpaceRDI object: lnprob_Nrings_cached modifies it on every call
# (set_crop, set_circumstellar_model), so walkers sharing one object overwrite each other's model.
# Passing wdb as an argument to emcee does not work well either: the pool pickles the arguments for every batch of
# walkers. Instead, the arguments are stored at module level and inherited by worker processes created with the
# 'fork' start method, so each worker gets a private copy once, when the pool starts, and the function sent to the
# pool (lnprob_Nrings_parallel) is cheap to pickle.

_parallel_state = {}


def set_parallel_lnprob(**kwargs):
    """
    Stores the arguments of lnprob_Nrings_cached (everything except 'parameters', given as keywords, e.g.
    rdi_reduc=..., roi=..., distance=..., pix_sr=..., crop_size=..., wdb=..., fm_cache=..., ...) for use by
    lnprob_Nrings_parallel. Must be called before make_parallel_pool, since workers copy the state when created.
    """
    _parallel_state.clear()
    _parallel_state.update(kwargs)


def lnprob_Nrings_parallel(parameters):
    """lnprob_Nrings_cached(parameters, **state), with state set by set_parallel_lnprob."""
    return lnprob_Nrings_cached(parameters, **_parallel_state)


def _fork_blas_probe():
    # Run in a forked process by make_parallel_pool: large enough to use multi-threaded BLAS if it is enabled
    a = np.ones((1000, 1000))
    a @ a


def _init_parallel_worker(worker_ncores):
    # Limit each worker to worker_ncores threads (BLAS and winnie's joblib calls) so the processes don't oversubscribe the CPU.
    # Nothing in here may raise: a failing initializer makes the pool restart its workers forever (silent hang).
    try:
        from threadpoolctl import threadpool_limits
        global _worker_threadpool_limits
        _worker_threadpool_limits = threadpool_limits(limits=worker_ncores) # kept alive for the life of the worker
    except Exception:
        pass # threadpoolctl missing: rely on the *_NUM_THREADS environment variables set before numpy was imported
    wdb = _parallel_state.get('wdb', None)
    if wdb is not None:
        wdb.ncores = worker_ncores
        if wdb.convolver is not None:
            wdb.convolver.ncores = worker_ncores


def make_parallel_pool(processes=8, worker_ncores=1):
    """
    Creates a pool of 'processes' worker processes (start method 'fork') for use with
    emcee.EnsembleSampler(..., ffm.lnprob_Nrings_parallel, pool=pool). Call set_parallel_lnprob first.
    Use it as a context manager ('with ffm.make_parallel_pool(8) as pool:') so the workers are closed afterwards.
    """
    import multiprocessing, os, warnings
    if len(_parallel_state) == 0:
        raise ValueError("Call set_parallel_lnprob(...) before make_parallel_pool.")
    if 'fork' not in multiprocessing.get_all_start_methods():
        raise RuntimeError("make_parallel_pool needs the 'fork' start method, which this platform (e.g. Windows) does not "
                           "have. Run the serial sampler instead.")
    # Multi-threaded BLAS libraries are not all safe to use in forked workers: Apple Accelerate (numpy/scipy wheels on
    # macOS >= 14) runs its threads through libdispatch and workers segfault on their first matrix product, and OpenMP
    # builds of OpenBLAS/MKL can deadlock. Limiting BLAS to one thread before numpy is first imported avoids both, at
    # little cost here (BLAS is a few % of a likelihood call).
    try:
        blas = np.show_config(mode='dicts')['Build Dependencies']['blas'].get('name', '').lower() # numpy >= 1.26
    except Exception:
        blas = ''
    if blas == 'accelerate' and os.environ.get('VECLIB_MAXIMUM_THREADS') != '1':
        raise RuntimeError("numpy uses Apple Accelerate, whose multi-threaded mode crashes in forked worker processes. "
                           "Add os.environ['VECLIB_MAXIMUM_THREADS'] = '1' at the top of the notebook, before numpy is "
                           "imported, then restart the kernel and run again.")
    missing = [v for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
               if os.environ.get(v) != '1']
    if missing and blas != 'accelerate':
        with warnings.catch_warnings(): # some imported packages (vip_hci, astropy, ...) install global 'ignore' filters
            warnings.simplefilter('always')
            warnings.warn(f"{', '.join(missing)} not set to 1 before numpy was imported; multi-threaded BLAS can hang "
                          "in forked worker processes with some builds. Set them at the top of the notebook if the run "
                          "stalls.", stacklevel=2)
    ctx = multiprocessing.get_context('fork')
    # The checks above can't tell whether the limits were set before numpy was imported (if not, they have no effect).
    # So test it: a forked process must be able to do a large matrix product, without crashing or hanging.
    probe = ctx.Process(target=_fork_blas_probe)
    probe.start()
    probe.join(timeout=60)
    if probe.is_alive():
        probe.terminate()
        raise RuntimeError("A forked test process hung in a matrix product, so the parallel MCMC would hang too. Set "
                           "OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and VECLIB_MAXIMUM_THREADS to '1' at "
                           "the top of the notebook, before numpy is imported, then restart the kernel.")
    if probe.exitcode != 0:
        raise RuntimeError(f"A forked test process crashed in a matrix product (exit code {probe.exitcode}), so the "
                           "parallel MCMC workers would crash too. Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, "
                           "MKL_NUM_THREADS and VECLIB_MAXIMUM_THREADS to '1' at the top of the notebook, before numpy "
                           "is imported (if numpy was already loaded, the setting has no effect), then restart the kernel.")
    return ctx.Pool(processes=processes, initializer=_init_parallel_worker, initargs=(worker_ncores,))


def obj_fn(p, rdi_reduc, wdb, roi, distance, err_weighting=False, err_map=None, rmax_accuracy=None,
           halfNbSlices=25, return_soln=False, q_clip=None, lsq_sfac=True,
           count=True, store_samples=False, clear_each_call=True, shared_params=None, ring_model='2PowerLaws',
           fm_cache=None):
    """
    Parameters
    ----------
    p: lmfit.parameter.Parameters
        LMFit parameters object containing at least: r0_1, h0_1, ain_1, aout_1, pa_1, incl_1, g1_1, g2_1, wg1_1
        (or r0_1, fwhm_1, ... instead of ain_1, aout_1 if ring_model='Gaussian')

    ring_model: str
        Radial profile of each ring: '2PowerLaws' or 'Gaussian'. See grater_nring_2hg_disk_model.

    fm_cache: dict or None
        If given (fm_cache = precompute_fm_rdi(wdb), computed for the current crop, zones and RDI presets of wdb), the
        forward-modeled RDI uses these precomputed reference-image quantities (fm_rdi_cached) instead of
        wdb.run_rdi(forward_model=True), which is equivalent but much faster. If None, wdb.run_rdi is used.

    rdi_reduc: winnie.space.SpaceReduction
        An RDI reduction of the data using the current settings in 'wdb', the Winnie SpaceRDI object.
        If rdi_reduc.derotated is True, the forward-modeled disk image is compared to the single
        derotated+combined rdi_reduc.im (aligned north-up). If False, it is compared to every roll
        in rdi_reduc.rolls simultaneously (each in its own detector-frame orientation).
        
    wdb: winnie.space.SpaceRDI
        The Winnie SpaceRDI object that was used to generate rdi_reduc
        
    roi: numpy.ndarray
        Boolean 'region of interest' array having the same shape as the last two axes of rdi_reduc.im /
        rdi_reduc.rolls that indicates which pixels should be included in the goodness of fit evaluation.
        The same 2D mask is applied to every roll when rdi_reduc.derotated is False.
        
    distance: float
        Distance to the target star in parsecs.
        
    err_weighting: bool
        If err_weighting is True, the array stored in rdi_reduc.err is used to weight the residuals.
        Unless manually changed, rdi_reduc.err is the propagated pixel uncertainty map based on the ERR
        FITS extension. Currently, these arrays are not accurate (because they neglect the noise reduction 
        from SpaceKLIP's use of psuedo reference pixels). This could be useful if you have significant small
        separation stellar residuals that are affecting your disk model fit.

    err_map: numpy.ndarray
        If err_weighting is True, err_map is used to weight the residuals instead of rdi_reduc.err. 
        Must have the same shape as rdi_reduc.im / rdi_reduc.rolls.
        
    rmax_accuracy: float
        The largest separation (in au) at which to calculate the disk model. If None, defaults to the edge
        of the FOV.
        
    halfNbSlices: int
        The number of planar slices to compute above and below the disk midplane when generating the raw 
        disk model.
        
    return_soln: bool
        If True, rather than returning a residual array, returns the forward modeled image and some other items
        
    q_clip: tuple or list or numpy.ndarray
        If not None, q_clip gives a lower and upper quantile bound for the residuals. Any values outside the 
        quantile range are clipped when evaluating goodness of fit. Can be useful for data with significant 
        artifacts within the region of interest (e.g., poor reference match, uncorrected hot pixels, etc). 
        E.g., q_clip = [5,95] will compute goodness of fit using only the inner 5th-95%ile of the distribution 
        of residual pixel values in your region of interest.
    
    lsq_sfac: bool
        If True, least-squares renormalize the brightness of the forward modeled disk image to match the data
        within the region of interest (and considering the uncertainties when err_weighting is True). 
        This should nearly always be True.
        
    count: bool
        If True, will advance and display a counter during the optimization. Requires setting 'counter = 0' somewhere
        outside of the function.
        
    store_samples: bool
        If True, each time obj_fn is called with return_soln=False, an array containing the parameters values and the 
        corresponding chi-squared metric will be appended to a list titled samples. Requires setting 'samples = []'
        somewhere outside of the function.
    
    clear_each_calls: bool
        If True, clears the output every time the function is called. Mostly to prevent accumulating a large number 
        of print statements if the loop triggers a warning somewhere.
    """
    pdict = p.valuesdict() 

    # If the data is derotated, compare against the single north-up combined image (.im);
    # otherwise cycle through every roll simultaneously in its native detector-frame orientation (.rolls).
    if rdi_reduc.derotated:
        data_cmp = rdi_reduc.im
        if err_weighting:
            if err_map is not None:
                sig = err_map
            else:
                sig = rdi_reduc.err
        else:
            sig = None
    else:
        data_cmp = rdi_reduc.rolls
        if err_weighting:
            if err_map is not None:
                sig = err_map
            else:
                sig = rdi_reduc.err_rolls
        else:
            sig = None


    # Generate the raw model oriented north-up
    # We generate a detector-sampled raw model, and then subpixelate it before PSF convolution
    # You could alter this to generate an oversampled model quite easily, but in my experience the
    # only difference is runtime.
    raw_model = grater_nring_2hg_disk_model(**pdict, cent=wdb.c_star, distance=distance,
                                            nx=wdb.nx, ny=wdb.ny, pxscale=wdb.pxscale,
                                            rmax_accuracy=rmax_accuracy, halfNbSlices=halfNbSlices, shared_params=shared_params,
                                            ring_model=ring_model)

    # Convolve the model using the prepared SpaceConvolution object then set it as the CSS model in the SpaceRDI object
    wdb.set_circumstellar_model(raw_model=raw_model, raw_model_pxscale=wdb.pxscale, raw_model_center=wdb.c_star)


    # Run RDI on the model to get the forward-modeled result
    # collapse_rolls must always be True here since fmrdi_reduc.rolls is required below
    # regardless of return_soln (tying it to return_soln left both im and rolls as None
    # whenever rdi_reduc.derotated is False, causing SpaceReduction to try loading a
    # nonexistent saved file instead of building the result in memory).
    if fm_cache is None:
        fmrdi_reduc = wdb.run_rdi(forward_model=True, save_products=False, collapse_rolls=True, derotate=rdi_reduc.derotated, correct_distortion=rdi_reduc.distortion_corrected)
    else:
        fmrdi_reduc = fm_rdi_cached(wdb, fm_cache, collapse_rolls=True, derotate=rdi_reduc.derotated, correct_distortion=rdi_reduc.distortion_corrected)
    model_cmp = fmrdi_reduc.im if rdi_reduc.derotated else fmrdi_reduc.rolls

    if lsq_sfac:
        # Rescale the model to minimize residuals with the data
        # Median filter both first to avoid skewing the rescaling with uncorrected hot pixels, etc.
        # median_filter_sequence and model_rescale_factor both operate on the trailing (ny, nx) axes,
        # so this works identically whether data_cmp/model_cmp are single images (derotated) or
        # per-roll cubes (not derotated) — all rolls are fit simultaneously in the latter case.
        sfac = model_rescale_factor(*median_filter_sequence(np.array([data_cmp, model_cmp]),
                                                            footprint=np.array([[0,1,0],
                                                                                [1,1,1],
                                                                                [0,1,0]]),
                                                            prop_threshold=0.8), sig=sig, mask=roi)
    else:
        sfac = 1
        
    # fmrdi_reduc.im only exists (is not None) when derotate=True (see imcube_to_products);
    # it must be rescaled alongside .rolls so that anything using fmrdi_reduc.im (e.g. plotting
    # cells) sees a result consistent with the residuals computed below.
    fmrdi_reduc.rolls *= sfac
    if rdi_reduc.derotated:
        fmrdi_reduc.im *= sfac
    model_cmp = fmrdi_reduc.im if rdi_reduc.derotated else fmrdi_reduc.rolls
    
    if return_soln:
        raw_model *= sfac
        wdb.imcube_css *= sfac
        return sfac, raw_model, fmrdi_reduc, wdb

    res = data_cmp - model_cmp
    
    if sig is not None:
        res /= sig
    res = res[..., roi]
    # print reduced chi-squared metric for current parameters
    Chi2 = np.nansum(res**2)
    dof = res.size - len(p)
    print(f"Chi2: {Chi2:.1f}, dof: {dof}, reduced Chi2: {Chi2/dof:.3f}")

    if q_clip is None:
        res = np.abs(res)
    else:
        low,upp = np.nanpercentile(res, q_clip)
        res = np.abs(res[(res >= low) & (res <= upp)])
        
    if clear_each_call:
        clear_output()

    if count:
        global counter
        counter += 1
        print('Models evaluated: {0: <16}'.format(counter), end='\r')
        
    if store_samples: # Append each set of paramaters and the corresponding chisq to a list
        chisq = np.nansum(res**2)
        sample = [pdict[key] for key in pdict.keys()]
        sample.append(chisq)
        samples.append(sample)
        
    return res


def model_rescale_factor(A, B, sig=None, mask=None):
    """
    Determines the value of scalar c such that:
        chi^2 = sum [ (A-c*B)^2 / sig^2 ]
    is minimized.
    
    Parameters
    ----------
    A : numpy.ndarray
        Array of measurements
    B : numpy.ndarray
        Array of model values. Shape must match A and B
    sig : numpy.ndarray, optional
        The 1 sigma uncertainty for the measurements of A.
    mask : numpy.ndarray, optional
        A boolean mask with False for entries of A, B, and sig not to be
        utilized, and True for entries that are. Defaults to None.
    Returns
    -------
    c : float
        The scaling factor to multiply the model (B) by to achieve the minimum chi^2
        for measurements (A) having the given uncertainties (sig).
    """
    if np.shape(A) != np.shape(B):
        raise ValueError("A and B must be arrays of the same shape!")
    if sig is not None:
        if np.shape(A) != np.shape(sig):
            raise ValueError("A, B, and sig must be arrays of the same shape if sig is specified!")
    else:
        sig = 1
    if mask is None:
        c = np.nansum(A * B / (sig ** 2)) / np.nansum((B ** 2) / (sig ** 2))
    elif np.shape(mask)[-2:] != np.shape(A)[-2:]:
        raise ValueError("If provided, mask's shape must match the final axes of A, B, and sig!")
    else:
        Amsk, Bmsk = A[..., mask], B[..., mask]
        if np.ndim(sig) != 0:
            Smsk = sig[..., mask]
        else:
            Smsk = sig
        c = np.nansum(Amsk * Bmsk / (Smsk ** 2)) / np.nansum((Bmsk ** 2) / (Smsk ** 2))
    return c

def priors(pdict, prior_ranges):
    """
    Define the prior ranges for the parameters.
    Returns True if any parameter is outside its prior range, False otherwise.
    """
    if prior_ranges==None:
        prior_ranges={}

    # Check if any parameter is outside its prior range
    for key, (lower, upper) in prior_ranges.items():
        if key in pdict: # check if the key exists in pdict to avoid KeyError
            if not (lower <= pdict[key] <= upper):
                return True
    return False




def flip_xlabels(ax):
    xticks = ax.get_xticks()
    ax.set_xticklabels([f"$0''$" if x == 0 else f"${-x:.0f}''$" for x in xticks])
    return ax

# flip sign of x-axis to follow astronomical convention of east to the left and west to the right
def flip_xaxis(ax):
    ax.invert_xaxis()
    return ax

