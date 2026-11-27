import json
import numpy as np
from scipy.special import gammaln


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
from vip_hci.fm.scattered_light_disk import DustEllipticalDistribution2PowerLaws
from astropy import convolution
import stpsf
from IPython.display import clear_output
import astropy.units as u
from spaceKLIP import starphot

# Module-level counter used by obj_fn's progress display (see 'count' parameter).
# 'global counter' inside obj_fn binds to this module's namespace, not the caller's,
# so setting 'counter = 0' in a notebook has no effect here; use 'ffm.counter = 0' instead.
counter = 0


## FM functions adapted so rings share same inc, PA and scattering parameters (but not necessarily the same r0, h0, ain, aout, or gamma)

# Now, let's a) generalize the GRaTer model to a composite image formed from the super-position of N individual ring-like components, and b) set up an objective function to optimize a disk model for the data.

# (we don't actually need more than 1 ring here — I just had the code already and thought it might be helpful for other applications)

class DustEllipticalDistribution2PowerLawsEps(DustEllipticalDistribution2PowerLaws):
    """
    VIP's DustEllipticalDistribution2PowerLaws with a free smoothing exponent eps for the two power laws
    (used by grater_2hg_disk_model when eps != 2, by swapping the class of VIP's own density object).

    The radial profile is evaluated at the semi-major axis a_eff = r*(1-e*cos(theta))/(1-e^2), as in VIP:
        rho_mid(a_eff) = dens_at_r0 * [2 / ((a_eff/r0)^(-eps*ain) + (a_eff/r0)^(-eps*aout))]^(1/eps)
    eps = 2 is VIP's (GRaTer's, Augereau et al. 1999) prescription. A larger eps gives a sharper turnover between
    the two slopes (a broken power law for eps -> inf), a smaller one a broader, rounder peak; the slopes far from
    r0 are ain and aout for any eps. The vertical profile is VIP's: exp(-(|z|/H)^gamma), with H = ksi0*(a_eff/r0)^beta
    (ksi0 is VIP's scale height, not the standard deviation; see vip_scale_height).

    Everything else (rmax, zmax, e, amin, itiltthreshold) is VIP's. rmax only depends on aout, which sets the
    slope at large radii for any eps. Note that VIP's print_info still computes the ring FWHM with eps = 2.
    """
    eps = 2.

    def set_eps(self, eps):
        """Set the smoothing exponent (> 0) and update the peak radii (as VIP does for eps = 2)."""
        if not eps > 0:
            raise ValueError(f'eps must be > 0 (got {eps}).')
        self.eps = float(eps)
        if self.ain != self.aout:
            self.apeak = self.a*np.power(-self.ain/self.aout, 1./(self.eps*(self.ain-self.aout)))
            Gamma_in, Gamma_out = self.ain+self.beta, self.aout+self.beta
            self.apeak_surface_density = self.a*np.power(-Gamma_in/Gamma_out, 1./(self.eps*(Gamma_in-Gamma_out)))

    def density_cylindrical(self, r, costheta, z):
        """Return the particle volume density at r, theta, z (VIP's, with smoothing exponent eps)."""
        radial_ratio = r/(self.p/(1-self.e*costheta))
        with np.errstate(over='ignore', divide='ignore'): # r -> 0 or large eps: den -> inf, density -> 0
            den = (np.power(radial_ratio, -self.eps*self.ain) +
                   np.power(radial_ratio, -self.eps*self.aout))
        radial_density_term = np.power(2./den, 1./self.eps)*self.dens_at_r0
        if self.pmin > 0:
            radial_density_term[r/(self.pmin/(1-self.e*costheta)) <= 1] = 0
        den2 = (self.ksi0*np.power(radial_ratio, self.beta))
        # Could change to np.exp(-1/2*np.power(np.abs(z)/den2, self.gamma)) for a Gaussian vertical profile, but that would be inconsistent with VIP's 2PowerLaws implementation.
        vertical_density_term = np.exp(-np.power(np.abs(z)/den2, self.gamma))
        return radial_density_term*vertical_density_term


def vip_scale_height(sigma_z, gamma=2.):
    """
    Converts a vertical standard deviation sigma_z into the scale height ksi of VIP's vertical profile
    exp(-(|z|/ksi)^gamma), whose standard deviation is ksi*sqrt(Gamma(3/gamma)/Gamma(1/gamma)).
    For gamma = 2 (Gaussian), ksi = sqrt(2)*sigma_z.

    Our model functions take h0 = sigma_z/r (at the reference radius) and pass ksi0 = vip_scale_height(h0*r_ref, gamma)
    to the density classes, so H(r) = h0*r is the vertical standard deviation of the dust density for any gamma.
    """
    return sigma_z*np.exp(0.5*(gammaln(1./gamma) - gammaln(3./gamma)))


def _set_zmax_nsigma(dist, zmax_nsigma, sigma_ref, r_ref, r_z):
    """
    Sets the integration height of the density object 'dist' to zmax_nsigma vertical standard deviations at radius r_z
    (sigma_ref: standard deviation at r_ref, flaring as (r/r_ref)^beta). Does nothing if zmax_nsigma is None (then
    zmax is set by the density's 'accuracy', as in VIP).
    """
    if zmax_nsigma is None:
        return
    dist.zmax = zmax_nsigma*sigma_ref*(r_z/r_ref)**dist.beta
    dist.itiltthreshold = np.rad2deg(np.arctan(dist.rmax/dist.zmax))


## Ray tracing with a line-of-sight range adapted to the flared disk (alternative to VIP's compute_scattered_light)
# VIP integrates every line of sight over the same range of heights, |z| <= 2*zmax, where zmax is the height at which
# the density drops below 'accuracy' at a single reference radius, and spaces the nodes geometrically towards the
# midplane. With linear flaring the disk is much thinner than that at small radii (H proportional to r), so few slices badly
# overestimate the column density there. compute_scattered_light_wedge instead integrates each line of sight only
# where it is inside the flared disk, |z| <= c*r with c = zmax/r_ref, i.e. the same accuracy level at every radius.
# The limits are where the ray itself crosses that envelope, so a ray passing close to the star in projection but
# through high-altitude material at large radii (inclined disks) is still integrated over all of it.

def _wedge_slope(dist):
    """
    Slope c of the envelope |z| <= c*r outside which the density of 'dist' is below its 'accuracy', for linear
    flaring (beta = 1). All our density classes use H = ksi0*a_eff/r_ref with a_eff <= r/(1-e), and set zmax at a
    radius r_z: VIP's 2PowerLaws (and 2PowerLawsEps) at a, the Gaussian and TanhGaps ones at their outer edge amax.
    """
    r_z = dist.amax if hasattr(dist, 'amax') else dist.a
    return dist.zmax/(r_z*(1.-dist.e))


def compute_scattered_light_wedge(vip_disk, halfNbSlices=25, lwidth=None):
    """
    Equivalent of vip_disk.compute_scattered_light(halfNbSlices) (vip_disk: VIP ScatteredLightDisk) in which each line
    of sight is integrated over the section inside the flared disk, |z| <= c*r (see _wedge_slope), instead of
    VIP's fixed |z| <= 2*zmax for every pixel. Requires linear flaring (beta = 1).

    For each pixel, the line of sight l (as in VIP: l = 0 at the sky plane through the star) crosses the midplane at
    lz0 and is inside the envelope for l1 <= l <= l2:
      - if tan(incl) < 1/c (the line of sight is steeper than the disk surface), [l1, l2] is the single interval
        between the two crossings of the envelope;
      - otherwise (very inclined disks) the ray can leave the envelope and re-enter it at larger radii, so the
        whole ray within r <= rmax is integrated.
    [l1, l2] is also limited to r <= rmax. The 2*halfNbSlices-1 nodes are placed on either side of lz0 and the
    integral is a trapezoidal sum, as in VIP.

    lwidth: node spacing. 0 spaces the nodes uniformly on each side of lz0; lwidth > 0 spaces them geometrically
        towards lz0 as VIP does (VIP uses lwidth = 100). None (default): 0 if c*tan(incl) < 0.5, else 10 (the rays then
        graze the disk, their range is long compared with the disk thickness, and the density is concentrated near
        the midplane crossing).
        Tests on a 3-ring model (h0 = 0.05, rings at 32, 47 and 70 au) gave maximum errors with halfNbSlices = 4 of
        4e-4 of the peak at incl = 19 deg, 1e-2 at 60 deg and 5e-2 at 80 deg (VIP's sampling: 0.12, 0.11 and 0.14).
    """
    d = vip_disk
    dist = d.dust_density.dust_distribution_calc
    if getattr(dist, 'beta', 1.) != 1.:
        raise NotImplementedError("compute_scattered_light_wedge needs linear flaring (beta = 1); "
                                  "use ray_sampling='vip' for other beta.")
    d.check_inclination()
    c = _wedge_slope(dist)
    cosi, sini = d.cosi, d.sini

    # same pixel selection as VIP
    rho2_map = d.x_map**2 + d.y_map**2
    valid = (dist.rmax**2 - rho2_map > 0.) * (rho2_map - d.rmin**2 > 0.)
    x = d.x_map[valid]
    y = d.y_map[valid]
    ycs = cosi*y # y_disk = ycs + sini*l
    zsn = -sini*y # z_disk = zsn + cosi*l
    lz0 = y*np.tan(np.deg2rad(d.itilt)) # midplane crossing

    # limit r <= rmax (cylinder around the star; padded by the disk offset)
    rcyl = dist.rmax + np.hypot(d.xdo, d.ydo)
    if sini > 1e-12:
        half = np.sqrt(np.maximum(rcyl**2 - x**2, 0.))
        lc1, lc2 = (-ycs-half)/sini, (-ycs+half)/sini
    else:
        lc1, lc2 = np.full_like(x, -np.inf), np.full_like(x, np.inf)

    # envelope: (zsn + cosi*l)^2 <= c^2*((x-xdo)^2 + (ycs + sini*l - ydo)^2), i.e. A*l^2 + B*l + C <= 0
    A = cosi**2 - c**2*sini**2
    if A > 0:
        B = 2.*(zsn*cosi - c**2*sini*(ycs-d.ydo))
        C = zsn**2 - c**2*((x-d.xdo)**2 + (ycs-d.ydo)**2)
        sq = np.sqrt(np.maximum(B**2 - 4.*A*C, 0.))
        l1 = np.maximum((-B-sq)/(2.*A), lc1)
        l2 = np.minimum((-B+sq)/(2.*A), lc2)
    else:
        l1, l2 = lc1, lc2
    l2 = np.maximum(l2, l1)
    if lwidth is None:
        lwidth = 0. if c*np.tan(np.deg2rad(d.itilt)) < 0.5 else 10.
    lc = np.clip(lz0, l1, l2) # nodes are centred on the midplane crossing (if inside the limits)

    if lwidth > 0:
        tmp = (np.exp(np.arange(halfNbSlices)*np.log(lwidth+1.)/(halfNbSlices-1.))-1.)/lwidth
    else:
        tmp = np.linspace(0., 1., halfNbSlices)
    s_nodes = np.concatenate((-tmp[:0:-1], tmp)) # -1 ... 1

    acc = np.zeros_like(x)
    l_prev = f_prev = None
    for s in s_nodes:
        l = lc + s*((l2-lc) if s >= 0 else (lc-l1))
        yd = ycs + sini*l
        zd = zsn + cosi*l
        d2star = x**2 + yd**2 + zd**2
        dstar = np.sqrt(d2star)
        rstar = np.sqrt(x**2 + yd**2)
        thetastar = np.arctan2(yd, x)
        with np.errstate(invalid='ignore', divide='ignore'):
            cosphi = (rstar*sini*np.sin(thetastar) + zd*cosi)/dstar
            r = np.sqrt((x-d.xdo)**2 + (yd-d.ydo)**2)
            theta = np.arctan2(yd-d.ydo, x-d.xdo)
            rho = d.dust_density.density_cylindrical(r, np.cos(theta-np.deg2rad(d.omega)), zd)
            f = rho*d.phase_function.compute_phase_function_from_cosphi(cosphi)/d2star
        f = np.nan_to_num(f, nan=0., posinf=0., neginf=0.)
        if l_prev is not None:
            acc += (l-l_prev)*(f+f_prev)/2.
        l_prev, f_prev = l, f

    d.scattered_light_map.fill(0.)
    d.scattered_light_map[valid] = acc*d.pxInAU**2
    if d.flux_max is not None:
        d.scattered_light_map *= d.flux_max/np.nanmax(d.scattered_light_map)
    return d.scattered_light_map


def _compute_scattered_light(vip_disk, halfNbSlices, ray_sampling='wedge', ray_lwidth=None):
    """Runs VIP's ray tracing (ray_sampling='vip') or compute_scattered_light_wedge (ray_sampling='wedge')."""
    if ray_sampling == 'vip':
        return vip_disk.compute_scattered_light(halfNbSlices=halfNbSlices)
    elif ray_sampling == 'wedge':
        return compute_scattered_light_wedge(vip_disk, halfNbSlices=halfNbSlices, lwidth=ray_lwidth)
    raise ValueError(f"Unknown ray_sampling '{ray_sampling}'; use 'vip' or 'wedge'.")


def grater_2hg_disk_model(r0, h0, ain, aout, pa, incl, g1, g2, wg1,
                          e=0., omega=0., gamma=2., beta=1., xdo=0, ydo=0,
                          cent=np.array([160.,160.]), distance=10., nx=320, ny=320,
                          pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                          halfNbSlices=25, polar=False, flux_max=1.0, eps=2., ray_sampling='wedge', ray_lwidth=None,
                          zmax_nsigma=3.):
    """
    A simple ring-like disk morphology based on Augereau et al. (1999) and assuming a linear combo of two H-G SPFs as the scattering phase function.

    r0: fiducial radius in au
    h0: aspect ratio sigma_z/r at r0, where sigma_z is the vertical standard deviation of the dust density (for any
        gamma; VIP's own scale height is ksi0 = vip_scale_height(h0*r0, gamma), i.e. sqrt(2)*h0*r0 for gamma = 2)
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
    eps: smoothing exponent of the two power laws, rho_mid ∝ [(r/r0)^(-eps*ain) + (r/r0)^(-eps*aout)]^(-1/eps)
         (must be > 0). eps = 2 (default) is VIP's / GRaTer's prescription and runs VIP's own code unchanged;
         otherwise see DustEllipticalDistribution2PowerLawsEps.

    cent: the pixel position for the center of the disk (generally the location of the star in the data)
    distance: distance to the target in parsecs
    nx: number of x-axis pixels for the image
    ny: number of y-axis pixels for the image
    pxscale: the pixel scale for the data; either a float (must be arcsec/pixel) or astropy units (any units that can be cast to arcsec/pixel)
    accuracy: the numerical accuracy for the model; pixels with density below this value will be set to zero
    rmax_accuracy: if accuracy is None, the model's accuracy is set such that non-zero values are achieved to this separation (in au)
    halfNbSlices: the number of planar slices to compute above and below the disk midplane
    ray_sampling: 'wedge' (default) uses compute_scattered_light_wedge, which integrates each line of sight only where
        it is inside the flared disk (|z| <= zmax*r/r0), so the sampling follows the local scale height (needs
        beta = 1); 'vip' uses VIP's ray tracing, which integrates every line of sight over the same heights |z| <= 2*zmax
    ray_lwidth: node spacing for ray_sampling='wedge' (None: automatic; 0: uniform; > 0: geometric towards the midplane,
        as VIP with 100); see compute_scattered_light_wedge
    zmax_nsigma: the maximum height of the integration, zmax, in vertical standard deviations at r0 (default 3). With
        ray_sampling='wedge', each line of sight is integrated over |z| <= zmax_nsigma*H(r) at every radius (for a
        Gaussian vertical profile, 3 sigma leaves out 0.27% of the column); with ray_sampling='vip' over a slab
        |z| <= 2*zmax. None: zmax is set by accuracy, as in VIP (zmax = ksi0*(-ln(accuracy))^(1/gamma))
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
    dens = {'name': '2PowerLaws', 'ain': ain, 'aout': aout, 'a': r0, 'ksi0': vip_scale_height(h0*r0, gamma),
            'e': e, 'gamma': gamma, 'beta': beta, 'accuracy': accuracy}

    vip_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=(pxscale << u.arcsec/u.pixel).value,
                                  pa=pa-180., omega=omega, xdo=xdo, ydo=ydo,
                                  density_dico=dens, spf_dico=spf, flux_max=flux_max,
                                  xs=cent[0], ys=cent[1])

    vip_disk.dust_density.accuracy = accuracy
    vip_disk.dust_density.dust_distribution_calc.rmax = vip_disk.dust_density.dust_distribution_calc.a*accuracy**(1/vip_disk.dust_density.dust_distribution_calc.aout)
    if eps != 2:
        # keep VIP's density object (rmax, zmax, e, amin) and only replace its radial profile by the eps one
        vip_disk.dust_density.dust_distribution_calc.__class__ = DustEllipticalDistribution2PowerLawsEps
        vip_disk.dust_density.dust_distribution_calc.set_eps(eps)
    _set_zmax_nsigma(vip_disk.dust_density.dust_distribution_calc, zmax_nsigma, h0*r0, r0, r0)
    vip_disk.check_inclination = dont_actually_check_inclination
    disk = _compute_scattered_light(vip_disk, halfNbSlices, ray_sampling=ray_sampling, ray_lwidth=ray_lwidth)
    return fix_star_cross(disk, cent)


class DustEllipticalDistributionGaussian:
    """
    Gaussian ring dust density distribution, written as a drop-in replacement for VIP's
    DustEllipticalDistribution2PowerLaws (VIP's ScatteredLightDisk only uses density_cylindrical,
    rmax, zmax and itiltthreshold from it).

    The radial profile is a Gaussian in semi-major axis, a_eff = r*(1-e*cos(theta))/(1-e^2), centred on r0
    with the given FWHM. The vertical profile is the same as VIP's: exp(-(|z|/H)^gamma), with
    H = ksi0*(a_eff/r0)^beta (ksi0 is VIP's scale height, not the standard deviation; see vip_scale_height).

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
                                halfNbSlices=25, polar=False, flux_max=1.0, surface_density=True,
                                ray_sampling='wedge', ray_lwidth=None, zmax_nsigma=3.):
    """
    Same as grater_2hg_disk_model, but with a Gaussian radial density profile instead of two power laws.

    r0: ring centre (semi-major axis) in au
    fwhm: ring full width at half maximum in au
    h0: aspect ratio sigma_z/r at r0 (sigma_z: vertical standard deviation, see grater_2hg_disk_model)
    surface_density: if True, the surface density is Gaussian; if False, the midplane volume density is Gaussian
    accuracy: the density is truncated where the Gaussian drops below this fraction of its peak (default 1e-3, i.e. ~3.7 sigma)
    zmax_nsigma: maximum integration height in vertical standard deviations (here at the outer edge of the ring); see
        grater_2hg_disk_model
    rmax_accuracy: if not None, the integration is additionally truncated at this separation (in au)

    All other parameters as in grater_2hg_disk_model.
    """
    if accuracy is None:
        accuracy = 1e-3

    spf = {'name':'DoubleHG', 'g': [g1, g2], 'weight': wg1, 'polar': polar}
    dens = {'a': r0, 'fwhm': fwhm, 'ksi0': vip_scale_height(h0*r0, gamma), 'e': e, 'gamma': gamma, 'beta': beta,
            'surface_density': surface_density}

    # VIP only implements '2PowerLaws', so build the disk with a placeholder density and swap in the Gaussian one
    vip_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=(pxscale << u.arcsec/u.pixel).value,
                                  pa=pa-180., omega=omega, xdo=xdo, ydo=ydo,
                                  density_dico={'name': '2PowerLaws'}, spf_dico=spf, flux_max=flux_max,
                                  xs=cent[0], ys=cent[1])

    gauss_dens = DustEllipticalDistributionGaussian(accuracy, dens)
    _set_zmax_nsigma(gauss_dens, zmax_nsigma, h0*r0, r0, gauss_dens.amax) # at the outer edge, as with accuracy
    if rmax_accuracy is not None:
        gauss_dens.rmax = min(gauss_dens.rmax, rmax_accuracy)
    vip_disk.dust_density.accuracy = accuracy
    vip_disk.dust_density.type = 'Gaussian'
    vip_disk.dust_density.dust_distribution_calc = gauss_dens
    vip_disk.check_inclination = lambda: None # see dont_actually_check_inclination in grater_2hg_disk_model
    disk = _compute_scattered_light(vip_disk, halfNbSlices, ray_sampling=ray_sampling, ray_lwidth=ray_lwidth)
    return fix_star_cross(disk, cent)


def tanh_gaps_profile(r, rmin, rmax, lin, lout, gamma_sd, gaps=()):
    """
    Radial profile of the 'TanhGaps' model (normalised so that Sigma_c = 1):
        Sigma(r) = 1/4 (r/rmin)^gamma_sd (1+tanh[(r-rmin)/lin]) (1+tanh[(rmax-r)/lout]) * prod_k G_k(r),
        G_k(r) = 1 - d_k exp(-0.5((r-r_k)/sigma_k)^2), sigma_k = fwhm_k/(2 sqrt(2 ln2)),
    with gaps a sequence of (r_k, fwhm_k, d_k). r in the same units as rmin, rmax, lin, lout, r_k and fwhm_k.
    """
    r = np.asarray(r, dtype=float)
    S = 0.25*np.power(r/rmin, gamma_sd)*(1.+np.tanh((r-rmin)/lin))*(1.+np.tanh((rmax-r)/lout))
    for rg, fwhmg, dg in gaps:
        sigg = fwhmg/(2.*np.sqrt(2.*np.log(2.)))
        S = S*(1.-dg*np.exp(-0.5*((r-rg)/sigg)**2))
    return S


def gap_params_to_list(gap_params):
    """Converts {'r_gap1':..., 'fwhm_gap1':..., 'd_gap1':..., 'r_gap2':...} into [(r_gap1, fwhm_gap1, d_gap1), ...]."""
    gaps = []
    k = 1
    while f'r_gap{k}' in gap_params:
        gaps.append((gap_params[f'r_gap{k}'], gap_params[f'fwhm_gap{k}'], gap_params[f'd_gap{k}']))
        k += 1
    return gaps


class DustEllipticalDistributionTanhGaps:
    """
    Power-law disk with tanh inner and outer edges and Gaussian gaps (see tanh_gaps_profile), written as a drop-in
    replacement for VIP's DustEllipticalDistribution2PowerLaws (as DustEllipticalDistributionGaussian).

    The radial profile is evaluated at the semi-major axis a_eff = r*(1-e*cos(theta))/(1-e^2). The vertical profile
    is exp(-(|z|/H)^gamma), with H = ksi0*(a_eff/rmin)^beta (ksi0 is VIP's scale height at rmin, not the standard
    deviation; see vip_scale_height).
    If surface_density is True (default), the profile describes the surface density (the midplane density is divided
    by H/ksi0); if False, it describes the midplane volume density.
    """

    def __init__(self, accuracy=1.e-3, density_dico={}):
        self.accuracy = accuracy
        self.set_density_distribution(density_dico)

    def set_density_distribution(self, density_dico):
        self.rmin = float(density_dico.get('rmin', 30.))
        self.rmax_edge = float(density_dico.get('rmax', 80.))
        self.lin = float(density_dico.get('lin', 5.))
        self.lout = float(density_dico.get('lout', 5.))
        self.gamma_sd = float(density_dico.get('gamma_sd', 0.))
        self.gaps = list(density_dico.get('gaps', []))
        self.e = float(np.clip(density_dico.get('e', 0.), 0., 0.99))
        self.ksi0 = float(density_dico.get('ksi0', 1.))
        self.gamma = float(density_dico.get('gamma', 2.))
        self.beta = float(density_dico.get('beta', 1.))
        self.surface_density = bool(density_dico.get('surface_density', True))

        # Far from an edge, (1+tanh(x))/2 ~ exp(2x), so each edge drops below 'accuracy' at 0.5*ln(1/accuracy) l
        nl = 0.5*np.log(1./self.accuracy)
        self.amin = max(self.rmin - nl*self.lin, 0.)
        self.amax = self.rmax_edge + nl*self.lout
        # maximum distance from the star (apocentre of the outer edge) for the integration, AU
        self.rmax = self.amax*(1.+self.e)
        # maximum height for the integration, AU (evaluated at amax so the flared outer edge isn't truncated)
        self.zmax = self.ksi0*(self.amax/self.rmin)**self.beta*(-np.log(self.accuracy))**(1./self.gamma)
        self.itiltthreshold = np.rad2deg(np.arctan(self.rmax/self.zmax))

    def density_cylindrical(self, r, costheta, z):
        """Return the particle volume density at r, theta, z."""
        a_eff = r*(1.-self.e*costheta)/(1.-self.e**2)
        # avoid dividing by zero (and diverging power laws) for points passing (almost) through the star
        a_safe = np.maximum(a_eff, max(self.amin, 1e-3*self.rmin))
        radial_term = np.where(a_eff >= self.amin,
                               tanh_gaps_profile(a_safe, self.rmin, self.rmax_edge, self.lin, self.lout,
                                                 self.gamma_sd, self.gaps), 0.)
        h_ratio = np.power(a_safe/self.rmin, self.beta)
        if self.surface_density:
            radial_term = radial_term/h_ratio
        vertical_term = np.exp(-np.power(np.abs(z)/(self.ksi0*h_ratio), self.gamma))
        return radial_term*vertical_term

    def density_cartesian(self, x, y, z):
        r = np.sqrt(x**2+y**2)
        costheta = 0 if r == 0 else x/r
        return self.density_cylindrical(r, costheta, z)

    def print_info(self, pxInAu=None):
        print('Type of structure: power law with tanh edges and Gaussian gaps')
        print('rmin: {0:.2f} au (lin={1:.2f} au), rmax: {2:.2f} au (lout={3:.2f} au), gamma: {4:.2f}'.format(
            self.rmin, self.lin, self.rmax_edge, self.lout, self.gamma_sd))
        for k, (rg, fwhmg, dg) in enumerate(self.gaps):
            print('Gap {0}: r={1:.2f} au, FWHM={2:.2f} au, depth={3:.2f}'.format(k+1, rg, fwhmg, dg))
        print('Eccentricity: {0:.2f}'.format(self.e))
        print('Scale height: {0:.2f} au at {1:.2f} au'.format(self.ksi0, self.rmin))
        print('Maximum radius for integration: {0:.2f} au'.format(self.rmax))
        print('Maximum height for integration: {0:.2f} au'.format(self.zmax))


def grater_tanhgaps_2hg_disk_model(rmin, rmax, lin, lout, gamma_sd, h0, pa, incl, g1, g2, wg1,
                                   e=0., omega=0., gamma=2., beta=1., xdo=0, ydo=0,
                                   cent=np.array([160.,160.]), distance=10., nx=320, ny=320,
                                   pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                                   halfNbSlices=25, polar=False, flux_max=1.0, surface_density=True,
                                   ray_sampling='wedge', ray_lwidth=None, zmax_nsigma=3.,
                                   **gap_params):
    """
    Same as grater_2hg_disk_model, but with the radial profile of tanh_gaps_profile: a power law (exponent
    gamma_sd) between rmin and rmax with tanh edges of widths lin and lout (all in au), multiplied by Gaussian gaps.

    gap_params: r_gap1, fwhm_gap1, d_gap1, r_gap2, ... (centre and FWHM in au, fractional depth 0-1) for each gap.
    h0: aspect ratio sigma_z/r at rmin (sigma_z: vertical standard deviation, see grater_2hg_disk_model)
    surface_density: if True, the profile is the surface density; if False, the midplane volume density
    accuracy: the density is truncated where the edges drop below this fraction (default 1e-3)
    zmax_nsigma: maximum integration height in vertical standard deviations (here at the outer edge); see
        grater_2hg_disk_model
    rmax_accuracy: if not None, the integration is additionally truncated at this separation (in au)

    All other parameters as in grater_2hg_disk_model.
    """
    if accuracy is None:
        accuracy = 1e-3

    spf = {'name':'DoubleHG', 'g': [g1, g2], 'weight': wg1, 'polar': polar}
    dens = {'rmin': rmin, 'rmax': rmax, 'lin': lin, 'lout': lout, 'gamma_sd': gamma_sd,
            'gaps': gap_params_to_list(gap_params), 'ksi0': vip_scale_height(h0*rmin, gamma), 'e': e, 'gamma': gamma, 'beta': beta,
            'surface_density': surface_density}

    # VIP only implements '2PowerLaws', so build the disk with a placeholder density and swap in ours
    vip_disk = ScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=(pxscale << u.arcsec/u.pixel).value,
                                  pa=pa-180., omega=omega, xdo=xdo, ydo=ydo,
                                  density_dico={'name': '2PowerLaws'}, spf_dico=spf, flux_max=flux_max,
                                  xs=cent[0], ys=cent[1])

    tanh_dens = DustEllipticalDistributionTanhGaps(accuracy, dens)
    _set_zmax_nsigma(tanh_dens, zmax_nsigma, h0*rmin, rmin, tanh_dens.amax) # at the outer edge, as with accuracy
    if rmax_accuracy is not None:
        tanh_dens.rmax = min(tanh_dens.rmax, rmax_accuracy)
    vip_disk.dust_density.accuracy = accuracy
    vip_disk.dust_density.type = 'TanhGaps'
    vip_disk.dust_density.dust_distribution_calc = tanh_dens
    vip_disk.check_inclination = lambda: None # see dont_actually_check_inclination in grater_2hg_disk_model
    disk = _compute_scattered_light(vip_disk, halfNbSlices, ray_sampling=ray_sampling, ray_lwidth=ray_lwidth)
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
                                 ray_sampling='wedge', ray_lwidth=None, zmax_nsigma=3.,
                                 **disk_params):
    """
    ring_model: '2PowerLaws' (each ring needs r0_i, ain_i, aout_i), '2PowerLawsEps' (same, plus the smoothing
        exponent of the two power laws eps_i, see grater_2hg_disk_model; a ring with no eps_i uses eps = 2, i.e.
        the GRaTer model) or 'Gaussian' (each ring needs r0_i, fwhm_i).
    ray_sampling, ray_lwidth, zmax_nsigma: see grater_2hg_disk_model.
    """
    if ring_model in ('2PowerLaws', '2PowerLawsEps'):
        ring_fn = grater_2hg_disk_model
    elif ring_model == 'Gaussian':
        ring_fn = grater_gauss_2hg_disk_model
    else:
        raise ValueError(f"Unknown ring_model '{ring_model}'; use '2PowerLaws', '2PowerLawsEps' or 'Gaussian'.")

    shared_params = list(shared_params) if shared_params is not None else []

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
            if ring_model != '2PowerLawsEps' and 'eps' in ring_params:
                raise ValueError(f"eps (power-law smoothing) needs ring_model='2PowerLawsEps', not '{ring_model}'.")
            # obtain the flux scaling for this ring (default to 1 if not specified) and add the appropriately scaled ring model to the composite image
            F_i = ring_params.pop('F', 1)
            composite_image += F_i * ring_fn(**ring_params, cent=cent,
                                                            distance=distance, nx=nx, ny=ny,
                                                            pxscale=pxscale, accuracy=accuracy, 
                                                            rmax_accuracy=rmax_accuracy,
                                                            polar=polar, halfNbSlices=halfNbSlices, flux_max=1.0,
                                                            ray_sampling=ray_sampling, ray_lwidth=ray_lwidth,
                                                            zmax_nsigma=zmax_nsigma)
        i+=1
    return composite_image



def lnprob_Nrings(parameters, rdi_reduc, roi, distance, pix_sr, crop_size=None, h0=0.05, gamma=2.0, err=None, wdb=None, shared_params=None, prior_ranges=None, halfNbSlices=4, return_soln=False, plot_model=False, print_chisq=False, backscattering=False, ring_model='2PowerLaws', ref_ring=1,
                  ray_sampling='wedge', zmax_nsigma=3.):
    """
    parameters: [F, pa, incl, g1, (g2, wg1 if backscattering)], then for each ring i: [F_i, r0_i, <ring params>_i],
        except the reference ring (ref_ring, default 1), whose F is fixed to 1 and left out: [r0_ref, <ring params>_ref].
        <ring params> are (ain, aout) for ring_model='2PowerLaws', (ain, aout, eps) for ring_model='2PowerLawsEps'
        or (fwhm,) for ring_model='Gaussian'. For ring_model='TanhGaps': [F, pa, incl, g1, (g2, wg1)], [rmin, rmax, lin, lout, gamma_sd], then
        [r_gapk, fwhm_gapk, d_gapk] for each gap (shared_params and ref_ring are not used).
        See nring_parameters_to_pdict.

    ref_ring: int
        The ring used as flux reference (F=1); the other rings' F are relative to it. Default 1.

    halfNbSlices, ray_sampling, zmax_nsigma: ray tracing of the raw model, see grater_2hg_disk_model. With the default
        ray_sampling='wedge' and zmax_nsigma=3, halfNbSlices=4 gives errors of ~1e-3 of the peak for low inclinations.

    ring_model: str
        '2PowerLaws', '2PowerLawsEps', 'Gaussian' or 'TanhGaps'. With '2PowerLawsEps', each ring also has its own
        smoothing exponent eps_i of the two power laws (see grater_2hg_disk_model; eps = 2 is '2PowerLaws'). Set its
        prior with prior_ranges, e.g. {'eps_1': (0.5, 20.)}; eps_i <= 0 always returns -inf.

    wdb: winnie SpaceRDI object
        The forward-modeled RDI uses winnie's cached reference matrices (wdb._prepared_refmats), which a standard
        (forward_model=False) wdb.run_rdi builds for the crop, zones and RDI presets current at that time. Run one
        after changing any of these and before fitting (rdi_reduc is normally that run).

    raw_model:
        your idealized / unblurred disk model image oriented north-up.

    raw_model_pxscale:
        the pixelscale used for raw_model in arcsec/px (or any equivalent astropy units)

    raw_model_center:
        the position where the star would be in the model image (zero indexed and as [x,y])
    """
    if wdb is None:
        import winnie
        wdb = winnie.SpaceRDI(from_fits=rdi_reduc.filename)
    # else:
    #     wdbi=copy.deepcopy(wdb)  # Create a copy of the SpaceRDI object to avoid modifying the original one
    if err is None:
        err = 1
    
    if crop_size is not None:
        wdb.set_crop(crop_size)
    # y1,y2,x1,x2 = wdb._crop_indices

    ### extract disk parameters (shared parameters, then per-ring parameters)
    pdict, Nrings = nring_parameters_to_pdict(parameters, h0=h0, gamma=gamma, backscattering=backscattering, ring_model=ring_model, ref_ring=ref_ring)
    Fi = pdict['flux'] # mJy

    # apply priors
    if priors(pdict, prior_ranges=prior_ranges):
        return -np.inf  # Return negative infinity if any parameter is outside its prior range
    if ring_model == '2PowerLawsEps' and any(pdict[f'eps_{i+1}'] <= 0 for i in range(Nrings)):
        return -np.inf  # the power-law smoothing exponents must be positive

    if ring_model == 'TanhGaps':
        # require rmin < rmax, gaps ordered by increasing radius (Nrings is the number of gaps here)
        # and gaps narrower than their radius (fwhm_gap/r_gap < 1)
        r_gaps = np.array([pdict[f'r_gap{k+1}'] for k in range(Nrings)])
        fwhm_gaps = np.array([pdict[f'fwhm_gap{k+1}'] for k in range(Nrings)])
        if pdict['rmin'] >= pdict['rmax'] or np.any(np.diff(r_gaps) <= 0) or np.any(fwhm_gaps >= r_gaps):
            return -np.inf
    else:
        # require rings to be ordered by increasing r0 (avoids label switching between rings)
        r0s = [pdict[f'r0_{i+1}'] for i in range(Nrings)]
        if np.any(np.diff(r0s) <= 0):
            return -np.inf
    
    # Generate the raw model oriented north-up
    # We generate a detector-sampled raw model, and then subpixelate it before PSF convolution
    # You could alter this to generate an oversampled model quite easily, but in my experience the
    # only difference is runtime.
    if ring_model == 'TanhGaps':
        raw_model = grater_tanhgaps_2hg_disk_model(**{k: v for k, v in pdict.items() if k != 'flux'},
                                                   cent=wdb.c_star, distance=distance,
                                                   nx=wdb.nx, ny=wdb.ny, pxscale=wdb.pxscale,
                                                   rmax_accuracy=None, halfNbSlices=halfNbSlices,
                                                   ray_sampling=ray_sampling, zmax_nsigma=zmax_nsigma)
    else:
        raw_model = grater_nring_2hg_disk_model(**pdict, cent=wdb.c_star, distance=distance,
                                                nx=wdb.nx, ny=wdb.ny, pxscale=wdb.pxscale,
                                                rmax_accuracy=None, halfNbSlices=halfNbSlices, shared_params=shared_params,
                                                ring_model=ring_model, ray_sampling=ray_sampling, zmax_nsigma=zmax_nsigma)
    # normalize raw_model to have an integrated flux of F_i
    
    # print raw flux
    # print(f"Raw model flux: {np.sum(raw_model)*pix_sr*1e6:.3e} Jy")

    raw_model /= np.sum(raw_model)*pix_sr
    raw_model *= (Fi/1.0e9)  # Convert F_i from mJy to MJy for normalization

    # print raw flux after normalization
    # print(f"Raw model flux after normalization: {np.sum(raw_model)*pix_sr*1e6:.3e} Jy")

    

    # Convolve the model using the prepared SpaceConvolution object then set it as the CSS model in the SpaceRDI object
    wdb.set_circumstellar_model(raw_model=raw_model, raw_model_pxscale=wdb.pxscale, raw_model_center=wdb.c_star)

    # Run RDI on the model to get the forward-modeled result (uses winnie's cached reference matrices)
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


def nring_parameters_to_pdict(parameters, h0=0.05, gamma=2.0, backscattering=False, ring_model='2PowerLaws', ref_ring=1):
    """
    Converts the parameter vector used by lnprob_Nrings into a dictionary of disk parameters. Returns (pdict, Nrings).

    Parameter order: [flux, pa, incl, g1, (g2, wg1 if backscattering)], then for each ring i = 1..N in order of
    increasing radius: [F_i, r0_i, <ring params>_i], except for the reference ring (ref_ring), whose F is fixed to 1
    and is therefore left out: [r0_ref, <ring params>_ref]. <ring params> are (ain, aout) for ring_model='2PowerLaws',
    (ain, aout, eps) for ring_model='2PowerLawsEps' or (fwhm,) for ring_model='Gaussian'. With ref_ring=1 (default) the order is
    [..., r0_1, <ring params>_1, F_2, r0_2, <ring params>_2, F_3, ...].

    For ring_model='TanhGaps' (a single power-law disk with tanh edges and Gaussian gaps, see tanh_gaps_profile),
    the shared parameters are followed by [rmin, rmax, lin, lout, gamma_sd] and then [r_gapk, fwhm_gapk, d_gapk] for
    each gap k = 1..Ngaps (ref_ring is ignored). Returns (pdict, Ngaps).
    """
    Fi, pai, incli, g1i = parameters[0:4]
    if backscattering:
        g2i, wg1i = parameters[4:6]
        Nshared = 6
    else:
        g2i = 0
        wg1i = 1.0
        Nshared = 4

    if ring_model == 'TanhGaps':
        # a single disk [rmin, rmax, lin, lout, gamma_sd] followed by [r_gapk, fwhm_gapk, d_gapk] for each gap
        disk_keys = ['rmin', 'rmax', 'lin', 'lout', 'gamma_sd']
        gap_keys = ['r_gap', 'fwhm_gap', 'd_gap']
        Ngaps = (len(parameters) - Nshared - len(disk_keys)) / len(gap_keys)
        if Ngaps < 0 or not Ngaps.is_integer():
            raise ValueError(f"Wrong number of parameters for ring_model='TanhGaps': the disk needs "
                             f"{', '.join(disk_keys)}, and each gap {', '.join(gap_keys)}.")
        Ngaps = int(Ngaps)
        pdict = {'flux': Fi, 'pa': pai, 'incl': incli, 'h0': h0, 'gamma': gamma,
                 'g1': g1i, 'g2': g2i, 'wg1': wg1i}
        pdict.update(zip(disk_keys, parameters[Nshared:Nshared+len(disk_keys)]))
        j = Nshared+len(disk_keys)
        for k in range(1, Ngaps+1):
            for key, val in zip(gap_keys, parameters[j:j+len(gap_keys)]):
                pdict[f'{key}{k}'] = val
            j += len(gap_keys)
        return pdict, Ngaps

    ring_keys_by_model = {'2PowerLaws': ['ain', 'aout'], '2PowerLawsEps': ['ain', 'aout', 'eps'], 'Gaussian': ['fwhm']}
    if ring_model not in ring_keys_by_model:
        raise ValueError(f"Unknown ring_model '{ring_model}'; use {', '.join(map(repr, ring_keys_by_model))} or 'TanhGaps'.")
    ring_keys = ring_keys_by_model[ring_model]

    # N rings take N*(1+len(ring_keys)) + (N-1) parameters (every ring but the reference one has an F)
    Nrings = (len(parameters) - Nshared + 1) / (2 + len(ring_keys))
    if not Nrings.is_integer():
        raise ValueError(f"Wrong number of ring parameters: each ring needs {', '.join(['F', 'r0']+ring_keys)}, "
                         f"except the reference ring (ref_ring={ref_ring}), which has no F.")
    Nrings = int(Nrings)
    if not 1 <= ref_ring <= Nrings:
        raise ValueError(f"ref_ring={ref_ring} but the parameters describe {Nrings} ring(s).")

    pdict = {'flux': Fi, 'pa': pai, 'incl': incli, 'h0': h0, 'gamma': gamma,
             'g1': g1i, 'g2': g2i, 'wg1': wg1i}
    j = Nshared
    for i in range(1, Nrings+1):
        keys = ['r0']+ring_keys if i == ref_ring else ['F', 'r0']+ring_keys
        for key, val in zip(keys, parameters[j:j+len(keys)]):
            pdict[f'{key}_{i}'] = val
        if i == ref_ring:
            pdict[f'F_{i}'] = 1.0
        j += len(keys)
    return pdict, Nrings


def set_reference_ring(p, ref_ring):
    """
    For lmfit Parameters used with obj_fn: makes ring 'ref_ring' the flux reference by fixing F_<ref_ring> = 1
    (vary=False) and letting the F of every other ring vary. Returns p (modified in place).
    """
    Fkeys = sorted([k for k in p if k.startswith('F_') and k[2:].isdigit()], key=lambda k: int(k[2:]))
    if f'F_{ref_ring}' not in p:
        raise ValueError(f"p has no F_{ref_ring}; rings found: {', '.join(Fkeys)}.")
    for k in Fkeys:
        if k == f'F_{ref_ring}':
            p[k].set(value=1.0, vary=False)
        else:
            p[k].set(vary=True)
    return p


## Parallel sampling (e.g. emcee with a pool of worker processes)
# Each worker must have its own copy of the SpaceRDI object: lnprob_Nrings modifies it on every call
# (set_crop, set_circumstellar_model), so walkers sharing one object overwrite each other's model.
# Passing wdb as an argument to emcee does not work well either: the pool pickles the arguments for every batch of
# walkers. Instead, the arguments are stored at module level and inherited by worker processes created with the
# 'fork' start method, so each worker gets a private copy once, when the pool starts, and the function sent to the
# pool (lnprob_Nrings_parallel) is cheap to pickle.

_parallel_state = {}


def set_parallel_lnprob(**kwargs):
    """
    Stores the arguments of lnprob_Nrings (everything except 'parameters', given as keywords, e.g.
    rdi_reduc=..., roi=..., distance=..., pix_sr=..., crop_size=..., wdb=..., ...) for use by
    lnprob_Nrings_parallel. Must be called before make_parallel_pool, since workers copy the state when created.
    """
    _parallel_state.clear()
    _parallel_state.update(kwargs)


def lnprob_Nrings_parallel(parameters):
    """lnprob_Nrings(parameters, **state), with state set by set_parallel_lnprob."""
    return lnprob_Nrings(parameters, **_parallel_state)


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
           ray_sampling='wedge', zmax_nsigma=3.):
    """
    Parameters
    ----------
    p: lmfit.parameter.Parameters
        LMFit parameters object containing at least: r0_1, h0_1, ain_1, aout_1, pa_1, incl_1, g1_1, g2_1, wg1_1
        (or r0_1, fwhm_1, ... instead of ain_1, aout_1 if ring_model='Gaussian')
        If p contains 'flux' (total disk flux in mJy), the raw model is normalized to that flux (as in lnprob_Nrings)
        and the flux is fitted like any other parameter; the least-squares rescaling (lsq_sfac) is then not used
        and sfac = 1. The F_i of the rings then only set their relative fluxes, so one of them must be fixed.
        With ring_model='2PowerLawsEps', p can also contain eps_i (e.g. p.add('eps_1', 2., min=0.5, max=20.)), the
        smoothing exponent of the two power laws of ring i (must be > 0, see grater_2hg_disk_model). A ring without
        eps_i uses eps = 2, i.e. the GRaTer model.

    ring_model: str
        Radial profile of each ring: '2PowerLaws', '2PowerLawsEps' or 'Gaussian'. See grater_nring_2hg_disk_model.

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

    ray_sampling, zmax_nsigma: ray tracing of the raw model, see grater_2hg_disk_model (default 'wedge' and 3;
        halfNbSlices=4 is then enough for low inclinations).
        
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
        This should nearly always be True, unless the flux is fitted as a parameter ('flux' in p), in which case
        it is ignored (no rescaling).
        
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

    # Total disk flux (mJy) fitted as a parameter: normalize the raw model to it below instead of rescaling the
    # forward model to the data by least squares
    flux = pdict.pop('flux', None)
    if flux is not None:
        lsq_sfac = False

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
                                            ring_model=ring_model, ray_sampling=ray_sampling, zmax_nsigma=zmax_nsigma)
    if flux is not None:
        # normalize the raw model (MJy/sr) to an integrated flux of 'flux' mJy, as in lnprob_Nrings
        pix_sr = ((wdb.pxscale << u.rad/u.pixel).value)**2
        raw_model /= np.sum(raw_model)*pix_sr
        raw_model *= flux/1.0e9 # mJy -> MJy

    # Convolve the model using the prepared SpaceConvolution object then set it as the CSS model in the SpaceRDI object
    wdb.set_circumstellar_model(raw_model=raw_model, raw_model_pxscale=wdb.pxscale, raw_model_center=wdb.c_star)


    # Run RDI on the model to get the forward-modeled result
    # collapse_rolls must always be True here since fmrdi_reduc.rolls is required below
    # regardless of return_soln (tying it to return_soln left both im and rolls as None
    # whenever rdi_reduc.derotated is False, causing SpaceReduction to try loading a
    # nonexistent saved file instead of building the result in memory).
    # The forward-modeled RDI uses winnie's cached reference matrices, built by the last standard (forward_model=False)
    # wdb.run_rdi, so run one after changing the crop, zones or RDI presets of wdb.
    fmrdi_reduc = wdb.run_rdi(forward_model=True, save_products=False, collapse_rolls=True, derotate=rdi_reduc.derotated, correct_distortion=rdi_reduc.distortion_corrected)
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
        sample = list(p.valuesdict().values()) # all parameters, including 'flux' (popped from pdict above)
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



def get_jwst_mag(sed, spec_type, instrument, filt, detector, exp_type):
    """
    Get the JWST magnitude of a star given its SED and spectral type.

    Parameters
    ----------
    sed : str
        Path to the stellar SED file (passed to starphot.get_stellar_magnitudes as starfile).
    spec_type : str
        Stellar spectral type.
    instrument : str
        JWST instrument in use, 'NIRCAM' or 'MIRI'.
    filt : str
        Filter name (e.g. 'F444W').
    detector : str
        Detector name (e.g. 'NRCALONG', 'NRCA1', 'MIRIMAGE').
    exp_type : str
        Exposure type from the FITS header (e.g. 'NRC_CORON', 'NRC_IMAGE',
        'MIR_4QPM', 'MIR_LYOT', 'MIR_IMAGE').

    Returns
    -------
    float
        Stellar Vega magnitude in the requested filter.
    """
    # Stellar magnitudes (vegamag) in each filter of the JWST instrument in use.
    primary_vegamag, _, _, _ = starphot.get_stellar_magnitudes(sed, spectral_type=spec_type, instrume=instrument,
                                                               detector=detector, exp_type=exp_type)
    # Return filter-specific magnitude
    return primary_vegamag[filt]


def format_SED_calcon(json_sed):
    """
    Format an SED from sdb (JSON) to use in spaceKLIP functions.

    Writes a two-column text file (wavelength [micron], flux [Jy]) next to the
    input file, with the .json extension replaced by .txt.

    Parameters
    ----------
    json_sed : str
        Path to the sdb SED JSON file.

    Returns
    -------
    str
        Path to the output text file.
    """
    with open(json_sed) as f:
        sed_data = json.load(f)
    w = sed_data['star_spec']['wavelength']  # microns
    flux = sed_data['star_spec']['fnujy']  # Jy

    file_path = json_sed.replace('.json', '.txt')
    np.savetxt(file_path, np.column_stack([w, flux]))
    return file_path
