import numpy as np
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft
import matplotlib.colors as cl
import os,sys
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib import rc
import matplotlib.pyplot as plt
import copy
import colorsys
from astropy.coordinates import SkyCoord, Distance, CartesianRepresentation, Angle
import astropy.units as u
from scipy.optimize import fsolve

    

G=6.67384e-11 # mks
M_sun= 1.9891e30 # kg
au=1.496e11 # m
c_light=2.99792458e8 # m/s
Kb=1.38064852e-23
h=6.62607004e-34 # mks

########################################
###### FUNCTIONS FOR RADIAL PROFILE ####
########################################

def stretching(phi, inc):
    """
    phi and inc in rad
    """
    return np.sqrt(np.cos(phi) ** 2 + np.sin(phi) ** 2 / np.cos(inc) ** 2)

def find_phic(inc, f):
    """
    find critical phi (phic) when the ratio a/l(phi) = f (f>1).
    inc in radians.
    f>1. 
    """
    
    if 1 / np.cos(inc) < f:
        print('All phis satisfy condition, using phic=pi/2')
        return np.pi / 2 
              
    # solve equation stretching-f=0
    func = lambda phi: stretching(phi, inc) - f
    phic = fsolve(func, np.pi / 4) # initial guess phic=45deg
    return phic[0]

def dsdx(x,a,b): # ds/dx over an ellipse
    
    return np.sqrt(1.  +  (b**2/a**4) *  x**2./(1.-(x/a)**2))

def Delta_s(xs, a,b): # delta s over an ellipse

    ys=np.zeros(len(xs))
    mask=((xs <= a ) & (xs>=-a)) # to avoid imaginary numbers 

    if xs[-1]<xs[0]:
        ys[mask]=b*np.sqrt(1.-xs[mask]**2.0/a**2)
    else:
        ys[mask]=-b*np.sqrt(1.-xs[mask]**2.0/a**2)
    return np.sqrt( (ys[:-1]-ys[1:])**2.0 + (xs[:-1]-xs[1:])**2.0 )

def simple_phi(phi): # returns phi between 0 and 2pi
    if abs(phi)!=2.0*np.pi:
        phir=phi%(2.0*np.pi)
    else: 
        phir=phi
    if phir<0.0:
        phir=2.0*np.pi+phir
    return phir

def x_phi(phi, a, b):
    
    phi=simple_phi(phi)
    
    if phi<=np.pi/2. or phi>=3.*np.pi/2.0:
        sign=1.0
    else:
        sign=-1.0
    return sign/np.sqrt(np.tan(phi)**2.0/b**2.0 + 1./a**2.)

def y_phi(phi, a, b):
    
    phi=simple_phi(phi)
    
    if phi>=0.0 and phi<=np.pi:
        sign=1.0
    else:
        sign=-1.0
    
    return sign/np.sqrt(1./b**2.0 + 1./(a**2.*np.tan(phi)**2.0))


def separate_intervals(phi1, phi2):
    # Figure out the right ranges of xs' to integrate. Does phi=0.0 pr phi=180 is contained in range?
    phis=[phi1]
    if phi1>phi2: # passes through zero
        # we need to figure out if passes through 180 first
        if phi1<np.pi:
            phis.append(np.pi)
        phis.append(0.0)
    if phi2>np.pi and phis[-1]<np.pi:
        phis.append(np.pi)
    phis.append(phi2)
    return np.array(phis)



def arc_length(a,b,phi1, phi2,  Nint=1000000):
    # a is semi-major axis (=1)
    # b is the semi-minor axis (=1/aspect_ratio)
    # returns arc length from phi1 to phi2 which are defined as 0 at the disc PA and grow anticlockwise in the sky

    # works with delta y / delta x instead of dy/dx
   

    # translates phi1 and phi2 to angles between 0 and 2pi
    phi1=simple_phi(phi1)
    phi2=simple_phi(phi2)
    #print(phi1,phi2)
    if phi2==phi1: # catches error when phi1 and phi2 are the same if one wants to integrate over whole circumpherence
        phi1=0.0
        phi2=2.*np.pi

    # Figure out the right ranges of xs' to integrate. Is phi=0.0 or phi=180 contained in range?
    phis=separate_intervals(phi1, phi2)
    
    Nph=len(phis)
    Arc_length=0.0

    for i in range(Nph-1):

        # if phis[i]==0.0:
        #     phi_i=1./Nint
        # elif phis[i]==np.pi:
        #     phi_i=np.pi+1./Nint
        # else:
        phi_i=phis[i]
            
        # if phis[i+1]==0.0:
        #     phi_ii=2.0*np.pi-1./Nint
        # elif phis[i+1]==np.pi:
        #     phi_ii=np.pi-1./Nint
        # else:
        phi_ii=phis[i+1]  
 
        x1=x_phi(phi_i,a,b)
        x2=x_phi(phi_ii, a,b)
        dx=(x2-x1)/(Nint-1)
        xs=np.arange(x1,x2,dx)

        Arc_length+=np.sum(Delta_s(xs, a,b))
        
    return Arc_length, phis




def ellipse(x0,y0,phi,chi,a, PA):

    ## phi is a position angle in the plane of the sky
    
    # x0,y0 ellipse center
    # phi pa at which calculate x,y phi=0 is +y axis
    # chi aspect ratio of ellipse with chi>1
    # a semi-major axis
    # a/chi semi-minor axis
    # PA  pa of ellipse 0 is north and pi/2 is east

    phipp= phi-PA
    
    
    xpp=x_phi(np.pi/2.0-phipp, a/chi, a)
    ypp=y_phi(np.pi/2.0-phipp, a/chi, a)
    #xpp = (a/chi) * np.sin(phipp) 
    #ypp =    a    * np.cos(phipp)

    xp =  xpp*np.cos(PA) + ypp*np.sin(PA)
    yp = -xpp*np.sin(PA) + ypp*np.cos(PA)
    
    xc = xp + x0
    yc = yp + y0
    
    return xc , yc
 
def radial_profile(image, image_pb=None, x0=0., y0=0., PA=0., inc=0., rmax=5,Nr=100, Nphi=100, rms=0., BMAJ_arcsec=1., ps_arcsec=0.05, error_std=False, arc='elipse', rmin=0., wedge_width=0.):

    # image is a 2D image to use to extract the radial profile
    # image_pb is an image of the primary beam (optional)
    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)
    # rms [same units as image] is the image rms.
    # BMAJ_arcsec [arcsec] is the beam major axis
    # ps_arcsec [arcsec] is the pixel size in arcsec
    
    # ################ SPATIAL GRID
    
    # calculate critical PA
    if wedge_width>0.:
        phic_deg=wedge_width/2.
    else:
        phic=find_phic(inc*np.pi/180., 1.3)
        phic_deg=phic*180./np.pi

    phis_E=np.linspace(PA-phic_deg, PA+phic_deg, Nphi)
    phis_W=phis_E+180.


    Ir_E=radial_profile_wedge(image,
                              image_pb=image_pb, 
                              x0=x0,
                              y0=y0,
                              PA=PA,
                              inc=inc,
                              rmax=rmax,
                              Nr=Nr, 
                              phis=phis_E, 
                              rms=rms, 
                              BMAJ_arcsec=BMAJ_arcsec,
                              ps_arcsec=ps_arcsec,
                              error_std=error_std,
                              rmin=rmin)

    Ir_W=radial_profile_wedge(image,
                              image_pb=image_pb, 
                              x0=x0,
                              y0=y0,
                              PA=PA,
                              inc=inc,
                              rmax=rmax,
                              Nr=Nr, 
                              phis=phis_W, 
                              rms=rms, 
                              BMAJ_arcsec=BMAJ_arcsec,
                              ps_arcsec=ps_arcsec,
                              error_std=error_std,
                              rmin=rmin)


    r, I, I_err = Ir_E[0], np.mean((Ir_E[1], Ir_W[1]), axis=0), np.hypot(Ir_E[2], Ir_W[2]) / 2

    
    return np.array([r, I, I_err]) 

def radial_profile_wedge(image, image_pb=None, x0=0., y0=0., PA=0., inc=0., rmax=5,Nr=100, phis=np.linspace(0., 360., 100), rms=0., BMAJ_arcsec=1., ps_arcsec=0.05, error_std=False, arc='elipse', rmin=0.):

    # image is a 2D image to use to extract the radial profile
    # image_pb is an image of the primary beam (optional)
    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)
    # rms [same units as image] is the image rms.
    # BMAJ_arcsec [arcsec] is the beam major axis
    # ps_arcsec [arcsec] is the pixel size in arcsec
    
    # ################ SPATIAL GRID
    
    Np=len(image[:,0])
    
    # check if image_pb is an array:
    if not hasattr(image_pb,"__len__"):
        image_pb=np.ones((Np, Np))
    
    xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)
    
    # R phi
    if PA<0.: PA=PA+180.
    PA_rad=PA*np.pi/180.0
    phis_rad=phis*np.pi/180.0
    
    dphi=abs(phis_rad[1]-phis_rad[0])
    Nphi=len(phis_rad)

    if rmin==0. or rmin<0.:
        rmin=rmax/1.0e3
    rs=np.linspace(rmin,rmax,Nr)

    ecc= np.sin(inc*np.pi/180.0)
    chi=1.0/(np.sqrt(1.0-ecc**2.0)) # aspect ratio between major and minor axis (>=1)
    
    ##### Calculate averaged profile
    
    Irs1=np.zeros((Nr,Nphi))
    Irs2=np.zeros((Nr,Nphi)) # pb
    for i_r in range(Nr):
        ai=rs[i_r]
        for i_p in range(Nphi):

            phi1=phis_rad[i_p]  # in the plane of the sky
            XS1,YS1=ellipse(x0,y0,phi1,chi,ai, PA_rad)

            
            ip1 = -int(XS1/ps_arcsec)+Np//2
            jp1 = int(YS1/ps_arcsec)+Np//2

            Irs1[i_r,i_p] = image[jp1,ip1] 
            Irs2[i_r,i_p] = image_pb[jp1,ip1]
        
    Ir1=np.nanmean(Irs1, axis=1) # mean intensity in Jy/beam
    Ir2=np.zeros(Nr)
    
    for i in range(Nphi):
        Ir2=Ir2+(rms/Irs2[:,i])**2.0

    Ir2=np.sqrt(Ir2/(Nphi))

    # Calculate number of independent points 

    if arc=='simple_elipse': # full 2pi ellipse
        arclength=(Nphi-1)*dphi* np.sqrt(  (1.0 + (1.0/chi)**2.0 )/2.0 ) # normalice 
    else: # partial ellipse
        arclength, phiint= arc_length(1.0,1.0/chi, phis_rad[0]-PA_rad, phis_rad[-1]-PA_rad)

    print('arc length = {:.1f} deg '.format( arclength*180.0/np.pi))
    Nindeps_1=rs*arclength/BMAJ_arcsec
    Nindeps_1[Nindeps_1<1.0]=1.0
    
    if error_std:
        Err_1=np.nanstd(Irs1, axis=1)/np.sqrt(Nindeps_1)

    else:
        Err_1=Ir2/np.sqrt(Nindeps_1)

    return np.array([rs, Ir1, Err_1]) 


def flux_profile(image, pb, x0, y0, PA, inc, rmax, rms, BMAJ, BMIN, BPA,  dpix,  rmin=0., h=0., make_figure=True ):

    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # rms [Jy]
    # dpix [arcsec]
    # ################ SPATIAL GRID
    Npix=len(image[:,0])

    Rmax=2.0*rmax # pick a larger radius to make sure flux profile saturates 
    Nr=int(round((Rmax-rmin)/dpix/8))  
    rs=np.linspace(rmin,Rmax,Nr)
    
    xs, ys, xedge, yedge = xyarray(Npix, dpix)

    xm, ym = np.meshgrid(xs-x0, ys-y0, sparse=False, indexing='xy')

    PA_rad=PA*np.pi/180.
    inc_rad=inc*np.pi/180.
    
    xp = xm * np.sin(PA_rad) + ym *np.cos(PA_rad) ###  along major axis
    yp = xm * np.cos(PA_rad) - ym *np.sin(PA_rad) ### along minor axis

    xpp=xp
    ypp=yp/np.cos(inc_rad)

    rp= np.sqrt( xp**2.0 + yp**2.0 ) # projected radius
    rpp=np.sqrt( xpp**2.0 + ypp**2.0 ) # deprojected radius

    PAs=np.arctan2(xm,ym)*180./np.pi# -180 to 180
    PAs[PAs<0.0]=PAs[PAs<0.0]+360. # from 0 to 360, discontinuity in 0.0


    Beam_area=np.pi*BMAJ*BMIN/(4.0*np.log(2.0)) # in arcsec2
    beam_maj=np.sqrt((BMAJ*np.cos( (PA-BPA)*np.pi/180. ))**2 + (BMIN*np.sin( (PA-BPA)*np.pi/180.  ))**2) # beam projected along disc major axis
    beam_min=np.sqrt((BMAJ*np.cos( (PA+90.-BPA)*np.pi/180. ))**2 + (BMIN*np.sin( (PA+90.-BPA)*np.pi/180.  ))**2) # beam projected along disc minor axis

    
    ##### Calculate flux within rs

    F=np.zeros((Nr,2))
    rmsmap=rms/pb
    rmsmap2=(rmsmap)**2.0

    
    for i_r in range(Nr):

        mask_rh=generate_mask_rh(xpp, ypp, h, inc_rad, rs[i_r], rmin)
        mask_beam=generate_mask_beam(xpp, ypp, beam_min, inc_rad, rs[i_r], rmin)

        mask=mask_rh | mask_beam
        
        F[i_r,0]= np.sum(image[mask])*(dpix**2.0)/Beam_area # Jy
        F[i_r,1]= np.sum(rmsmap2[mask]) # Jy/beam Note: /beam is ok as it is corrected later
        
        # Correct by number of independent points

        if len(image[mask])*dpix**2.>Beam_area: # if area is larger than a beam
            Nind_Npix= dpix**2.0/Beam_area
            F[i_r,1]= np.sqrt(F[i_r,1]) * np.sqrt(Nind_Npix) 
        elif len(image[mask])>0.: # if area is smaller than a beam
            F[i_r,1]=np.sqrt(np.nanmean(rmsmap2[mask]))
        else: # no pixels in mask
            F[i_r,1]=np.nan
            
    # image[mask]=0.0

    if make_figure: # make figure of mask of rmax

        mask_rh=generate_mask_rh(xpp, ypp, h, inc_rad, rmax, rmin)
        mask_beam=generate_mask_beam(xpp, ypp, beam_min, inc_rad, rmax, rmin)

        mask=mask_rh | mask_beam
        
        fig=plt.figure(figsize=(4,4))
        ax=fig.add_subplot(111)
        ax.pcolormesh(xedge, yedge, image*pb, cmap='inferno', vmax=np.nanmax(image[mask]*pb[mask]), vmin=-3*rms)#, zorder=1)
        ax.pcolormesh(xedge, yedge, mask, cmap='Greys_r', alpha=0.15*(mask>0))#, zorder=2 )

        ax.set_xlabel('RA [arcsec]')
        ax.set_ylabel('Dec [arcsec]')
        ax.set_xlim(Rmax, -Rmax)
        ax.set_ylim(-Rmax, Rmax)

        xc=Rmax -3.*BMAJ/2.#abs(minor_ticks[1]-minor_ticks[0])
        yc=-Rmax +3.*BMAJ/2.#abs(minor_ticks[1]-minor_ticks[0])
            
        width= BMAJ
        height= BMIN
        pa=BPA
        elli=Ellipse((xc,yc),width,height,angle=90.-pa,linewidth=0,fill=True,color='white', alpha=1.0)
        ax.add_patch(elli)
        
        ax.set_aspect('equal')
        plt.tight_layout()
        #plt.show()

        return np.array([rs, F[:,0], F[:,1]]), fig 

    else:
        return np.array([rs, F[:,0], F[:,1]]) 

########################################
###### FUNCTIONS FOR VERTICAL PROFILE ####
########################################


def get_vertical_profile(xs, ys, image, PA=0., beam=1.0, rms=0., Rmin=0., Rmax=1.0, Zmax=1., x0=0., y0=0., side='both', Ndown=1):


    dpix=np.abs(xs[1]-xs[0])
    xm, ym = np.meshgrid(xs-x0, ys-y0)
    
    dz=dpix
    print('Pixel size (arcsec): ',dz )
    Nz=int(round(2*Zmax/dpix/Ndown ))  

    Z_edges=np.linspace(-Zmax, Zmax, Nz+1)
    Zs=(Z_edges[1:]+Z_edges[:-1])/2.
    print('Sampling in z (arcsec): ', np.abs(Zs[1]-Zs[0]) )

    rhom=xm*np.sin(PA*np.pi/180.)+ym*np.cos(PA*np.pi/180.)
    zm= -xm*np.cos(PA*np.pi/180.)+ym*np.sin(PA*np.pi/180.)

    rhoz=np.zeros((2, Nz))

    if Rmax<Rmin or Rmax<=0.0 or Rmin<0.0:
        sys.exit(f'ERROR: invalid Rmax and/or Rmin, Rmin={Rmin}, Rmax={Rmax}')

    for i in range(Nz):

        # define slab
        if side=='both':
            maski= (np.abs(rhom)>=Rmin) & (np.abs(rhom)<=Rmax) & (Z_edges[i]<=zm) & (zm<=Z_edges[i+1])
        elif side=='PA':
            maski= (Rmin<=rhom) & (rhom<=Rmax) & (Z_edges[i]<=zm) & (zm<=Z_edges[i+1])
        elif side=='PA+180':
            maski= (-Rmin>=rhom) & (rhom>=-Rmax) & (Z_edges[i]<=zm) & (zm<=Z_edges[i+1])
        else:
            sys.exit('ERROR: invalid side')
        # sum slab
        rhoz[0, i]=np.mean(image[maski])

        # Nbeams
        if side=='both':
            l=2*(Rmax-Rmin)
        else:
            l=(Rmax-Rmin)
        Nbeams=l/beam
        rhoz[1, i] = rms/np.sqrt(Nbeams)

    return Zs, rhoz

def generate_mask_rh(xpp, ypp, h, inc_rad, rmax, rmin):


    H1pp=h*rmax*np.tan(inc_rad)
    H2pp=h*rmin*np.tan(inc_rad)

    mask_rh1= np.full(xpp.shape, False) # for outer edge
    mask_rh2= np.full(xpp.shape, False) # for inner edge

    mask_x1=np.abs(xpp)<=rmax # maximum radius
    mask_x2=np.abs(xpp)<=rmin # minimum radius

    mask_rh1[mask_x1] = (ypp[mask_x1]<np.sqrt(rmax**2-xpp[mask_x1]**2)+H1pp) & (ypp[mask_x1]>-np.sqrt(rmax**2-xpp[mask_x1]**2)-H1pp) # outer edge
    mask_rh2[mask_x2] = (ypp[mask_x2]>np.sqrt(rmin**2-xpp[mask_x2]**2)-H2pp) | (ypp[mask_x2]<-np.sqrt(rmin**2-xpp[mask_x2]**2)+H2pp) # inner edge

    # combine both
    mask_rh=mask_rh1
    mask_rh[mask_x2] = mask_rh1[mask_x2] & mask_rh2[mask_x2]

    return mask_rh

def generate_mask_beam(xpp, ypp, beam, inc_rad, rmax, rmin):

    
    #mask_beam=(np.abs(yp)<=beam_min) & (np.abs(xpp)<=rs[i_r]) & (rpp>=rmin) # rectangle 2 beams wide from major axis of the disc. Important for highly inclined discs.  
    
    mask_b1= np.full(xpp.shape, False) # for outer edge
    mask_b2= np.full(xpp.shape, False) # for inner edge

    mask_x1=np.abs(xpp)<=rmax # maximum radius
    mask_x2=np.abs(xpp)<=rmin # minimum radius

    # beampp=beam/np.cos(inc_rad) # beam along minor axis deprojected
    beampp=beam*np.tan(inc_rad) # beam along minor axis deprojected, it uses tan instead of cos such that for face on discs it becomes 0
    
    mask_b1[mask_x1] = (ypp[mask_x1]<np.sqrt(rmax**2-xpp[mask_x1]**2)+beampp) & (ypp[mask_x1]>-np.sqrt(rmax**2-xpp[mask_x1]**2)-beampp) # outer edge
    mask_b2[mask_x2] = (ypp[mask_x2]>np.sqrt(rmin**2-xpp[mask_x2]**2)-beampp) | (ypp[mask_x2]<-np.sqrt(rmin**2-xpp[mask_x2]**2)+beampp) # inner edge

    # combine both
    mask_b=mask_b1
    mask_b[mask_x2] = mask_b1[mask_x2] & mask_b2[mask_x2]

    return mask_b

########################################
###### FUNCTIONS FOR AZIMUTHAL PROFILE ####
########################################


def flux_azimuthal_profile(image, image_pb=None, x0=0., y0=0., PA=0., inc=0., rmin=0.5, rmax=1., NPA=100, dPA=0.,  rms=0.,  BMAJ_arcsec=1., BMIN_arcsec=1., ps_arcsec=0.05, error_std=False):
    # calculate average intensity profile as a function of azimuth between rmin and rmax in arcsec  
    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)
    

    Np=len(image[:,0])
    
    # check if image_pb is an array:
    if not hasattr(image_pb,"__len__"):
        image_pb=np.ones((Np, Np))

    # ################ SPATIAL GRID
    # X,Y

    xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)
    xv, yv = np.meshgrid(xs-x0, ys-y0, sparse=False, indexing='xy')

    PA_rad=PA*np.pi/180.0
    inc_rad=inc*np.pi/180.0

    ecc= np.sin(inc_rad)
    chi=1.0/(np.sqrt(1.0-ecc**2.0)) # aspect ratio between major and minor axis (>=1)

    Beam_area=np.pi*BMAJ_arcsec*BMIN_arcsec/(4.0*np.log(2.0)) # in arcsec2
 
    ##### Calculate flux within rmin, rmax and PA bins

    F=np.zeros((NPA,2))

    xpp = xv * np.cos(PA_rad) - yv *np.sin(PA_rad) ### along minor axis
    ypp = xv * np.sin(PA_rad) + yv *np.cos(PA_rad) ###  along major axis

    rdep=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 ) # real deprojected radius

    PAs=np.arctan2(xv,yv) # rad 
    PAs[PAs<0.]=PAs[PAs<0.]+2*np.pi # 0 to 2pi

    rmsmap2=(rms/image_pb)**2.0
    
    PA_mid=np.linspace(0., 2*np.pi, NPA) # rad

    if dPA<=0.:
        dPA_rad=abs(PA_mid[1]-PA_mid[0])
    else:
        dPA_rad=dPA*np.pi/180.

    
    d=abs(rmax-rmin)

    for i_pa in range(NPA):
        
        mask=(rdep<=rmax) & (rdep>rmin) & (PAs>PA_mid[i_pa]-dPA_rad) & (PAs<=PA_mid[i_pa]+dPA_rad)
        # roll over lower boundary in phi

        if PA_mid[i_pa]-dPA_rad<0.: # add lower end
            mask_l=(rdep<=rmax) & (rdep>rmin) & (PAs>PA_mid[i_pa]-dPA_rad+2*np.pi)
            mask=mask | mask_l
            
        if PA_mid[i_pa]+dPA_rad>2.0*np.pi: # add higher end
            
            mask_h=(rdep<=rmax) & (rdep>rmin) & (PAs<=PA_mid[i_pa]+dPA_rad-2*np.pi)
            mask=mask | mask_h


        ## average intensity between PA bins and radial bins
        F[i_pa,0]= np.average(image[mask])


        # Number of independent points based on the area
        Area=np.sum(mask)*ps_arcsec**2.0 # arcsec2
        N_area=Area/Beam_area

        # Number of independent points based on arc
        arclength, phiint= arc_length(1.0,1.0/chi, PA_mid[i_pa]-dPA_rad, PA_mid[i_pa]+dPA_rad)
        N_arc=arclength*(rmax+rmin)/2. / BMAJ_arcsec
        

        # Number of independent points based on radial extent
        dp=d*np.sqrt( np.cos(PA_mid[i_pa]-PA_rad)**2. + np.sin(PA_mid[i_pa]-PA_rad)**2.*np.cos(inc_rad)**2.)
        N_r=dp/BMAJ_arcsec

        # use as number of independent points the maximum between beams along arc, beams along radial extent or beams in area being averaged.
        Nind=max([N_area, N_arc, N_r])
        if error_std: # estimate error using standard deviation
            F[i_pa,1]=np.std(image[mask])/np.sqrt(Nind)
        else: # estimate error using input rms
            F[i_pa,1]= min(np.sqrt(np.mean(rmsmap2[mask])), np.sqrt(np.mean(rmsmap2[mask]))/np.sqrt(Nind))

    
    return np.array([PA_mid*180./np.pi, F[:,0], F[:,1]]) # rs, I, eI 




########################################
###### MISCELANEOUS UNCTIONS ###########
########################################


#### simply load fits image
def load_fits(fits_path, rms=0., pbcor=False, output_unit=''):
    
    ##### LOAD IMAGE
    if pbcor and not fits_path.endswith('pbcor.fits'): # load primary beam corrected image
        fits_path=fits_path.replace('.fits', '.pbcor.fits')
        
    fits = pyfits.open(fits_path) # open image cube
    image = get_last2d(fits[0].data) 

    #### READ HEADER
    header= fits[0].header
    ps_deg=float(header['CDELT2'])
    ps_mas= ps_deg*3600.0*1000.0 # pixel size input in mas
    ps_arcsec=ps_deg*3600.0
    
    Npix=np.shape(image)[0]
        
    if pbcor:
        fitspb= pyfits.open(fits_path.replace('pbcor.fits', 'pb.fits')) 
        pb = get_last2d(fitspb[0].data) 

    else:
        pb=np.ones((Npix,Npix))

    rmsmap=rms/pb

   

    try:
        BMAJ=float(header['BMAJ'])*3600.0 # arcsec 
        BMIN=float(header['BMIN'])*3600.0 # arcsec 
        BPA=float(header['BPA']) # deg 
        print("beam = %1.2f x %1.2f" %(BMAJ, BMIN))
    except:
        BMAJ=0.0
        BMIN=0.0
        BPA=0.0
    try:
        if header['CTYPE3'] == 'FREQ':
            wave=c_light/header['CRVAL3']*1.0e6
            print('wavelength = %1.3f um'%(wave))
    except: print('no CTYPE3')    
    if header['BUNIT']=='JY/PIXEL' and output_unit=='JY/ARCSEC2':
        image=image/(ps_arcsec**2.0) # Jy/pixel to Jy/arcsec2


    xs, ys, xedge, yedge=xyarray(Npix, ps_arcsec)

    ret_list= (xs, ys, xedge, yedge, image,)

    
    if BMAJ>0:
        ret_list+=(pb, rmsmap, BMAJ, BMIN, BPA)

    return ret_list

#### load fits image (and primary beam) and extract its radial profile
def radial_profile_fits_image(fits_path, fits_path_pb='', x0=0., y0=0., PA=0., inc=0., rmax=5., Nr=100, Nphi=100, rms=0., error_std=False, arcsec2=False, arc='elipse', rmin=0., wedge_width=0., ret_beam=False):

    # fits_path is the path to the fits image
    # fits_path_pb (optional) is the path to the primary beam image. If given, fits_path is assumed to be primary beam corrected
    # arcsec2: if True, transform the image and rms from Jy/beam to Jy/arcsec2
    # ret_beam: if True, also return BMAJ in arcsec
    # the rest of the parameters are passed to radial_profile()

    loaded=load_fits(fits_path, rms=rms)
    xs, ys, xedge, yedge, image = loaded[:5]
    ps_arcsec=abs(xs[1]-xs[0])
    Npix=np.shape(image)[0]

    if len(loaded)>5:
        BMAJ, BMIN = loaded[8:10]
    else:
        BMAJ, BMIN = 0., 0.

    if fits_path_pb!='':
        pb=get_last2d(pyfits.open(fits_path_pb)[0].data)
    else:
        pb=np.ones((Npix,Npix))

    if arcsec2:
        beam_area=np.pi*BMAJ*BMIN/(4.*np.log(2.))
        image=image/beam_area
        rms=rms/beam_area
        print('transforming to Jy/arcsec2')

    Ir=radial_profile(image,
                      image_pb=pb,
                      x0=x0,
                      y0=y0,
                      PA=PA,
                      inc=inc,
                      rmax=rmax,
                      Nr=Nr,
                      Nphi=Nphi,
                      rms=rms,
                      BMAJ_arcsec=BMAJ,
                      ps_arcsec=ps_arcsec,
                      error_std=error_std,
                      arc=arc,
                      rmin=rmin,
                      wedge_width=wedge_width)

    if ret_beam:
        return Ir, BMAJ
    else:
        return Ir

#### load fits image with the option of trimming the field of view and increasing the resolution
def fload_fits_image(path_image, path_pbcor='', rms=0., ps_final=0., XMAX=0., YMAX=0.0,  remove_star=False, output='', return_coordinates=False, centered=True, dpix=None): # for images from CASA

    ### PS_final in mas

    ##### LOAD IMAGE
    data1, header1	= pyfits.getdata(path_image, header=True) # open image cube
    # #### READ HEADER
    # header1	= pyfits.getheader(path_image)

    try:
        if  header1['NAXIS3']==1:
            data1 	= get_last2d(data1) # [0,0,:,:] # extract image matrix
        elif header1['NAXIS3']>1:
            data1 	= get_last3d(data1) # [0,0,:,:] # extract image matrix
    except:  # in case NAXIS3 does not exist
        # try:
        data1 	= get_last2d(data1) # [0,0,:,:] # extract image matrix
        # except:
        #     print('first object in fits file did not have image')
        #     header1	= fit1[1].header
        #     data1 	= get_last2d(fit1[1].data) # [0,0,:,:] # extract image matrix
   
    if dpix==None:
        try:
            ps_deg1=float(header1['CDELT2'])
        except:
            print('no CDELT2')
            ps_deg1=float(header1['CD2_2'])
    else:
        ps_deg1=dpix/3600.
        
    ps_mas1= ps_deg1*3600.0*1000.0 # pixel size input in mas
    ps_arcsec1=ps_deg1*3600.0

    print(data1.shape)
    Ny, Nx = data1.shape[-2], data1.shape[-1]
    print(Ny, Nx)

    if path_pbcor!='':
        fit2	= pyfits.open(path_pbcor) #abrir objeto cubo
        data2 	= get_last2d(fit2[0].data) # [0,0,:,:] #extraer matriz de datos

    else:
        data2=np.ones((Ny,Nx))
        
    rmsmap=rms/data2

    try:
        BMAJ=float(header1['BMAJ'])*3600.0 # arcsec 
        BMIN=float(header1['BMIN'])*3600.0 # arcsec 
        BPA=float(header1['BPA']) # deg 
        print("beam = %1.2f x %1.2f" %(BMAJ, BMIN))
    except:
        BMAJ=0.0
        BMIN=0.0
        BPA=0.0
    try:
        if header1['CTYPE3'] == 'FREQ':
            wave=c_light/header1['CRVAL3']*1.0e6
            print('wavelength = %1.3f um'%(wave))
    except: print('no CTYPE3')    
    if header1['BUNIT']=='JY/PIXEL' and output=='JY/ARCSEC2':
        data1=data1/(ps_arcsec1**2.0) # Jy/pixel to Jy/arcsec2
    # x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec1)

    if remove_star and header1['NAXIS3']==1:
        ij=np.unravel_index(np.argmax(data1, axis=None), data1.shape)
        print(ij)
        data1[ij]=0.0
    

    if ps_final>0.0:
        psf_arcsec=ps_final/1000.0
    else:
        ps_final=ps_mas1
        psf_arcsec=ps_arcsec1
    if XMAX<=0.0:
        XMAX=ps_arcsec1*Nx/2.
        YMAX=ps_arcsec1*Ny/2.
    elif YMAX<=0.0:
        YMAX=XMAX
    
    print(XMAX, YMAX)
    Nfx=int(round(XMAX*2.0/(ps_final/1000.0)))
    Nfy=int(round(YMAX*2.0/(ps_final/1000.0)))

    print('Nfx, Nfy = ', Nfx, Nfy)
        
    xf=np.zeros(Nfx+1)
    yf=np.zeros(Nfy+1)

    
    if not centered:
        # # pad image to make sure center is in the middle of a pixel
        # pad_x=int(round(header1['CRPIX1'] - Nfx/2.0))
        # pad_y=int(round(header1['CRPIX2'] - Nfy/2.0))   

        # # pad with zeros at the edges where it is needed
        # data1=np.pad(data1, ((min(pad_y, 0), max(0, pad_y)),(min(pad_x, 0), max(0, pad_x)), 'constant', constant_values=0))

        # roll image to put center in the middle of a pixel
        roll_x=-int(round(header1['CRPIX1'] - Nfx/2.0))
        roll_y=-int(round(header1['CRPIX2'] - Nfy/2.0))

        # print('roll_x, roll_y = ', roll_x, roll_y)
        data1=np.roll(data1, (roll_y, roll_x), axis=(0,1))
        # for i in range(Nfx+1):
        #     xf[i]=-(i-header1['CRPIX1'])*psf_arcsec
        # for i in range(Nfy+1):
        #     yf[i]=(i-header1['CRPIX2'])*psf_arcsec 

    for i in range(Nfx+1):
        xf[i]=-(i-Nfx/2.0)*psf_arcsec
    for i in range(Nfy+1):
        yf[i]=(i-Nfy/2.0)*psf_arcsec 

   

        
    try:
        if  header1['NAXIS3']==1:
            image=interpol(Nx,Nfx,ps_mas1,ps_final, data1)

        else: ### interpolation not yet implemented for image cube
            image=data1
    except: # in case NAXIS3 does not exist
        image=interpol(Nx,Nfx,ps_mas1,ps_final, data1)

    ret_list= (image,)
        
    if path_pbcor!='':
        rmsmap_out=interpol(Nx,Nfx,ps_mas1,ps_final,rmsmap)
        ret_list+=(rmsmap_out, xf, yf)
    else:
        ret_list+=(xf, yf)
    if BMAJ>0.0:
        ret_list+=(BMAJ, BMIN, BPA)

    if return_coordinates:

        RA=float(header1['CRVAL1']) # deg
        Dec=float(header1['CRVAL2']) # deg

        try:
            radesys=header1['RADESYS'].lower()
        except:
            radesys='icrs'
        c0=SkyCoord(RA, Dec, frame=radesys, unit=(u.deg, u.deg))

        ret_list+=(c0,)
        
    return ret_list


def xyarray(Np, ps_arcsec):

    xedge=np.zeros(Np+1)
    yedge=np.zeros(Np+1)

    xs=np.zeros(Np)
    ys=np.zeros(Np)

    for i in range(Np+1):

        xedge[i]=-(i-Np/2.0)*ps_arcsec#-ps_arcsec/2.0        
        yedge[i]=(i-Np/2.0)*ps_arcsec#+ps_arcsec/2.0

    for i in range(Np):
        xs[i]=-(i-Np/2.0)*ps_arcsec-ps_arcsec/2.0           
        ys[i]=(i-Np/2.0)*ps_arcsec+ps_arcsec/2.0

    return xs, ys, xedge, yedge




def Gauss2d(xi , yi, x0,y0,sigx,sigy,theta):

        xp= (xi-x0)*np.cos(theta) + (yi-y0)*np.sin(theta)
        yp= -(xi-x0)*np.sin(theta) + (yi-y0)*np.cos(theta)

        a=1.0/(2.0*sigx**2.0)
        b=1.0/(2.0*sigy**2.0)

        return np.exp(- ( a*(xp)**2.0 + b*(yp)**2.0 ) )#/(2.0*np.pi*sigx*sigy)



def Convolve_beam(path_image, BMAJ, BMIN, BPA, tag_out=''):

    #  -----cargar fit y extraer imagen

    fit1	= pyfits.open(path_image) #abrir objeto cubo de datos
    
    data1 	= get_last2d(fit1[0].data) # [0,0,:,:] # extract image matrix

    print(np.shape(data1))

    header1	= fit1[0].header
    ps_deg=float(header1['CDELT2'])
    ps_mas= ps_deg*3600.0*1000.0 # pixel size input in mas
    dtheta=ps_deg*np.pi/180.0 # dtheta in rad

    M=len(data1[:,0])

    N=M # dim output

    ps1= ps_mas # pixel size input in mas
    ps2= ps1 # pixel size output in mas

    dtheta2=ps2*np.pi/(3600.0*1000.0*180.0)

    d=0

    Fin1=data1[:,:]#*1e23/(dtheta**2.0) #  JY/PIXEL to ergs/s cm2 Hz sr

    x1=np.zeros(N)
    y1=np.zeros(N)
    for i in range(N):
        x1[i]=(i-M/2.0)*ps1
        y1[i]=(i-N/2.0)*ps1
    # for i in range(N):
    #     x1[i]=(i-M/2.0)*ps1
    #     y1[i]=(i-N/2.0)*ps1

    # BMAJ = float(sys.argv[1])#2.888e-4#*3600.0 #deg
    # BMIN = float(sys.argv[2])#2.240e-4#*3600.0  #deg
    # PA   = float(sys.argv[3])#63.7       # deg


    sigx=BMIN*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0))) 
    sigy=BMAJ*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0)))  
    theta=BPA*np.pi/180.0   #np.pi/4.0

    # Fout1=interpol(N,M,ps1,ps2,Fin1,sigx,sigy,theta)

    Gaussimage=np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            x=(i-N/2.0)*ps2
            y=(j-N/2.0)*ps2
            Gaussimage[j,i]=Gauss2d(x,y,0.0,0.0,sigx,sigy,theta)
    # Gaussimage=Gaussimage/np.max(Gaussimage)


    Fout1=convolve_fft(Fin1,Gaussimage, normalize_kernel=False)

    # a=BMAJ*np.pi/180.0
    # b=BMIN*np.pi/180.0
    # C=np.pi*a*b/(4.0*ma.log(2.0)) # in strad
    # Fout1=Fout1*C/(dtheta**2.0) # Jy/pixel to Jy/beam


    # x2=np.zeros(N+1)
    # y2=np.zeros(N+1)
    # for i in range(N+1):
    #     x2[i]=-(i-(N-1)/2.0)*ps2-ps2/2.0
    #     y2[i]=(i-(N-1)/2.0)*ps2-ps2/2.0

    # x2=x2/1000.0
    # y2=y2/1000.0

    header1['BMIN'] = BMIN
    header1['BMAJ'] = BMAJ
    header1['BPA'] = BPA

    header1['BUNIT']='JY/BEAM'

    path_fits=path_image[:-5]+'_beamconvolved'+tag_out+'.fits'
    os.system('rm '+ path_fits)
    pyfits.writeto(path_fits, Fout1, header1, output_verify='fix')


def Convolve_beam_cube(path_image, BMAJ, BMIN, BPA):

    #  -----cargar fit y extraer imagen

    fit1	= pyfits.open(path_image) #abrir objeto cubo de datos
    
    data1 	= fit1[0].data # [0,0,:,:] # extract image matrix

    print(np.shape(data1))

    header1	= fit1[0].header
    ps_deg=float(header1['CDELT2'])
    ps_mas= ps_deg*3600.0*1000.0 # pixel size input in mas
    dtheta=ps_deg*np.pi/180.0 # dtheta in rad

    N=len(data1[0,0,0,:])
    Nf=len(data1[0,:,0,0])
    ps1= ps_mas # pixel size input in mas
    ps2= ps1 # pixel size output in mas

    dtheta2=ps2*np.pi/(3600.0*1000.0*180.0)

    d=0

    Fin1=data1[:,:]#*1e23/(dtheta**2.0) #  JY/PIXEL to ergs/s cm2 Hz sr

    x1=np.zeros(N)
    y1=np.zeros(N)
    for i in range(N):
        x1[i]=(i-N/2.0)*ps1
        y1[i]=(i-N/2.0)*ps1


    # BMAJ = float(sys.argv[1])#2.888e-4#*3600.0 #deg
    # BMIN = float(sys.argv[2])#2.240e-4#*3600.0  #deg
    # PA   = float(sys.argv[3])#63.7       # deg


    sigx=BMIN*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0))) 
    sigy=BMAJ*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0)))  
    theta=BPA*np.pi/180.0   #np.pi/4.0

    # Fout1=interpol(N,M,ps1,ps2,Fin1,sigx,sigy,theta)

    Gaussimage=np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            x=(i-N/2.0)*ps2
            y=(j-N/2.0)*ps2
            Gaussimage[j,i]=Gauss2d(x,y,0.0,0.0,sigx,sigy,theta)
    # Gaussimage=Gaussimage/np.max(Gaussimage)

    Fout1=np.zeros((1,Nf,N,N))
    for k in range(Nf):
        Fout1[0,k,:,:]=convolve_fft(Fin1[0,k,:,:],Gaussimage, normalize_kernel=False)

    # a=BMAJ*np.pi/180.0
    # b=BMIN*np.pi/180.0
    # C=np.pi*a*b/(4.0*ma.log(2.0)) # in strad
    # Fout1=Fout1*C/(dtheta**2.0) # Jy/pixel to Jy/beam


    # x2=np.zeros(N+1)
    # y2=np.zeros(N+1)
    # for i in range(N+1):
    #     x2[i]=-(i-(N-1)/2.0)*ps2-ps2/2.0
    #     y2[i]=(i-(N-1)/2.0)*ps2-ps2/2.0

    # x2=x2/1000.0
    # y2=y2/1000.0

    header1['BMIN'] = BMIN
    header1['BMAJ'] = BMAJ
    header1['BPA'] = BPA

    header1['BUNIT']='Jy/beam'

    path_fits=path_image[:-5]+'_beamconvolved.fits'
    os.system('rm '+ path_fits)
    pyfits.writeto(path_fits, Fout1, header1, output_verify='fix')


    
def fcolor_black_white(i,N):

    l=i*1.0/(N-1)
    rgb=[l, l, l]

    return cl.colorConverter.to_rgb(rgb)
    
def fcolor_blue_red(i,N):

    rgb=[i*1.0/(N-1), 0.0, 1.0-i*1.0/(N-1)]

    return cl.colorConverter.to_rgb(rgb)

def fcolor_green_yellow(i,N):

    cmap=plt.get_cmap('viridis')

    x=i*1./(N-1)

    return cmap(x)

def fcolor_plasma(i,N):

    cmap=plt.get_cmap('plasma')

    x=i*1./(N-1)

    return cmap(x)


def get_last2d(data):
    if data.ndim <= 2:
        return data
    slc = [0] * (data.ndim - 2)    
    slc += [slice(None), slice(None)]
    return data[tuple(slc)]
    
def get_last3d(data):
    if data.ndim <= 3:
        return data
    slc = [0] * (data.ndim - 3)    
    slc += [slice(None), slice(None), slice(None)]
    return data[tuple(slc)]
    

def inter(Nin,Nout,i,j,ps1,ps2,Fin):
    
    # get values of new pixels ij from input image Fin with NinxNin pixels
    
    f=0.0
    S=0.0
    a=0.5*ps1 # sigma
    di=(i-Nout/2.0)*ps2
    dj=(j-Nout/2.0)*ps2
	
    ni=int(di/ps1+Nin/2.0-5.0*a/ps1)
    mi=int(dj/ps1+Nin/2.0-5.0*a/ps1)
    nmax=int(di/ps1+Nin/2.0+5.0*a/ps1)
    mmax=int(dj/ps1+Nin/2.0+5.0*a/ps1)

    if ni<0: ni=0
    if mi<0: mi=0
    if nmax<0: nmax=0
    if mmax<0: mmax=0
    if ni>Nin-1: ni=Nin-1
    if mi>Nin-1: mi=Nin-1
    if nmax>Nin-1: nmax=Nin-1
    if mmax>Nin-1: mmax=Nin-1
    
    for n in range(ni,nmax):
        
        dn=(n-Nin/2.0)*ps1
        k=0
        for m in range(mi,mmax):
	    
            dm=(m-Nin/2.0)*ps1
			
            r=np.sqrt((dn-di)**2.0+(dm-dj)**2.0)
            if r<3.*a: 
                P=np.exp(-r**2.0/(2.0*a**2.0))
                f=f+P*Fin[n,m]
                S=S+P
                k=1

            elif k==1: break
            
    if S==0.0: return 0.0		
    else:
        return f/S

def inter_vector(Nin,Nout,i,j,ps1,ps2,Fin): # still much slower than the one with for loops
    
    # get values of new pixels ij from input image Fin with NinxNin pixels
    
    f=0.0
    S=0.0
    a=1.0*ps1 # sigma
    di=(i-Nout/2.0)*ps2 # pixel position in output grid
    dj=(j-Nout/2.0)*ps2 # pixel position in output grid

    nsigma=2.
    
    ni=int(di/ps1+Nin/2.0-nsigma*a/ps1)
    mi=int(dj/ps1+Nin/2.0-nsigma*a/ps1)
    nmax=int(di/ps1+Nin/2.0+nsigma*a/ps1)
    mmax=int(dj/ps1+Nin/2.0+nsigma*a/ps1)

    if ni<0: ni=0
    if mi<0: mi=0
    if nmax<0: nmax=0
    if mmax<0: mmax=0
    if ni>Nin-1: ni=Nin-1
    if mi>Nin-1: mi=Nin-1
    if nmax>Nin-1: nmax=Nin-1
    if mmax>Nin-1: mmax=Nin-1

    ns=np.arange(Nin)
    ms=np.arange(Nin)

    nm, mm= np.meshgrid(ns, ms)

    mask= (nm>ni) & (nm<nmax) & (mm>mi) & (mm<mmax)

    dn=(nm-Nin/2.0)*ps1
    dm=(mm-Nin/2.0)*ps1

    r=np.sqrt((dn-di)**2.0+(dm-dj)**2.0)

    P=np.exp(-r**2.0/(2.0*a**2.0))

    # plt.imshow(P)
    # plt.show()
    
    f=np.sum(P[mask]*Fin[mask])
    S=np.sum(P[mask])

    if S==0:
        return 0.0
    else:
        return f/S
    
   
    

    
def interpol(Nin,Nout,ps1,ps2,Fin):
    print(ps1, ps2, Nin, Nout)

    if abs(ps1-ps2)/ps1>0.001: # significant pixel size difference 
        F=np.zeros((Nout,Nout), dtype=np.float64)
        for i in range(Nout):
            for j in range (Nout):	
                F[i,j]=inter(Nin,Nout,i,j,ps1,ps2,Fin)
    elif Nin>Nout: # same pixel but output image is smaller
        diff=Nin-Nout
        if diff%2==0:
            print('no interpolation and different size')
            i1=diff//2
            i2=Nin-diff//2
            F=Fin[i1:i2,i1:i2]
        else:
            F=np.zeros((Nout,Nout), dtype=np.float64)
            for i in range(Nout):
                for j in range (Nout):	
                    F[i,j]=inter_vector(Nin,Nout,i,j,ps1,ps2,Fin)
    elif Nout>Nin and ps1==ps2: # same pixel but output image is larger so need to pad image

        F=fpad_image(Fin, Nout, Nout, Nin, Nin)
        
    elif Nout>Nin:
        raise ValueError('Output image size is larger than input')
    else:
        print('no interpolation and same size')
        F=Fin
    return F

def fpad_image(image_in, pad_x, pad_y, nx, ny):

    if image_in.shape[-2:] != (pad_x,pad_y):
        pad_image = np.zeros((pad_x,pad_y))
        if nx%2==0 and ny%2==0: # even number of pixels
            pad_image[
                      pad_y//2-ny//2:pad_y//2+ny//2,
                      pad_x//2-nx//2:pad_x//2+nx//2] = image_in[:,:]
        else:                  # odd number of pixels
            pad_image[
                      pad_y//2-(ny-1)//2:pad_y//2+(ny+1)//2,
                      pad_x//2-(nx-1)//2:pad_x//2+(nx+1)//2] = image_in[:,:]
        return pad_image

    else:                      # padding is not necessary as image is already the right size (potential bug if nx>pad_x)
        return image_in


def get_beam_size(path_image, verbose=True):

    fit1	= pyfits.open(path_image) # open image cube
    #### READ HEADER
    header1	= fit1[0].header
    BMAJ=float(header1['BMAJ'])*3600.0 # arcsec 
    BMIN=float(header1['BMIN'])*3600.0 # arcsec 
    BPA=float(header1['BPA']) # deg 

    if verbose:
        print("beam = %1.2f x %1.2f" %(BMAJ, BMIN))

    return BMAJ, BMIN, BPA

def get_wavelength(path_image):

    fit1	= pyfits.open(path_image) # open image cube
    #### READ HEADER
    header1	= fit1[0].header
    
    freq=float(header1['CRVAL3']) # GHz
    wav=c_light*1.0e3/freq

    return wav,freq



def fload_fits_image_mira(path_image, ps_final, XMAX): # for images from CASA

    ### PS_final in mas

    ##### LOAD IMAGE
    fit1	= pyfits.open(path_image) # open image cube
    data1 	= get_last2d(fit1[0].data) # [0,0,:,:] # extract image matrix

    #### READ HEADER
    header1	= fit1[0].header
    try:
        ps_deg1=float(header1['CDELT2'])
    except:
        print('no CDELT2')
        ps_deg1=float(header1['CD2_2'])
        
    if ps_deg1<1.0: # i.e. in degrees rather than mas
        ps_mas1= ps_deg1*3600.0*1000.0 # pixel size input in mas
    else:
        ps_mas1= ps_deg1 # pixel size input in mas
  
    
    N1=len(data1[:,0])

    Nf=int(XMAX*2.0/(ps_final))
    
    xf=np.zeros(Nf+1)
    yf=np.zeros(Nf+1)
    for i in range(Nf+1):
        xf[i]=-(i-Nf/2.0)*ps_final  
        yf[i]=(i-Nf/2.0)*ps_final

    image=interpol(N1,Nf,ps_mas1,ps_final, data1)
    return image, xf, yf
    


def rainbow_cmap():



    cdict3 = {'red': ((0.0, 0.0, 0.0),
                      (0.4, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.6, 0.0, 0.0),
                      (0.7, 1.0, 1.0),
                      (0.8, 1.0, 1.0),
                      (0.9, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),
              'green':((0.0, 0.0, 0.0),
                       (0.4, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (0.6, 1.0, 1.0),
                       (0.7, 1.0, 1.0),
                       (0.8, 0.0, 0.0),
                       (0.9, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
              'blue': ((0.0, 0.0, 0.0),
                       (0.4, 1.0, 1.0),
                       (0.5, 1.0, 1.0),
                       (0.6, 0.0, 0.0),
                       (0.7, 0.0, 0.0),
                       (0.8, 0.0, 0.0),
                       (0.9, 1.0, 1.0),
                       (1.0, 1.0, 1.0))}



    my_cmap = cl.LinearSegmentedColormap('my_colormap',cdict3,256)

    return my_cmap


#########################################
############## DATA CUBE ANALYSIS #######
#########################################

def fload_fits_cube(path_cube, line='CO32', type_data='data', output=''): # for images from CASA

    if line=='CO32':
        f_line=345.79599 # GHz
    if line=='CO21':
        f_line=230.538000 # GHz

    if line=='13CO21':
        f_line=220.3986 # GHz

    if line=='C18O21':
        f_line=219.56035410 # GHz
        
    if line=='HCN43':
        f_line=354.50547590 # GHz
    if line=='C10':
        f_line=492.1606510 # GHz
        
    ##### LOAD CUBE
    fit1	= pyfits.open(path_cube) # open image cube
    data1 	= get_last3d(fit1[0].data) # [0,0,:,:] # extract image matrix

    #### READ HEADER
    header1	= fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_mas1= ps_deg1*3600.0*1000.0 # pixel size input in mas
    ps_arcsec1=ps_deg1*3600.0
    
    N1=len(data1[0,0,:])
    Nf=len(data1[:,0,0])

    if type_data=='data':
        try:
            BMAJ=float(header1['BMAJ'])*3600.0 # arcsec 
            BMIN=float(header1['BMIN'])*3600.0 # arcsec 
            BPA=float(header1['BPA']) # deg 
            print("beam = %1.2f x %1.2f" %(BMAJ, BMIN))
        except:
            header = pyfits.getheader(path_cube)
            if header.get('CASAMBM', False):
                beam = pyfits.open(path_cube)[1].data
                beam = np.median([b[:3] for b in beam.view()], axis=0)
                BMAJ=beam[0]
                BMIN=beam[1]
                BPA= beam[2]
                print("beam = %1.2f x %1.2f" %(BMAJ, BMIN))

                
    ########### SPATIAL GRID

    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec1)

        
    ########## FREQUENCY GRID
    ckms=299792.458 # km/s
        
    df=float(header1['CDELT3'])/1.0e9 # GHz
    k0=float(header1['CRPIX3'])

    f0=float(header1['CRVAL3'])/1.0e9 - (k0-1)*df # GHz
    print('v0 = ',(f0-f_line)/f_line *ckms )
    

    fs=np.linspace(f0,f0+df*(Nf-1),Nf) #GHz
    vs=-(fs-f_line)*ckms/f_line  # km/s
    
    dv=vs[1]-vs[0] # km/s
    print("dv [km/s] = ", dv)
    print("dnu [GHz] = ", df)

    if header1['BUNIT']=='JY/PIXEL' and output=='JY/ARCSEC2':
        data1=data1/(ps_arcsec1**2.0) # Jy/pixel to Jy/arcsec2
    
    if type_data=='data':
        return data1, ps_arcsec1, x1, y1, x1edge, y1edge, BMAJ, BMIN, BPA, fs, vs, dv
    else:
        return data1, ps_arcsec1, x1, y1, x1edge, y1edge, fs, vs, dv

    

def moment_0(path_cube, line='CO32', v0=0.0, dvel=10.0,  rmin=0.0, inc=90.0, M_star=1.0, ps_final=0.0, XMAX=0.0, rms=0.0):

    data1, ps_arcsec1, x1, y1, x1edge, y1edge, BMAJ, BMIN, BPA, fs, vs, dv = fload_fits_cube(path_cube, line='CO32')

    if rmin!=0.0:
        Dvel=np.sqrt(G*M_star*M_sun/(rmin*au)) *np.sin(inc*np.pi/180.0) /1.0e3
    else:
        Dvel=dvel
    print('Dvel = [km/s]', Dvel)
    mask_v=(vs>=v0-Dvel) & (vs<=v0+Dvel)
    print(len(vs[mask_v]), len(vs))
    moment0=np.sum(data1[mask_v,:,:], axis=0)*abs(dv) # Jy km/s
    if rms>=0.0:
        rms_moment0=rms*abs(dv)*np.sqrt( 2*Dvel/abs(dv) )
    print('dvel, dv = ', Dvel, dv)
    if ps_final==0.0 or XMAX==0.0:
        
        return moment0, x1edge, y1edge, BMAJ, BMIN, BPA, dv, rms_moment0
    else:
        Nf=int(XMAX*2.0/(ps_final/1000.0))
        psf_arcsec=ps_final/1000.0

        xf, yf, xfedge, yfedge = xyarray(Nf, psf_arcsec)

        N1=len(data1[0,0,:])
        ps_mas1=ps_arcsec1*1.0e3

        image=interpol(N1,Nf,ps_mas1,ps_final, moment0)
   
        return image, xfedge, yfedge, BMAJ, BMIN, BPA, dv, rms_moment0
        
def moment_0_shifted(cube, xs, ys, ps_arcsec, BMAJ, vs, v0 , x0, y0, PA, inc, M_star, dpc, Dvel0=3.2, f1=1.0, f2=1.1, sign=1.0, width=3.):
    Npix=len(cube[0,0,:])
    Nf=len(vs)
    dv=vs[1]-vs[0]
    ### calculate shift matrix
    shiftm=int(np.sign(sign))*np.rint(f_shift(Npix, x0, y0, ps_arcsec, PA*np.pi/180., inc*np.pi/180., M_star, dpc, rlim=0.5*BMAJ*dpc)/abs(dv)).astype(int)
    # plt.pcolormesh(shiftm, vmin=-50, vmax=50)
    # plt.colorbar()
    # plt.show()
    ### calculate moment 0
    moment0=np.zeros((Npix,Npix))
    rmsmap=np.ones((Npix,Npix)) # rms in Moment0

    Xs, Ys=np.meshgrid(xs, ys, indexing='xy')

    Rs=np.sqrt((Xs-x0)**2.+(Ys-y0)**2.)

    rlim=BMAJ
    Dvel1=width*np.abs(dv) # 6 channels wide
    Dvel0=np.abs(Dvel0) # make sure it is possitive
    if dv>0.0:
        k_min1=max(0, int((v0-Dvel1-vs[0])/dv) ) # to use beyond f2*rlim
        k_max1=min(Nf, int((v0+Dvel1-vs[0])/dv) ) # idem

        k_min0=max(0, int((v0-Dvel0-vs[0])/dv) )
        k_max0=min(Nf, int((v0+Dvel0-vs[0])/dv) )
    else:
        k_min1=max(0, int((v0+Dvel1-vs[0])/dv) ) # to use beyond f2*rlim
        k_max1=min(Nf, int((v0-Dvel1-vs[0])/dv) ) # idem

        k_min0=max(0, int((v0+Dvel0-vs[0])/dv) )
        k_max0=min(Nf, int((v0-Dvel0-vs[0])/dv) )
        
    print(k_min0, k_max0)
    
    for j in range(Npix):
        for i in range(Npix):

            
            if Rs[j,i]>=rlim*f2: ## safe to use keplerian mask
                spectrum_shifted = np.roll(cube[:,j,i], shiftm[j,i], axis=0)

                moment0[j,i]=np.sum(spectrum_shifted[k_min1:k_max1+1], axis=0)*np.abs(dv)
                
            elif Rs[j,i]>=rlim*f1 and Rs[j,i]<rlim*f2:  ## transition region
                spectrum_shifted = np.roll(cube[:,j,i], shiftm[j,i], axis=0)

                Dvelx=Dvel0 +(Rs[j,i] - rlim*f1)*(Dvel1-Dvel0)/(f2*rlim-f1*rlim)

                if dv>0.0:
                    k_minx=max(0, int((v0-Dvelx-vs[0])/dv) )
                    k_maxx=min(Nf, int((v0+Dvelx-vs[0])/dv) )
                else:
                    k_minx=max(0, int((v0+Dvelx-vs[0])/dv) )
                    k_maxx=min(Nf, int((v0-Dvelx-vs[0])/dv) )

                    
                moment0[j,i]=np.sum(spectrum_shifted[k_minx:k_maxx+1], axis=0)*np.abs(dv)
                rmsmap[j,i]=np.sqrt((k_maxx-k_minx)*1.0/( (k_max1-k_min1) ))
               
            else: # too close to star, no shift 
                moment0[j,i]=np.sum(cube[k_min0:k_max0+1, j, i], axis=0)*np.abs(dv)
                rmsmap[j,i]=np.sqrt((k_max0-k_min0)*1.0/( (k_max1-k_min1) ))
    

    return moment0, rmsmap

def Flux_inside_cube(amin, amax, cube , ps_arcsec, vs, Dvel, v0, PArad, incrad, x0, y0, averaged=True):
    
    # returns flux in Jy (cube must be in Jy/arcsec2) 

    dv=vs[1]-vs[0] # km/s
    Nf=len(vs)
    if Dvel>0.0 and dv>0.0:
        k_min=max(0, int(round((v0-Dvel-vs[0])/dv)) )
        k_max=min(Nf, int(round((v0+Dvel-vs[0])/dv)) )
    elif Dvel>0.0 and dv<0.0:
        k_min=max(0, int(round((v0+Dvel-vs[0])/dv)) )
        k_max=min(Nf, int(round((v0-Dvel-vs[0])/dv)) )
        # print k_min,k_max, dv, Dvel, v0
    else:
        print('error, dv<0 or Dvel<0')
        return -1
    
    F=0.0 # integrated  flux
    # Npix=0  # number of pixels over which we are integrating (not used)
    Nfr=k_max-k_min+1 # number of frequency points over which we will integrate
    Rmss=np.zeros(Nf-Nfr) # array where we will compute the rms of the spectrum
    N1=len(cube[0,0,:]) # number of pixels image
    chi=1.0/np.cos(incrad) # aspect ratio of disc

    
    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec)

    for i in range(N1):
        for j in range(N1):

            xi=x1[i]-x0
            yi=y1[j]-y0      
            xpp = xi * np.cos(PArad) - yi *np.sin(PArad) ### along minor axis
            ypp = xi * np.sin(PArad) + yi *np.cos(PArad) ###  along major axis       
            r=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 )
            
            if r<=amax and r>=amin:
                # Npix+=1
                # plt.plot(xi,yi, 'o',color='blue')                            
                F+=np.sum(cube[k_min:k_max+1,j,i]) # Jy/arcsec
                Rmss[0:k_min]=  Rmss[0:k_min]+  cube[0:k_min,j,i]     
                Rmss[k_min:]=  Rmss[k_min:]+  cube[k_max+1:,j,i]    
    # plt.xlim(10.0,-10.0)
    # plt.ylim(-10.0,10.0)
    # plt.show()
    # print('area = %1.1e arcsec2'%(Npix*ps_arcsec**2.0))
    Delta=(ps_arcsec**2.0)*abs(dv) # constant to obtain total flux in Jy km/s
    Rms=np.std(Rmss)
    if averaged:
        factor=1.6 # typically for 0.8 km/s wide channels that result from averaging 0.4km/2 channels
        ##### https://help.almascience.org/index.php?/Knowledgebase/Article/View/29
    else:
        factor=2.667 #  for 0.4 km/s wide channels without averaging
    dF=Rms*np.sqrt(Nfr*factor)*Delta # derived from:
    # rms * sqrt(Nfr) * X * dv = rms * (dv*factor) * sqrt(Nfr/factor) (true number of independent measurements)
    # or  as dF= rms/sqrt(Nfr/factor) * Nfr * dv 
    
    # print Nfr, Delta, factor
    if dF==0.0:
        return 1.0e-6, 1.0 
    else:
        return F*Delta , dF


def Spectrum(amin,amax, cube,  ps_arcsec, vs, Dvel, v0, PArad, incrad, x0, y0):
    # return spectrum in Jy (cube must be in Jy/arcsec2) 

    dv=vs[1]-vs[0] # km/s
    Nf=len(vs)
    # print v0, Dvel, vs[0], dv 
    if Dvel>0.0 and dv>0.0:
        k_min=max(0, int(round((v0-Dvel-vs[0])/dv)))
        k_max=min(Nf, int(round((v0+Dvel-vs[0])/dv)))
    elif Dvel>0.0 and dv<0.0:
        k_min=max(0, int(round((v0+Dvel-vs[0])/dv)))
        k_max=min(Nf, int(round((v0-Dvel-vs[0])/dv)))
    else:
        print('error, dv<0 or Dvel<0')
        return -1


    bad_chan=2
    F=np.zeros(Nf) # spectrum
    F2=np.zeros(Nf-(k_max-k_min+1)-2*bad_chan) # spectrum without line where we calculate rms, removing bad_chan channels from each edge
    Npix=0 # number of pixels over which we integrate
    N1=len(cube[0,0,:]) # number of pixels image
    chi=1.0/np.cos(incrad) # aspect ratio of disc

    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec)

    
    for i in range(N1):
        for j in range(N1):
            
            xi=x1[i]-x0
            yi=y1[j]-y0   
            xpp = xi * np.cos(PArad) - yi *np.sin(PArad) ### along minor axis
            ypp = xi * np.sin(PArad) + yi *np.cos(PArad) ###  along major axis
            # PAi=np.arctan2(xi,yi)*180.0/np.pi # North=0                    
            r=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 )
            
            if r<amax and r>amin:
                Npix+=1.0
                # for k in range(Nf):           
                F[:]+=cube[:,j,i] # Jy/arcsec
    F2[0:k_min-bad_chan]=F[bad_chan:k_min]
    F2[k_min-bad_chan:]=F[k_max+1:-(bad_chan)]
    # corr=np.correlate(F2, F2, mode='full')
    # plt.plot(corr[corr.size//2:])
    # plt.axvline(k_min)
    # plt.axvline(k_max)
    # plt.show()
    rms=np.nanstd(F2)                        
    Delta=(ps_arcsec**2.0)
    
    return F*Delta, rms*Delta, Delta, Npix, k_max, k_min

def sub_baseline_spectrum(I, vs, Nf, k_min, k_max, order=4 ):


    I2=np.zeros(Nf-(k_max-k_min+1))
    I2[:k_min]=I[:k_min]
    I2[k_min:]=I[k_max+1:]

    vs2=np.zeros(Nf-(k_max-k_min+1))
    vs2[:k_min]=vs[:k_min]
    vs2[k_min:]=vs[k_max+1:]

    pfit = np.polyfit(vs2, I2, order)   # Fit a 2nd order polynomial to (x, y) data
    Baseline = np.polyval(pfit, vs)

    return Baseline, I-Baseline


def f_shift(N1, x0, y0, ps_arcsec, PA, inc, M_star, dpc, vlim=10.0, rlim=0.0):

    # vlim in km/s
    ### return matrix of the size of the image with the shifts on each pixel
    shifts=np.zeros((N1,N1))

    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec)
    chi=1.0/np.cos(inc) # aspect ratio of disc

    for i in range(N1):
        for j in range(N1):

            xi=x1[i]-x0 # x in sky in arcsec
            yi=y1[j]-y0 # y in sky in arcsec
                    
            xpp = xi * np.cos(PA) - yi *np.sin(PA) ###  along minor axis in arcsec
            ypp = xi * np.sin(PA) + yi *np.cos(PA) ###  along major axis in arcsec
                    
            PAi=np.arctan2(xi,yi) # North=0 in rad

            # if PAi<0.0: PAi=2.0*np.pi+PAi
            
            r=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 )*dpc # deprojected radius of that pixel AU
            vk=np.sqrt(G*M_star/ (r*au) )/1.0e3  # km/s

            fsky=PAi-PA # [rad] angle between disc PA and pixel in plane of sky
            fdisc=np.arctan2(xpp*chi,ypp) # [rad] angle between disc PA and pixel in plane of disc
            vr=vk*np.cos(fdisc)*np.sin(inc)
            if np.abs(vr)<vlim:
                shifts[j,i]=vr
            else: shifts[j,i]=vlim*vr/abs(vr)

            if r<rlim:
                shifts[j,i]=0.0

    return shifts # km/s

def plot_cube(filename,cube, ps_arcsec, xedge, yedge, vs, v0=0., Dv=10., rms=0.0, vmin=0.0, vmax=100.0, colormap='inferno', tickcolor='white', XMAX=10.0, major_ticks=np.arange(-15, 20.0, 5.0) , minor_ticks=np.arange(-15.0, 15.0+1.0, 1.0), BMAJ=0.0, BMIN=0.0, BPA=0.0, show_beam=True, loc_beam='ll', show=True, clabel=r'Intensity [$\mu$Jy beam$^{-1}$]', formatcb='%1.0f', cbticks=np.arange(-500.0,500.0,50.0), contours=True, c_levels=[3.0,5.0, 8.0],star=True, xstar=0.0, ystar=0.0, cbar_log=False, xunit='arcsec', bad_color=(0,0,0), ruller=False, dpc=10. ):


    plt.style.use('style1')
    font= {'family':'Times New Roman', 'size': 11}
    rc('font', **font)

    YMAX=XMAX
    dv=abs(vs)
    chan_plots=[]
    for i in range(len(vs)):
        if vs[i]>=v0-Dv and vs[i]<=v0+Dv:
            chan_plots.append(i)
    Nchan=len(chan_plots)
    Nr=int(np.sqrt(Nchan))+1
    Nc=int(np.sqrt(Nchan))
    if Nr*Nc<Nchan: Nc+=1
    print(Nchan, Nr, Nc)
    fig = plt.figure(figsize=(Nc*2,Nr*2)) #(8,6))

    for i in range(Nchan):
        
        axi=fig.add_subplot(Nr,Nc,i+1)


        if not cbar_log:
            my_cmap = copy.copy(plt.cm.get_cmap(colormap)) # copy the default cmap
            my_cmap.set_bad(bad_color)
            pc=axi.pcolormesh(xedge,yedge,cube[chan_plots[i],:,:], vmin=vmin, vmax=vmax,  cmap=my_cmap, rasterized=True)
  
            #pc.set_edgecolor('face')
            # cb= fig.colorbar(pc,orientation='horizontal',label=clabel,format=formatcb, ticks=cbticks, pad=0.12)
            # cb.ax.minorticks_on()
        else:
            my_cmap = copy.copy(plt.cm.get_cmap(colormap)) # copy the default cmap
            my_cmap.set_bad(bad_color)
            pc=axi.pcolormesh(xedge,yedge,image, norm=LogNorm(vmin=vmin, vmax=vmax),  cmap=my_cmap, rasterized=True)
            #pc.set_edgecolor('face')
            # cb= fig.colorbar(pc,orientation='horizontal',label=clabel, pad=0.12)

        c1=fcolor_black_white(0.5,3)
        c2=fcolor_black_white(1.0,3)
        c3=fcolor_black_white(1.5,3)

        if contours:
            PS=abs(yedge[1]-yedge[0])
            c1=axi.contour(xedge[:-1]-PS/2.0,yedge[:-1]+PS/2.0, cube[chan_plots[i],:,:]/rms, levels=c_levels,colors=[c1,c2, c3], linewidths=1.0)
        # cb.add_lines(con2)

        axi.set_xticks(major_ticks)                                                       
        axi.set_xticks(minor_ticks, minor=True)                                           
        axi.set_yticks(major_ticks)                                                       
        axi.set_yticks(minor_ticks, minor=True) 

        ### this doesn't work anymore
        # for tick in axi.get_xticklines():
        #     tick.set_color(tickcolor)

        # for minortick in axi.xaxis.get_minorticklines():
        #     minortick.set_color(tickcolor)

        # for tick in axi.get_yticklines():
        #     tick.set_color(tickcolor)

        # for minortick in axi.yaxis.get_minorticklines():
        #     minortick.set_color(tickcolor)

        axi.spines['bottom'].set_color(tickcolor)
        axi.spines['top'].set_color(tickcolor)
        axi.spines['left'].set_color(tickcolor)
        axi.spines['right'].set_color(tickcolor)

        axi.tick_params(axis='both', colors=tickcolor, which='both')
        
        axi.set_aspect('equal')

        axi.set_xlim(XMAX,-XMAX)
        axi.set_ylim(-XMAX,XMAX)

        if  i==Nc*(Nr-1):
            axi.set_xlabel('RA offset ['+xunit+']')
            axi.set_ylabel('DEC offset ['+xunit+']')

            #---add beam
            if BMAJ!=0.0 and BMIN!=0.0 and show_beam:
                if loc_beam=='lr':
                    xc=-XMAX+2.0*BMAJ#abs(minor_ticks[1]-minor_ticks[0])
                    yc=-YMAX+2.0*BMAJ#abs(minor_ticks[1]-minor_ticks[0])
                else:
                    xc=XMAX-2.0*BMAJ#abs(minor_ticks[1]-minor_ticks[0])
                    yc=-YMAX+2.0*BMAJ#abs(minor_ticks[1]-minor_ticks[0])
        
                width= BMAJ
                height= BMIN
                pa=BPA
                elli=Ellipse((xc,yc),width,height,angle=90.-pa,linewidth=0,fill=True,color='white', alpha=1.0)
       
                axi.add_patch(elli)

        else:

            xticks1=axi.xaxis.get_major_ticks()
            yticks1=axi.yaxis.get_major_ticks()
              
            for j in range(len(xticks1)):
                xticks1[j].label1.set_visible(False)
            for j in range(len(yticks1)):
                yticks1[j].label1.set_visible(False)
            
        if star:
            # add star
            axi.plot(xstar , ystar, marker='+', color=tickcolor, markersize=3.5, mew=1.0)

        if ruller:
            x1=-XMAX+2.0*abs(minor_ticks[1]-minor_ticks[0])
            x2=x1+30./dpc
            yc=-XMAX+2.0*abs(minor_ticks[1]-minor_ticks[0])
            axi.plot([x1,x2], [yc, yc], color='white')
            axi.text(x2, yc*0.95, '30 au', color='white')

        xtext=-XMAX*0.2
        ytext=XMAX*0.8
        axi.text(xtext, ytext,'%1.1f km/s'%(vs[chan_plots[i]]-v0), color='white')

        
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=1.0, wspace=0.01, hspace=0.01)
    print('saving...')

    plt.savefig(filename, dpi=500)#,format='png', dpi=500)
    print('saved')

    if show:
        plt.show()









def save_image(filename, image, xedge, yedge, rms=0.0, rmsmap=0.0, vmin=0.0, vmax=100.0, colormap='inferno', tickcolor='white', XMAX=10.0, YMAX=0.0, major_ticks=np.arange(-15, 20.0, 5.0) , minor_ticks=np.arange(-15.0, 15.0+1.0, 1.0), BMAJ=0.0, BMIN=0.0, BPA=0.0, show_beam=True, loc_beam='ll', show=True, clabel=r'Intensity [$\mu$Jy beam$^{-1}$]', formatcb='%1.0f', cbticks=np.arange(-500.0,500.0,50.0), contours=True, c_levels=[3.0,5.0, 8.0], star=True, xstar=0.0, ystar=0.0, starcolor='white', cbar_log=False, xunit='arcsec', bad_color=(0,0,0), ruller=False, dpc=10., planet=False, xplt=0.0, yplt=0.0, pltcolor='white', title='', white_back=False, vmin2=0., axes=True, cbar=True):


    plt.style.use('style1')
    font= {'family':'Times New Roman', 'size': 11}
    rc('font', **font)

    if YMAX<=0.0:
        YMAX=XMAX

    # ysize=YMAX/XMAX
    if cbar:
        fig = plt.figure(figsize=(4,4.6))#(8,6))
    else:
        fig = plt.figure(figsize=(4,4.))#(8,6))

    ax1=fig.add_subplot(111)


    if not cbar_log:
        my_cmap = copy.copy(plt.cm.get_cmap(colormap)) # copy the default cmap
        if white_back:
            # background
            image_masked=np.where(image<vmin, np.nan, image)
            pcb=ax1.pcolormesh(xedge,yedge,image, vmin=vmin2, vmax=vmin,  cmap='binary', rasterized=True)
            pc=ax1.pcolormesh(xedge,yedge,image_masked, vmin=vmin, vmax=vmax,  cmap=my_cmap, rasterized=True)
        else:
            my_cmap.set_bad(bad_color)
            pc=ax1.pcolormesh(xedge,yedge,image, vmin=vmin, vmax=vmax,  cmap=my_cmap, rasterized=True)
  
        #pc.set_edgecolor('face')
        if cbar:
            cb= fig.colorbar(pc,orientation='horizontal',label=clabel,format=formatcb, ticks=cbticks, pad=0.12)
            cb.ax.minorticks_on()
    else:
        my_cmap = copy.copy(plt.cm.get_cmap(colormap)) # copy the default cmap
        my_cmap.set_bad(bad_color)
        pc=ax1.pcolormesh(xedge,yedge,image, norm=LogNorm(vmin=vmin, vmax=vmax),  cmap=my_cmap, rasterized=True)
        #pc.set_edgecolor('face')
        if cbar:
            cb= fig.colorbar(pc,orientation='horizontal',label=clabel, pad=0.12)

    c1=fcolor_black_white(1.0,3)
    c2=fcolor_black_white(1.0,3)
    c3=fcolor_black_white(1.0,3)
    # c1=fcolor_black_white(0.5,3)
    # c2=fcolor_black_white(1.0,3)
    # c3=fcolor_black_white(1.5,3)

    if contours:
        PS=abs(yedge[1]-yedge[0])
        if type(rmsmap) is np.ndarray:
            c1=plt.contour(xedge[:-1]-PS/2.0,yedge[:-1]+PS/2.0, image/rmsmap, levels=c_levels,colors=[c1,c2, c3], linewidths=1.0)
        else:
            c1=plt.contour(xedge[:-1]-PS/2.0,yedge[:-1]+PS/2.0, image/rms, levels=c_levels,colors=[c1,c2, c3], linewidths=1.0)
        # cb.add_lines(con2)

    ax1.set_xticks(major_ticks)                                                       
    ax1.set_xticks(minor_ticks, minor=True)                                           
    ax1.set_yticks(major_ticks)                                                       
    ax1.set_yticks(minor_ticks, minor=True) 

    ax1.tick_params(axis='both', which='both', color=tickcolor, labelcolor='black')

    #### code below not working anymore
    # for tick in ax1.get_xticklines():
    #     tick.set_color(tickcolor)

    # for minortick in ax1.xaxis.get_minorticklines():
    #     minortick.set_color(tickcolor)

    # for tick in ax1.get_yticklines():
    #     tick.set_color(tickcolor)

    # for minortick in ax1.yaxis.get_minorticklines():
    #     minortick.set_color(tickcolor)

    ax1.spines['bottom'].set_color(tickcolor)
    ax1.spines['top'].set_color(tickcolor)
    ax1.spines['left'].set_color(tickcolor)
    ax1.spines['right'].set_color(tickcolor)

    ax1.set_xlabel('RA offset ['+xunit+']')
    ax1.set_ylabel('DEC offset ['+xunit+']')
    ax1.set_xlim(XMAX,-XMAX)
    ax1.set_ylim(-YMAX,YMAX)
    ax1.set_aspect('equal')

    #---add beam
    print(BMAJ, BMIN, show_beam)
    if BMAJ!=0.0 and BMIN!=0.0 and show_beam:
        if loc_beam=='lr':
            xc=-XMAX+1.5*BMAJ/2.#abs(minor_ticks[1]-minor_ticks[0])
            yc=-YMAX+1.5*BMAJ/2.#abs(minor_ticks[1]-minor_ticks[0])
        else:
            xc=XMAX-1.5*BMAJ/2.#abs(minor_ticks[1]-minor_ticks[0])
            yc=-YMAX+1.5*BMAJ/2.#abs(minor_ticks[1]-minor_ticks[0])
            # xc=0.9
            # yc=-0.9
        width= BMAJ
        height= BMIN
        pa=BPA
        elli=Ellipse((xc,yc),width,height,angle=90.-pa,linewidth=0,fill=True,color='white', alpha=1.0)
        #elli.set_clip_box(ax1.bbox)
        #elli.set_alpha(0.7)
        #elli.set_facecolor('black')
        #ax1.add_artist(elli)
        ax1.add_patch(elli)

    if star:
        # add star
        ax1.plot(xstar , ystar, marker='+', color=starcolor, markersize=3.5, mew=1.0)
    if planet:
        # add planet
        ax1.plot(xplt , yplt, marker='x', color=pltcolor, markersize=3.5, mew=1.0)

    if ruller:
        x1=-XMAX+2.0*abs(minor_ticks[1]-minor_ticks[0])
        x2=x1+100./dpc
        yc=-XMAX+2.0*abs(minor_ticks[1]-minor_ticks[0])
        plt.plot([x1,x2], [yc, yc], color='white')
        plt.text(x2, yc*0.95, '100 au', color='white')

    if title!='':
        xc=XMAX-1.0*abs(minor_ticks[1]-minor_ticks[0])
        yc=XMAX-2.0*abs(minor_ticks[1]-minor_ticks[0])
        plt.text(xc, yc, title, color='white')

    if not axes:
        plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0.17, bottom=0.01, right=0.97, top=1.0)
    print('saving...')

    plt.savefig(filename, dpi=500)#,format='png', dpi=500)
    print('saved')

    if show:
        plt.show()

    

def cartesian2polar(outcoords, inputshape, origin, fieldscale=1.): # by S. Perez
    """Coordinate transform for converting a polar array to Cartesian coordinates. 
    inputshape is a tuple containing the shape of the polar array. origin is a
    tuple containing the x and y indices of where the origin should be in the
    output array."""

    rindex, thetaindex = outcoords
    x0, y0 = origin

    theta = thetaindex * 2 * np.pi / (inputshape[0]-1)
    #theta = 2. * np.pi - theta
    
    y = rindex*np.cos(theta)/fieldscale
    x = rindex*np.sin(theta)/fieldscale

    #print "r",rindex,"theta",theta,"x",x,"y",y
    
    ix = -x + x0
    iy = y +  y0

    return (iy,ix)








### NOT USED

# def ellipse_old(x0,y0,phi,chi,a, PA):

#     ## phi is a position angle, but in the plane of the disc (i.e. inclined)
    
#     # x0,y0 ellipse center
#     # phi pa at which calculate x,y phi=0 is +y axis
#     # chi aspect ratio of ellipse with chi>1
#     # a semi-major axis
#     # a/chi semi-minor axis
#     # PA  pa of ellipse 0 is north and pi/2 is east

#     phipp= phi-PA
    
#     xpp = (a/chi) * np.sin(phipp) 
#     ypp =    a    * np.cos(phipp)

#     xp =  xpp*np.cos(PA) + ypp*np.sin(PA)
#     yp = -xpp*np.sin(PA) + ypp*np.cos(PA)
    
#     xc = xp + x0
#     yc = yp + y0
    
#     return xc , yc


# def radial_profile_fits_model(fitsfile, x0, y0, PA, inc, rmax,Nr, phis, arc='elipse', cube=False, rmin=0.):

    
#     fit1=pyfits.open(fitsfile)
#     if cube==False:
#         data1 	= get_last2d(fit1[0].data) #extraer matriz de datos

#     else:
#         data1   = get_last3d(fit1[0].data)
#         Nlam=data1.shape[0]
#     Np1=data1.shape[-1]
#     # print np.shape(data1)
#     header1=fit1[0].header
#     ps_deg1=float(header1['CDELT2'])
#     ps_arcsec1=ps_deg1*3600.0
#     ps_rad1=ps_deg1*np.pi/180.0

#     # change units from Jy/pix to Jy/arcsec
#     if header1['BUNIT']=='JY/PIXEL':

#         data1=data1/(ps_arcsec1**2)

#     if cube==False:
#         return radial_profile(data1, np.ones((Np1,Np1)), x0, y0, PA, inc, rmax,Nr, phis, rms=0.0, BMAJ_arcsec=1.0, ps_arcsec=ps_arcsec1, arc=arc, rmin=rmin)

#     else:
#         Srs=[]
#         for ilam in range(Nlam):
#             Sri= radial_profile(data1[ilam,:,:], np.ones((Np1,Np1)), x0, y0, PA, inc, rmax,Nr, phis, rms=0.0, BMAJ_arcsec=1.0, ps_arcsec=ps_arcsec1, arc=arc, rmin=rmin)
#             Srs.append(Sri)
#         return Srs


# def radial_profile_fits_image(fitsfile_pbcor, fitsfile_pb, x0, y0, PA, inc, rmax,Nr, phis, rms, error_std=False, arcsec2=False , arc='elipse', ret_beam=False):


#     fit1=pyfits.open(fitsfile_pbcor)
#     data1 = get_last2d(fit1[0].data) #extraer matriz de datos


#     # print np.shape(data1)
#     header1=fit1[0].header
#     ps_deg1=float(header1['CDELT2'])
#     ps_arcsec1=ps_deg1*3600.0
#     ps_rad1=ps_deg1*np.pi/180.0
#     Np1=len(data1[:,0])

#     if fitsfile_pb!='':
#         fit2=pyfits.open(fitsfile_pb)
#         data2 = get_last2d(fit2[0].data) #extraer matriz de datos
#     else:
#         data2=np.ones((Np1,Np1))

    
#     BMAJ=float(header1['BMAJ']) # deg
#     BMIN=float(header1['BMIN']) # deg
#     BPA=float(header1['BPA']) # deg

#     BMAJ_arcsec1=BMAJ*3600.0
#     BMIN_arcsec1=BMIN*3600.0
#     beam_area=np.pi*BMAJ_arcsec1*BMIN_arcsec1/(4.*np.log(2.))
    
#     print("beam = %1.2f x %1.2f" %(BMAJ_arcsec1, BMIN_arcsec1))
#     if arcsec2:
#         data1=data1/beam_area
#         rms=rms/beam_area
#         print('transforming to Jy/arcsec2')
#     if not ret_beam:
#         return radial_profile(data1, data2, x0, y0, PA, inc, rmax,Nr, phis, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, ps_arcsec=ps_arcsec1, error_std=error_std, arc=arc)
#     else:
#         return radial_profile(data1, data2, x0, y0, PA, inc, rmax,Nr, phis, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, ps_arcsec=ps_arcsec1, error_std=error_std, arc=arc), BMAJ_arcsec1




# def flux_profile(image, image_pb, x0, y0, PA, inc, rmax,Nr, rms,  BMAJ_arcsec, BMIN_arcsec, ps_arcsec, rs=np.array([]), refine=1, phi1=0.0, phi2=360.0, rmin=0.0):

#     # x0, y0 are RA DEC offsets in arcsec
#     # PA and inc are PA and inc of the disc in deg
#     # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
#     # Nr is the number of radial points to calculate
#     # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)
#     # refine makes the image with higher resolution such that it works well for inclined discs
#     # ################ SPATIAL GRID

#     # XY


#     if len(rs)==0:
#         rs=np.linspace(rmax/1.0e3,rmax,Nr)
#     rmax=np.max(rs)
#     Nr=len(rs)

#     Np=len(image[:,0])
    
#     if (refine>1.0 and refine<10.0):
#         ps_f=ps_arcsec*1.e3/refine
#         Npf=int(rmax/(ps_f/1.0e3)*2.)
#         image   =interpol(Np,Npf,ps_arcsec*1.e3,ps_f, image)
#         image_pb=interpol(Np,Npf,ps_arcsec*1.e3,ps_f, image_pb)

#         ### redefine values for image dimensions and pixel zie
#         Np=len(image[:,0])
#         ps_arcsec=ps_f/1.0e3

#     elif refine>10.0:
#         print('too much refining')
#         sys.exit()
        

#     xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)

    
#     # R phi

#     PA_rad=PA*np.pi/180.0
#     phi1_rad=simple_phi(phi1*np.pi/180.0 )
#     phi2_rad=simple_phi(phi2*np.pi/180.0 )

    
#     #print rs
    
#     ecc= np.sin(inc*np.pi/180.0)
#     chi=1.0/(np.sqrt(1.0-ecc**2.0)) # aspect ratio between major and minor axis (>=1)

#     Beam_area=np.pi*BMAJ_arcsec*BMIN_arcsec/(4.0*np.log(2.0)) # in arcsec2

    
#     ##### Calculate flux within rs
#     F=np.zeros((Nr,2))


#     xv, yv = np.meshgrid(xs-x0, ys-y0, sparse=False, indexing='xy')

#     xpp = xv * np.cos(PA_rad) - yv *np.sin(PA_rad) ### along minor axis
#     ypp = xv * np.sin(PA_rad) + yv *np.cos(PA_rad) ###  along major axis

#     PAs=np.arctan2(xv,yv)# -pi to pi
#     PAs[PAs<0.0]=PAs[PAs<0.0]+2.*np.pi # from 0 to 2pi, discontinuity in 0.0

#     if phi2_rad>=phi1_rad:
#         mask_phis=(PAs>= phi1_rad) & (PAs<=phi2_rad)
#     else: ### goes through zero
#         mask_phis=(PAs>= phi1_rad) | (PAs<=phi2_rad)
      
#     # jet=image_pb*1.
#     # jet[mask_phis]=-1.0
#     # plt.pcolor(jet)
#     # plt.show()
#     rdep=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 ) # real deprojected radius
#     rmsmap2=(rms/image_pb)**2.0
#     for i_r in range(Nr):
#         try:
#             mask_r=(rdep<=rs[i_r]) & (rdep>=rmin)
#         except:
#             mask_r=(rdep<=rs[i_r]) & (rdep>=rmin[i_r])
           
#         mask=mask_r & mask_phis
#         F[i_r,0]= np.sum(image[mask])*(ps_arcsec**2.0)/Beam_area
#         F[i_r,1]= np.sum(rmsmap2[mask]) # Jy/beam Note: /beam is ok as then it is correct

#         # Correct by number of independent points

#         if len(image[mask])*ps_arcsec**2.>Beam_area:
#             F[i_r,1]= np.sqrt(F[i_r,1]) * np.sqrt(ps_arcsec**2.0/Beam_area)
#         else:
#             F[i_r,1]=rms
#         # if  F[i_r,1]==0.0 and rs[i_r]**2.0*np.pi*np.cos(inc*np.pi/180):
#             # mask=
#             # F[i_r,1]= np.sum( rmsmap2[rdep<rs[i_r]])

#     # image[mask]=0.0  
#     # plt.pcolor(xs, ys, image)
#     # plt.contour(xs,ys,rdep, levels=rs[::3])
#     # plt.axes().set_aspect('equal')
#     # plt.show()
#     return np.array([rs, F[:,0], F[:,1]]) # rs, I, eI

# def flux_profile_edgeon(image, image_pb, x0, y0, PA, rmax,Nr, rms, BMAJ_arcsec, BMIN_arcsec, ps_arcsec, rs=np.array([]), refine=1, hv=0.05):

#     # x0, y0 are RA DEC offsets in arcsec
#     # PA and inc are PA and inc of the disc in deg
#     # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
#     # Nr is the number of radial points to calculate
#     # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)
#     # refine makes the image with higher resolution such that it works well for inclined discs
#     # ################ SPATIAL GRID

#     # XY


#     if len(rs)==0:
#         rs=np.linspace(rmax/1.0e3,rmax,Nr)
#     rmax=np.max(rs)
#     Nr=len(rs)

#     Np=len(image[:,0])
    
#     if (refine>1.0 and refine<10.0):
#         ps_f=ps_arcsec*1.e3/refine
#         Npf=int(rmax/(ps_f/1.0e3)*2.)
#         image   =interpol(Np,Npf,ps_arcsec*1.e3,ps_f, image)
#         image_pb=interpol(Np,Npf,ps_arcsec*1.e3,ps_f, image_pb)

#         ### redefine values for image dimensions and pixel zie
#         Np=len(image[:,0])
#         ps_arcsec=ps_f/1.0e3

#     elif refine>10.0:
#         print('too much refining')
#         sys.exit()
        
    
#     xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)

    
#     # R phi

#     PA_rad=PA*np.pi/180.0
#     # phis_rad=phis*np.pi/180.0 
#     # dphi=abs(phis_rad[1]-phis_rad[0])
#     # Nphi=len(phis_rad)
    
#     print(BMAJ_arcsec)
    
#     Beam_area=np.pi*BMAJ_arcsec*BMIN_arcsec/(4.0*np.log(2.0)) # in arcsec2

    
#     ##### Calculate flux within rs
#     F=np.zeros((Nr,2))


#     xv, yv = np.meshgrid(xs-x0, ys-y0, sparse=False, indexing='xy')

#     xpp = xv * np.cos(PA_rad) - yv *np.sin(PA_rad) ### along minor axis
#     ypp = xv * np.sin(PA_rad) + yv *np.cos(PA_rad) ###  along major axis

#     # PAs=np.arctan2(xpp,ypp)*180.0/np.pi
#     # rdep=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 ) # real deprojected radius

#     rmsmap2=(rms/image_pb)**2.0
    
#     for i_r in range(Nr):
#         mask=((np.abs(ypp)<rs[i_r]) & ( (np.abs(xpp)<BMAJ_arcsec/1.5) | (np.abs(xpp)<rs[i_r]*hv)))
        
#         F[i_r,0]= np.sum( image[mask]*(ps_arcsec**2.0)/Beam_area)
#         F[i_r,1]= np.sum( rmsmap2[mask]) # Jy/beam Note: /beam is ok as then it is correct

#         # Correct by number of independent points
#         if len(image[mask])*ps_arcsec**2.>Beam_area:
#             F[i_r,1]= np.sqrt(F[i_r,1]) * np.sqrt(ps_arcsec**2.0/Beam_area)
#         else:
#             F[i_r,1]=rms
#     print('Beam area / pixel area',Beam_area/ps_arcsec**2.)
#     print('Beam area = %1.3f arcsec2'%Beam_area)
#     print('%1.1f beams within %1.1f arcsec'%(len(image[mask].flatten())*ps_arcsec**2./Beam_area, rs[i_r]))        


#     # image[mask]=0.0  
#     # plt.pcolor(xs, ys, image)
#     # #plt.contour(xs,ys,rdep, levels=rs[::3])
#     # plt.axes().set_aspect('equal')
#     # plt.show()
#     return np.array([rs, F[:,0], F[:,1]]) # rs, I, eI



"""
def flux_azimuthal_profile(image, image_pb, x0, y0, PA, inc, rmin, rmax, NPA, rms,  BMAJ_arcsec, BMIN_arcsec, ps_arcsec):

    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)
    
    # ################ SPATIAL GRID

    # XY

    Np=len(image[:,0])

    xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)

    
    # R phi

    PA_rad=PA*np.pi/180.0
    # phis_rad=phis*np.pi/180.0 
    # dphi=abs(phis_rad[1]-phis_rad[0])
    # Nphi=len(phis_rad)

    
    ecc= np.sin(inc*np.pi/180.0)
    chi=1.0/(np.sqrt(1.0-ecc**2.0)) # aspect ratio between major and minor axis (>=1)

    Beam_area=np.pi*BMAJ_arcsec*BMIN_arcsec/(4.0*np.log(2.0)) # in arcsec2

    
    ##### Calculate flux within rmin, rmax and PA bins
    F=np.zeros((NPA,2))


    xv, yv = np.meshgrid(xs-x0, ys-y0, sparse=False, indexing='xy')

    xpp = xv * np.cos(PA_rad) - yv *np.sin(PA_rad) ### along minor axis
    ypp = xv * np.sin(PA_rad) + yv *np.cos(PA_rad) ###  along major axis


    rdep=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 ) # real deprojected radius

    PAs=np.arctan2(xv,yv) # rad

    print('rms =', rms)
    rmsmap2=(rms/image_pb)**2.0
    
    print(Beam_area)
    PA_array=np.linspace(-np.pi, np.pi, NPA+1)
    PA_mid=PA_array[:-1]+(PA_array[1]-PA_array[0])/2.
    for i_pa in range(NPA):
        
        mask=(rdep<=rmax) & (rdep>rmin) & (PAs>PA_array[i_pa]) & (PAs<=PA_array[i_pa+1])
    
        F[i_pa,0]= np.sum(image[mask]*(ps_arcsec**2.0)/Beam_area)
        F[i_pa,1]= np.sum( rmsmap2[mask]) # Jy/beam Note: /beam is ok as then it is correct

        # Correct by number of independent points
    
        F[i_pa,1]= np.sqrt(F[i_pa,1]) * np.sqrt(ps_arcsec**2.0/Beam_area)
        
    # plt.pcolor(xs, ys, image)
    # plt.contour(xs,ys,rdep)
    # plt.show()
    return np.array([PA_mid, F[:,0], F[:,1]]) # rs, I, eI 


def flux_profile_fits_image(fitsfile_pbcor, fitsfile_pb, x0, y0, PA, inc, rmax,Nr, rms, rs=np.array([]), refine=1.0, phi1=0.0, phi2=360.0, rmin=0.0):


    fit1=pyfits.open(fitsfile_pbcor)
    data1=get_last2d(fit1[0].data) 

    header1=fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_arcsec1=ps_deg1*3600.0
    ps_rad1=ps_deg1*np.pi/180.0
    Np1=len(data1[:,0])

    if fitsfile_pb!='':
        fit2=pyfits.open(fitsfile_pb)
        data2 = get_last2d(fit2[0].data) #extraer matriz de datos
    else:
        data2=np.ones((Np1,Np1))

    
    BMAJ=float(header1['BMAJ']) # deg
    BMIN=float(header1['BMIN']) # deg
    BPA=float(header1['BPA']) # deg

    BMAJ_arcsec1=BMAJ*3600.0
    BMIN_arcsec1=BMIN*3600.0
    if inc<90.0:
        return flux_profile(data1, data2, x0, y0, PA, inc, rmax,Nr, phi1=phi1, phi2=phi2, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, BMIN_arcsec=BMIN_arcsec1,  ps_arcsec=ps_arcsec1, rs=rs, refine=refine, rmin=rmin)
    elif inc==90.0:
        return flux_profile_edgeon(data1, data2, x0, y0, PA,  rmax,Nr, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, BMIN_arcsec=BMIN_arcsec1,  ps_arcsec=ps_arcsec1, rs=rs, refine=refine)
        
def flux_azimuthal_profile_fits_image(fitsfile_pbcor, fitsfile_pb, x0, y0, PA, inc, rmin, rmax, NPA,  rms):


    fit1=pyfits.open(fitsfile_pbcor)
    data1=get_last2d(fit1[0].data) 

    header1=fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_arcsec1=ps_deg1*3600.0
    ps_rad1=ps_deg1*np.pi/180.0
    Np1=len(data1[:,0])

    if fitsfile_pb!='':
        fit2=pyfits.open(fitsfile_pb)
        data2 = get_last2d(fit2[0].data) #extraer matriz de datos
    else:
        data2=np.ones((Np1,Np1))

    
    BMAJ=float(header1['BMAJ']) # deg
    BMIN=float(header1['BMIN']) # deg
    BPA=float(header1['BPA']) # deg

    BMAJ_arcsec1=BMAJ*3600.0
    BMIN_arcsec1=BMIN*3600.0
    
    return flux_azimuthal_profile(data1, data2, x0, y0, PA, inc, rmin, rmax, NPA, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, BMIN_arcsec=BMIN_arcsec1,  ps_arcsec=ps_arcsec1)

"""
