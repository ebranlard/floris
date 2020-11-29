"""
References:
    [1] E. Branlard and M. Gaunaa, Superposition of vortex cylinders for steady and unsteady simulation of rotors of finite tip-speed ratio - Wind Energy, 2014

    [X] E. Branlard, M. Gaunaa - Cylindrical vortex wake model: right cylinder - Wind Energy, 2014
    [X] E. Branlard, M. Gaunaa - Cylindrical vortex wake model: skewed cylinder, application to yawed or tilted rotors - Wind Energy, 2015
    [X] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
"""
#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
# import unittest
import numpy as np
import scipy.optimize as sciopt
import scipy.integrate as sciint


def Ct_const_cutoff(CT0,r_bar_cut,vr_bar,r_bar_tip=None):
    """ Returns an almost constant Ct, 
       linearly dropping to zero below a cut-off radius
       linearly dropping to zero after a tip radius
    """
    Ct=np.ones(vr_bar.shape)*CT0
    I=vr_bar<r_bar_cut
    Ct[I]=CT0*vr_bar[I]/r_bar_cut
    if r_bar_tip is not None:
        I=vr_bar>r_bar_tip
        Ct[I]=CT0*(1-(vr_bar[I]-r_bar_tip)/(1-r_bar_tip))
    return Ct 

def CirculationFromPrescribedCt(vr_bar,vCt,Lambda,bSwirl): 
    """ Finds k to match a given Ct 
    Solves equation (32) of [1]
    Based on script fGetInductions_Prescribed_CT_CQ2.py
    """
    vk = np.zeros(vr_bar.shape)
    vlambda_r = Lambda * vr_bar
    if bSwirl:
        for ir,(r_bar,lambda_r,Ct) in enumerate(zip(vr_bar,vlambda_r,vCt)):
            if Ct>0 and r_bar>0:
                res=sciopt.minimize_scalar(lambda k:np.abs(k*(1+k/(4*lambda_r**2))-Ct), bounds=[0,1.8], method='bounded')
                vk[ir] = res.x
    else:
        vk = vCt
    return vk

def WakeVorticityFromCirculation_Discr(r_cp,Gamma_cp,R,U0,Omega,nB,bSwirl,method='analytical',bTanInd_k=True,bHighThrustCorr=True,r_cyl=None):
    """ 
    Implements discrete algorithm from [1] to find gamma_t, and the inductions
    """
    r_cp     = np.asarray(r_cp).ravel()
    Gamma_cp = np.asarray(Gamma_cp).ravel()
    if r_cyl is None:
        r_cyl    = np.zeros(r_cp.shape)
        if len(r_cyl)==1:
            r_cyl=np.array([R])
        else:
            r_cyl[:-1] = (r_cp[1:] + r_cp[:-1])/2
            r_cyl[-1]  = r_cp[-1] + (r_cp[-1]-r_cp[-2])/2
    else:
        r_cyl    = np.asarray(r_cyl).ravel()
    misc={}
    # --- Checks
    if np.amax(r_cyl) < np.amax(r_cp):
        warnings.warn('Cylinders are assumed to enclose the control points')
        import pdb; pdb.set_trace()
    if r_cyl[0] == 0:
        raise Exception('First cylinder at 0, impossible')

    ##  Interpolating Circulation on the control points
    Gamma_cp   = Gamma_cp * nB # IMPORTANT
    lambda_rcp = Omega*r_cp/U0
    k_r        = Gamma_cp*Omega/(np.pi*U0**2)      # Equation (27) of [Superp]
    Ct_KJ_cp   = k_r * (1+k_r / (4*lambda_rcp**2)) # Equation (32) of [Superp]
    ## Some init

    n = len(r_cyl)
    # Critical values for high-thrust Spera correction
    ac   = 0.34
    Ct_c = 4*ac*(1-ac)
    # ---  Finding Cylinders intensities
    if method.lower()=='analytical':
        # ---  Case 1: Using k / ct to determine gamma
        # Using a_prime to correct Ct
        # Tangential convection velocity
        if not bTanInd_k:
            ap_cyl_conv = 0 * Gamma_cp
        else:
            # Tangential convection velocity as the mean of velocity on both side of the cylinder)
            # See Equation (26) of [1]:  a'_c,i = (\Gamma_i+\Gamma_i+1)/(4 pi Omega R_i^2)
            ap_cyl_conv      = np.zeros(Gamma_cp.shape)
            ap_cyl_conv[:-1] = (Gamma_cp[:-1] + Gamma_cp[1:])/(4*np.pi*Omega*r_cyl[:-1])**2
            ap_cyl_conv[-1]  = Gamma_cp[-1]/(4*np.pi*Omega*r_cyl[-1])**2
        # --- Allocation
        b           = np.zeros(n) # Cummulative sum of gamma_bar
        S           = np.zeros(n) # Term under the square root
        Sbis        = np.zeros(n) # Term under the square root, account
        Ctrot       = np.zeros(n)
        Ctrot_loc   = np.zeros(n)
        Cti         = np.zeros(n)
        Cteff       = np.zeros(n)
        gamma_t_bar = np.zeros(n)
        # --- Init
        C        = Gamma_cp*Omega/(np.pi*U0**2)*(1 + ap_cyl_conv)
        lambda_R = Omega * r_cyl / U0
        # Looping through radii in reversed order
        for i in np.arange(n-1,-1,-1):
            if i==n-1:
                b[i]         = 1
                S[i]         = 1 - C[i]
                Sbis[i]      = 1 - C[i]
                Ctrot[i]     = 0
                Ctrot_loc[i] = 0
            else:
                b[i]         = 1 + np.sum(gamma_t_bar[i+1:])                         # Eq.(25)
                S[i]         = b[i]**2 - (k_r[i] - k_r[i+1])*(1 + ap_cyl_conv[i])    # Eq.(28)
                Ctrot_loc[i] = (k_r[i+1]/2)**2*(1/lambda_R[i]**2-1/lambda_R[i+1]**2) # Eq.(33)
            if not bSwirl:
                Ctrot[i] = 0
                Cti[i]   = k_r[i]
            else:
                Ctrot[i] = np.sum(Ctrot_loc[i+1:])
                Cti[i]   = k_r[i]*(1+k_r[i]/(4*lambda_R[i]**2)) # Eq.(32)
            Cteff[i] = Cti[i] - Ctrot[i] #Eq.(38)
            Sbis[i] = 1 - Cteff[i]
            if bSwirl :
                gamma_t_bar[i] = - b[i] + np.sqrt(Sbis[i]) #Eq.(28)
            else:
                gamma_t_bar[i] = - b[i] + np.sqrt(S[i]) 
#             print('b',b,'S',S)
            if bHighThrustCorr:
                if (Cteff[i] > Ct_c):
                    aeff = (Cteff[i] - 4*ac**2)/(4*(1-2*ac)) # Eq.(40)
                    gamma_t_bar[i] = - b[i] + (1-2*aeff)     # Eq.(41)
            #if np.abs(imag(gamma_t_bar(i))) > 0:
            #    gamma_t_bar[i] = - b[i] + 0
        Vc      = U0 * (b + gamma_t_bar / 2)
        h       = 2*np.pi*Vc/(Omega*(1+ap_cyl_conv))
        #print('Vc',Vc,'b',b,'gamma_t_bar',gamma_t_bar,'ap',ap_cyl_conv,'Omega',Omega,'h',h)


        # Analytical 1
        a_1=-np.cumsum(gamma_t_bar[-1::-1]/2);
        a_1=a_1[-1::-1]
        #if isequal(Algo.VCYL.PitchMethod,'Analytical')
        a=a_1;
        #elif isequal(Algo.VCYL.PitchMethod,'WrongAnalytical')
        #    # Analytical 2
        #    a_2=-(gamma_t_hat/2)/U0;
        #    a=a_2;
        #else
        #    a=a_1;
        #end
        if bSwirl:
            a_prime=Gamma_cp/(4*np.pi*Omega*r_cp**2)
        else:
            a_prime=a*0

        misc['a'] = a
        misc['a_prime'] = a_prime
        misc['Gamma_cp'] = Gamma_cp
        misc['r_cp'] = r_cp
        misc['h']     = h

        # --- Vortex intensities
        Gamma_tilde = Gamma_cp - np.concatenate((Gamma_cp[1:],[0])) #Gamma_tilde = Gamma_i-Gamma_{i+1}
        #gamma_t     = - Gamma_tilde/h
        gamma_t = gamma_t_bar * U0
        if bSwirl:
            gamma_l     =   Gamma_tilde/(2*np.pi*r_cp)
            Gamma_r     = - Gamma_cp[0]
        else:
            gamma_l     = None
            Gamma_r     = None

        return gamma_t,gamma_l,Gamma_r, misc

# --------------------------------------------------------------------------------}
# --- Main interfaces 
# --------------------------------------------------------------------------------{
def WakeVorticityFromCt(r,Ct,R,U0,Omega):
    """ Returns the wake vorticity intensity for a given Ct"""
    if Omega==np.inf:
        Omega=800 # TODO, implement infinite lambd formulae directly
    Lambda=Omega*R/U0
    bSwirl=Lambda<20
    vk=CirculationFromPrescribedCt(r/R,Ct,Lambda,bSwirl)
    Gamma = vk*np.pi*U0**2/Omega
    return WakeVorticityFromCirculation_Discr(r,Gamma,R,U0,Omega,nB=1,bSwirl=bSwirl)

def WakeVorticityFromGamma(r,Gamma,R,U0,Omega):
    """ Returns the wake vorticity intensity for a given circulation """
    if Omega==np.inf:
        Omega=800 # TODO, implement infinite lambd formulae directly
    Lambda=Omega*R/U0
    bSwirl=Lambda<20
    return WakeVorticityFromCirculation_Discr(r,Gamma,R,U0,Omega,nB=1,bSwirl=bSwirl)