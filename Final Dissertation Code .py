# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:04:04 2020

@author: Stevo
"""

import numpy as np, matplotlib.pyplot as plt

solution_volume=120 # Total volume of the solution [cm^3]
beaker_radius=3.5 #Radius of beaker [cm]
solution_height=solution_volume/(np.pi*beaker_radius**2) #Height of the collumn of solution [cm]
h_top=1e-2 # Thickness of the top layer of the solution [cm]
h_bot=1e-2 # Thickness of the bottom layer of the solution [cm]
h_bulk=solution_height-h_top-h_bot ## Thickness of the middle layer of the solution [cm]
surf_bulk=h_top/h_bulk # Ratio of the volumes of the top and middle layers of the solution [cm]
bot_bulk=h_bot/h_bulk # Ratio of the volumes of the bottom and middle layers of the solution [cm]

MB_top=np.array([4.17e-9]) # Concentration of MB in the top layer of solution [mol cm^{-3}]
MB_bulk=np.array([4.17e-9]) # Concentration of MB in the middle layer of solution [mol cm^{-3}]
MB_bot=np.array([4.17e-9])  # Concentration of MB in the bottom layer of solution [mol cm^{-3}]

O2_top=np.array([2.56e-7]) #256e-9 Concentration of O2 in the top layer of solution [mol cm^{-3}]
O2_bulk=np.array([2.56e-7])  #256e-9,2.559973287518336550e-07 Concentration of O2 in the bottom layer of solution [mol cm^{-3}]
O2_bot=np.array([2.56e-7]) #256e-9 Concentration of O2 in the bottom layer of solution [mol cm^{-3}]

t=np.array([0])
dt=1
X=1e-4 # Photocatalyst thickness [cm]
A = 78.5 # Cross-sectional area of the photocatalyst [cm^2]
IF=2.49e15 # Incident flux of photons [cm^{-2}s{-1}]
T = 0.949 # Transmission of photons into the photocatalyst [unitless]
I_0 = T*IF # Flux of photons entering the photocatalyst [cm^{-2}s{-1}]
alpha =3.33e4 #Absorption coefficient [cm^{-1}]
phi = 5.7e-2 # Quantum yield of electron-hole pair formation [unitless]
tau = 1e-9 # Electron-hole lifetime in the the 'bulk' photocatalyst [s]
h=1e-6 # Layer thickness over which the flux of charge carriers arrive at/ leave the photocatalyst surface.

D_n = 0.01  # Diffusion coefficient of electrons [cm^2 s^{-1}]
L_n = np.sqrt(D_n*tau) # Electron diffusion length [cm]
v_n = np.sqrt(D_n/tau) # Electron diffusion velocity [cm s^{-1}]
Lambda_n = (1/3)*(phi*alpha*I_0*tau)/(1-(alpha*L_n)**2) # Value of Lambda for electrons [cm^-3]

D_p = 0.01  # Diffusion coefficient of holes [cm^2 s^{-1}]
L_p = np.sqrt(D_p*tau) # Hole diffusion length [cm]
v_p = np.sqrt(D_p/tau) # Hole diffusion velocity [cm s^{-1}]
Lambda_p = (1/3)*(phi*alpha*I_0*tau)/(1-(alpha*L_p)**2) # Value of Lambda for holes [cm^-3]

# Rate constants used for simulating the reactions
k_red = 1e-18 # Rate constant for the reduction of molecular oxygen [cm^3 s^{-1}]
k_1 = 1e-8 # Rate constant for the oxidation of surface hydroxyl groups [cm^2 s^{-1}]
k_rec_1=1e-9 # Rate constant for electron-trapped hole rcombination [cm^3 s^{-1}]
k_rec_2=1e-8 # Rate constant for trapped electron-hole rcombination [cm^3 s^{-1}]
k_T_trap = 1e4 # Trapping rate constant multiplied by the trap concentration [s^{-1}]
k_gen=1e6 # General rate constant for the break down of intermediate species [mol^{-1} cm^3 s^{-1}]

# Rate constants for the radical reactions (may need double checking)
k_a=2.7e4 # Rate conctant for the reaction between hydroxyl radicals and hydrogen peroxide [mol^{-1} cm^3 s^{-1}]
k_b=7.5e6 # Rate conctant for the reaction between hydroxyl radicals and peroxide ions[mol^{-1} cm^3 s^{-1}]
k_c=8e6 # Rate conctant for the reaction between hydroxyl radicals and superoxide radicals [mol^{-1} cm^3 s^{-1}]
k_d=6.6e6 # Rate conctant for the reaction between hydroxyl radicals and peroxide radicals [mol^{-1} cm^3 s^{-1}]
k_e=5.5e6 # Rate conctant for the reaction between two hydroxyl radicals [mol^{-1} cm^3 s^{-1}]
k_f=3.7e-3 # Rate conctant for the reaction between peroxide radicals and hydrogen peroxide [mol^{-1} cm^3 s^{-1}]
k_g=8.6e2 # Rate conctant for the reaction between two peroxide radicals [mol^{-1} cm^3 s^{-1}]
k_h=0.13e-3 # Rate conctant for the reaction between superoxide radicals radicals and hydrogen peroxide [mol^{-1} cm^3 s^{-1}]
k_i=9.7e4 # Rate conctant for the reaction between superoxide radicals and peroxide radicals [mol^{-1} cm^3 s^{-1}]

#Concentrations of the different species in the reaction
top_OH_rad = np.zeros(1) # Concentration of free OH radicals in the top layer [mol cm^{-3}]
bulk_OH_rad = np.zeros(1) # Concentration of free OH radicals in the middle layer [mol cm^{-3}]
bot_OH_rad = np.zeros(1) # Concentration of free OH radicals in the bottom layer [mol cm^{-3}]

OH_rad_s = np.zeros(1) # Concentration of surface OH radicals [mol cm^{-3}]

OH_ion_s = np.ones(1)*3e14 # Concentration of surface OH ions [cm^{-2}]

top_H2O2 = np.zeros(1) # Concentration of H2O2 in the top layer [mol cm^{-3}]
bulk_H2O2 = np.zeros(1) # Concentration of H2O2 in the middle layer [mol cm^{-3}]
bot_H2O2 = np.zeros(1) # Concentration of H2O2 in the bottom layer [mol cm^{-3}]

top_HO2_rad = np.zeros(1) # Concentration of HO2 radicals in the top layer [mol cm^{-3}]
bulk_HO2_rad = np.zeros(1) # Concentration of HO2 radicals in the middle layer [mol cm^{-3}]
bot_HO2_rad = np.zeros(1) # Concentration of HO2 radicals in the bottom layer [mol cm^{-3}]

top_HO2_ion = np.zeros(1) # Concentration of HO2 ions in the top layer [mol cm^{-3}]
bulk_HO2_ion = np.zeros(1) # Concentration of HO2 ions in the middle layer [mol cm^{-3}]
bot_HO2_ion = np.zeros(1) # Concentration of HO2 ions in the bottom layer [mol cm^{-3}]

top_O2_rad = np.zeros(1) # Concentration of O2 radicals in the top layer [mol cm^{-3}]
bulk_O2_rad = np.zeros(1) # Concentration of O2 radicals in the middle layer [mol cm^{-3}]
bot_O2_rad = np.zeros(1) # Concentration of O2 radicals in the bottom layer [mol cm^{-3}]

top_H_ion = np.ones(1)*1e-10 #1e-10,2.144397134984911162e-07 Concentration of free H ions in the top layer [mol cm^{-3}]
bulk_H_ion = np.ones(1)*1e-10 #1e-10,2.144397134984911162e-07 Concentration of free H ions in the middle layer [mol cm^{-3}]
bot_H_ion = np.ones(1)*1e-10 #1e-10,2.144397134984911162e-07 Concentration of free H ions in the bottom layer [mol cm^{-3}]
H_H_ion = -774 # Salting-out coefficient of H ions [cm^3 mole^{-1}]

top_Cl_ion = np.ones(1)*1*1e-9 # Concentration of free Cl ions in the top layer [mol cm^{-3}]
bulk_Cl_ion = np.ones(1)*1*1e-9 # Concentration of free Cl ions in the middle layer [mol cm^{-3}]
bot_Cl_ion = np.ones(1)*1*1e-9 # Concentration of free Cl ions in the bottom layer [mol cm^{-3}]
H_Cl_ion = 29962 # Salting-out coefficient of Cl ions [cm^3 mole^{-1}]

top_OH_ion = np.ones(1)*1e-10 #1e-10,1.722076261290478890e-09 Concentration of free OH ions in the top layer [mol cm^{-3}]
bulk_OH_ion = np.ones(1)*1e-10 #1e-10 Concentration of free OH ions in the middle layer [mol cm^{-3}]
bot_OH_ion = np.ones(1)*1e-10 #1e-10 Concentration of free OH ions in the bottom layer [mol cm^{-3}]
H_OH_ion = 15997 # Salting-out coefficient of OH^- ions [cm^3 molecule^{-1}]

top_NO3_ion =np.zeros(1)#np.ones(1)*4.072919917830853071e-07 #,2.129743531505021857e-07 # Concentration of NO3 ions in the top layer [mol cm^{-3}]
bulk_NO3_ion =np.zeros(1)#np.ones(1)*4.072919917830853071e-07 #np.zeros(1),2.129743531505021857e-07 # Concentration of NO3 ions in the middle layer [mol cm^{-3}]
bot_NO3_ion =np.zeros(1)#np.ones(1)*4.072919917830853071e-07 #np.zeros(1),2.129743531505021857e-07 # Concentration of NO3 ions in the bottom layer [mol cm^{-3}]
H_NO3_ion = 4972 # Salting-out coefficient of NO3 ions [cm^3 mol^{-1}]

top_SO4_ion =np.zeros(1)#np.ones(1)*1.156256337517344000e-08 #np.zeros(1),6.047547675041391240e-09 # Concentration of SO4 ions in the top layer [mol cm^{-3}]
bulk_SO4_ion =np.zeros(1)# np.ones(1)*1.156256337517344000e-08#np.zeros(1), 6.047547675041391240e-09 # Concentration of SO4 ions in the middle layer [mol cm^{-3}]
bot_SO4_ion =np.zeros(1)#np.ones(1)*1.156256337517344000e-08 #np.zeros(1), 6.047547675041391240e-09 # Concentration of SO4 ions in the bottom layer [mol cm^{-3}]
H_SO4_ion = 43515 # Salting-out coefficient of SO4 ions [cm^3 mol^{-1}]

top_Int_1 =np.zeros(1)# np.ones(1)*1.453147429585184223e-15 #np.zeros(1),3.920705514947080628e-14 Concentration of Int_1 in the top layer [mol cm^{-3}]
bulk_Int_1 =np.zeros(1)# np.ones(1)*1.453147429585184223e-15 #np.zeros(1),3.920705514947080628e-14 Concentration of Int_1 in the middle layer [mol cm^{-3}]
bot_Int_1 = np.zeros(1)#np.ones(1)*1.453147429585184223e-15 #np.zeros(1),3.920705514947080628e-14 Concentration of Int_1 in the bottom layer [mol cm^{-3}]

top_Int_2 =np.zeros(1)# np.ones(1)*1.221956786819697820e-15 # np.zeros(1),3.330499843670521525e-14Concentration of Int_2 in the top layer [mol cm^{-3}]
bulk_Int_2 =np.zeros(1)# np.ones(1)*1.221956786819697820e-15 #np.zeros(1) Concentration of Int_2 in the middle layer [mol cm^{-3}]
bot_Int_2 = np.zeros(1)#np.ones(1)*1.221956786819697820e-15 # np.zeros(1)Concentration of Int_2 in the bottom layer [mol cm^{-3}]

top_Int_3 =np.zeros(1)# np.ones(1)*1.884441457680425942e-15#np.zeros(1) 5.204719730849766593e-14 Concentration of Int_3 in the top layer [mol cm^{-3}]
bulk_Int_3 =np.zeros(1)# np.ones(1)*1.884441457680425942e-15 #np.zeros(1) 5.204719730849766593e-14 Concentration of Int_3 in the middle layer [mol cm^{-3}]
bot_Int_3 =np.zeros(1)#np.ones(1)*1.884441457680425942e-15 # np.zeros(1),5.204719730849766593e-14 Concentrationof Int_3 in the bottom layer [mol cm^{-3}]

top_Int_4 =np.zeros(1)#np.ones(1)*3.322341949479200566e-14 # np.zeros(1),1.085587439363534479e-12Concentration of Int_4 in the top layer [mol cm^{-3}]
bulk_Int_4 =np.zeros(1)# np.ones(1)*3.322341949479200566e-14 #np.zeros(1),1.085587439363534479e-12 Concentration of Int_4 in the middle layer [mol cm^{-3}]
bot_Int_4 =np.zeros(1)# np.ones(1)*3.322341949479200566e-14 # np.zeros(1),1.085587439363534479e-12Concentration of Int_4 in the bottom layer [mol cm^{-3}]

top_Int_5 =np.zeros(1)# np.ones(1)*3.330534812195046516e-14 #np.zeros(1),1.088005139984968847e-12 Concentration of Int_1 in the top layer [mol cm^{-3}]
bulk_Int_5 =np.zeros(1)# np.ones(1)*3.330534812195046516e-14 #np.zeros(1),1.088005139984968847e-12 Concentration of Int_1 in the middle layer [mol cm^{-3}]
bot_Int_5 =np.zeros(1)# np.ones(1)*3.330534812195046516e-14 #np.zeros(1),1.088005139984968847e-12 Concentration of Int_1 in the bottom layer [mol cm^{-3}]

top_N_OH_3 =np.zeros(1)# np.ones(1)*1.449669636574738723e-13 # np.zeros(1), 4.701165645699818952e-12Concentration of N(OH)3 in the top layer [mol cm^{-3}]
bulk_N_OH_3 =np.zeros(1)# np.ones(1)*1.449669636574738723e-13 # np.zeros(1), 4.701165645699818952e-12Concentration of N(OH)3 in the middle layer [mol cm^{-3}]
bot_N_OH_3 =np.zeros(1)# np.ones(1)*1.449669636574738723e-13#np.zeros(1), 4.701165645699818952e-12 Concentration of N(OH)3 in the bottom layer [mol cm^{-3}]

top_NH2_rad = np.zeros(1) # Concentration of NH2 radicals in the top layer [mol cm^{-3}]
bulk_NH2_rad = np.zeros(1) # Concentration of NH2 radicals in the middle layer [mol cm^{-3}]
bot_NH2_rad = np.zeros(1) # Concentration of NH2 radicals in the bottom layer [mol cm^{-3}]

top_CH3OH =np.zeros(1)# np.ones(1)*6.769588443918258612e-15 #np.zeros(1),1.850855748021801828e-13 Concentration of CH3OH in the top layer [mol cm^{-3}]
bulk_CH3OH =np.zeros(1)# np.ones(1)*6.769588443918258612e-15 #np.zeros(1),1.850855748021801828e-13 Concentration of CH3OH in the middle layer [mol cm^{-3}]
bot_CH3OH =np.zeros(1)# np.ones(1)*6.769588443918258612e-15#np.zeros(1),1.850855748021801828e-13 Concentration of CH3OH in the bottom layer [mol cm^{-3}]

top_CO2 = np.zeros(1) # Concentration of CO2 in the top layer [mol cm^{-3}]
bulk_CO2 = np.zeros(1) # Concentration of CO2 in the middle layer [mol cm^{-3}]
bot_CO2 = np.zeros(1) # Concentration of CO2 in the bottom layer [mol cm^{-3}]

top_TOC = np.ones(1)*16*4.17e-9 # Concentration of CO2 in the top layer [mol cm^{-3}]
bulk_TOC = np.ones(1)*16*4.17e-9 # Concentration of CO2 in the middle layer [mol cm^{-3}]
bot_TOC = np.ones(1)*16*4.17e-9 # Concentration of CO2 in the bottom layer [mol cm^{-3}]

OH_rad_s_rate = np.zeros(1) # Rate of surface hydroxyl radical formation [mol cm^{-3} s^{-1}]

n_s = float(v_n*Lambda_n/((k_rec_1*OH_ion_s+k_T_trap)*h+v_n)) # concentration of electrons at the photocatalyst surface [cm^{-3}]
A=k_rec_2*(v_p+h*k_1*OH_ion_s[-1]) # Value of A used to solve the quadratic formular for the concentration of holes at the photocatalyst surface
B=k_red*1.54e17*(v_p+h*k_1*OH_ion_s)+k_rec_2*(h*k_T_trap*n_s-v_p*Lambda_p) # Value of B used to solve the quadratic formular for the concentration of holes at the photocatalyst surface
C=-v_p*k_red*1.54e17*Lambda_p # Value of C used to solve the quadratic formular for the concentration of holes at the photocatalyst surface
p_s=float((-B+np.sqrt(B**2-4*A*C))/(2*A)) # Concentration of holes at the photocatalyst surface [cm^{-3}]
oc_trap = float(n_s*k_T_trap/(k_rec_2*p_s+k_red*1.54e17)) # Concentration of occupied electron traps at the photocatalyst surface [cm^{-3}]

D_O2=2.2e-5 # Diffusion coefficient for oxygen flux [cm^2 s^-1]
ff=1 # fudge factor
a=ff*D_O2/(h_bot**2) # Diffusion coefficient for the change in oxygen con centration [s^-1]
al=3.58
am=2.48
an=ff*(1.72e-5)/(h_bot**2)
aS=ff*(1.74e-5)/(h_bot**2)
ach3oh=ff*(1.7e-5)/(h_bot**2)
aco2=ff*(1.9e-5)/(h_bot**2)
FF=0.1
FF2=0.1
k_app=0.21*4.04e-4 # Apparent rate constant for removal of MB [s^{-1}] 5.35e-5 0.161

while bulk_CO2[-1] <0.9999*16*4.17e-9: #179790,240800 262810
    nt = np.array([t[-1] + dt]) #Calculates the latest elapsed time value
    t = np.hstack([t, nt]) # Constructs an array of elapsed time values
    
    
    nMB_top=np.array([MB_top[-1]+dt*a*((MB_top[-1]+MB_bulk[-1])/2-MB_top[-1])/al])
    MB_top = np.hstack([MB_top, nMB_top])
    
    nMB_bulk=np.array([MB_bulk[-1]+dt*a*((MB_top[-1]+MB_bot[-1])/2-MB_bulk[-1])/al])
    MB_bulk = np.hstack([MB_bulk, nMB_bulk])
    
    nMB_bot=np.array([MB_bot[-1]+dt*(a*((MB_bulk[-1]+MB_bot[-1])/2-MB_bot[-1])/al-k_app*MB_bot[-1]*((OH_rad_s[-1]+dt*(v_p*Lambda_p*k_1*OH_ion_s[-1]/((v_p+h*(k_1*OH_ion_s[-1]+(k_rec_2*k_T_trap*n_s)/(k_rec_2*p_s+k_red*O2_bot[-1]*(6.02e23))))*(6.02e23))))/7.52692727587e-09))])
    MB_bot = np.hstack([MB_bot, nMB_bot])
    
    beta= 0.5*(top_H_ion[-1]*H_H_ion+top_Cl_ion[-1]*H_Cl_ion+top_OH_ion[-1]*H_OH_ion+top_NO3_ion[-1]*H_NO3_ion+4*top_SO4_ion[-1]*H_SO4_ion) # The exponential factor for the decay in oxygen solubility [dimensionless]
    nO2_top=np.array([(256e-9)*np.exp(-beta)])
    O2_top = np.hstack([O2_top, nO2_top])
    
    nO2_bulk=np.array([O2_bulk[-1]+dt*(a*((O2_top[-1]+O2_bot[-1])/2-O2_bulk[-1])+k_c*bot_HO2_rad[-1]*bulk_H2O2[-1]+k_d*bulk_OH_rad[-1]*bulk_HO2_rad[-1]+k_f*bulk_HO2_rad[-1]*bulk_H2O2[-1]+k_g*bulk_HO2_rad[-1]**2+k_h*bulk_O2_rad[-1]*bulk_H2O2[-1]+k_i*bulk_O2_rad[-1]*bulk_HO2_rad[-1])])
    O2_bulk = np.hstack([O2_bulk, nO2_bulk])
    
    nO2_bot=np.array([O2_bot[-1]+dt*(a*((O2_bulk[-1]+O2_bot[-1])/2-O2_bot[-1])-k_red*oc_trap*O2_bot[-1]+k_c*bot_HO2_rad[-1]*bot_H2O2[-1]+k_d*bot_OH_rad[-1]*bot_HO2_rad[-1]+k_f*bot_HO2_rad[-1]*bot_H2O2[-1]+k_g*bot_HO2_rad[-1]**2+k_h*bot_O2_rad[-1]*bot_H2O2[-1]+k_i*bot_O2_rad[-1]*bot_HO2_rad[-1])])
    O2_bot = np.hstack([O2_bot, nO2_bot])
    
    
#    nO2_top=np.array([(256e-9)])
#    O2_top = np.hstack([O2_top, nO2_top])
#    
#    nO2_bulk=np.array([256e-9])
#    O2_bulk = np.hstack([O2_bulk, nO2_bulk])
#    
#    nO2_bot=np.array([256e-9])
#    O2_bot = np.hstack([O2_bot, nO2_bot])
    
    nOH_ion_s = np.array([3e14])
    OH_ion_s = np.hstack([OH_ion_s, nOH_ion_s])
    
    nOH_rad_s=np.array([OH_rad_s[-1]+dt*(v_p*Lambda_p*k_1*OH_ion_s[-1]/((v_p+h*(k_1*OH_ion_s[-1]+(k_rec_2*k_T_trap*n_s)/(k_rec_2*p_s+k_red*O2_bot[-1]*(6.02e23))))*(6.02e23)))-0.99*OH_rad_s[-1]-FF*dt*k_rec_1*n_s*OH_rad_s[-1]]) # Calculates the new concentration of hydroxyl radicals bound to the photocatalyst [mu mol cm{-3}]
    OH_rad_s = np.hstack([OH_rad_s, nOH_rad_s[-1]])
    
    ntop_OH_rad=np.array([top_OH_rad[-1]+a*((top_OH_rad[-1]+bulk_OH_rad[-1])/2-top_OH_rad[-1])*dt+dt*(k_f*top_HO2_rad[-1]*top_H2O2[-1]+k_h*top_O2_rad[-1]*top_H2O2[-1]-top_OH_rad[-1]*(k_a*top_H2O2[-1]+k_b*top_HO2_ion[-1]+k_c*top_O2_rad[-1]+k_d*top_HO2_rad[-1]+k_e*top_OH_rad[-1]+k_gen*(top_Int_1[-1]+top_Int_2[-1]+top_Int_3[-1]+top_Int_4[-1]+top_Int_5[-1])))]) # Concentration of free hydroxy radicals at the air-solution interface [mu mol cm^{-3}]
    top_OH_rad = np.hstack([top_OH_rad, ntop_OH_rad])
    
    nbulk_OH_rad=np.array([bulk_OH_rad[-1]+a*((top_OH_rad[-1]+bot_OH_rad[-1]/2-bulk_OH_rad[-1])*dt+dt*(k_f*bulk_HO2_rad[-1]*bulk_H2O2[-1]+k_h*bulk_O2_rad[-1]*top_H2O2[-1]-bulk_OH_rad[-1]*(k_a*bulk_H2O2[-1]+k_b*bulk_HO2_ion[-1]+k_c*bulk_O2_rad[-1]+k_d*bulk_HO2_rad[-1]+k_e*bulk_OH_rad[-1]+k_gen*(bulk_Int_1[-1]+bulk_Int_2[-1]+bulk_Int_3[-1]+bulk_Int_4[-1]+bulk_Int_5[-1]))))]) # Concentration of free hydroxy radicals at the air-solution interface [mu mol cm^{-3}]
    bulk_OH_rad = np.hstack([bulk_OH_rad, nbulk_OH_rad])
    
    nbot_OH_rad=np.array([bot_OH_rad[-1] +0.99*OH_rad_s[-1]+a*((bulk_OH_rad[-1]+bot_OH_rad[-1])/2-bot_OH_rad[-1])*dt+dt*(k_f*bot_HO2_rad[-1]*bot_H2O2[-1]+k_h*bot_O2_rad[-1]*bot_H2O2[-1]-bot_OH_rad[-1]*(k_a*bot_H2O2[-1]+k_b*bot_HO2_ion[-1]+k_c*bot_O2_rad[-1]+k_d*bot_HO2_rad[-1]+k_e*bot_OH_rad[-1]+k_gen*(bot_Int_1[-1]+bot_Int_2[-1]+bot_Int_3[-1]+bot_Int_4[-1]+bot_Int_5[-1])))]) # Concentration of free hydroxy radicals at the air-solution interface [mu mol cm^{-3}]
    bot_OH_rad = np.hstack([bot_OH_rad, nbot_OH_rad])
    
    ntop_Int_1=np.array([top_Int_1[-1]+dt*(a*((top_Int_1[-1]+bulk_Int_1[-1])/2-top_Int_1[-1])/al-FF*top_OH_rad[-1]*k_gen*top_Int_1[-1]/4)])
    top_Int_1 = np.hstack([top_Int_1, ntop_Int_1])
    nbulk_Int_1=np.array([bulk_Int_1[-1]+dt*(a*((top_Int_1[-1]+bot_Int_1[-1])/2-bulk_Int_1[-1])/al-FF*bulk_OH_rad[-1]*k_gen*bulk_Int_1[-1]/4)])
    bulk_Int_1 = np.hstack([bulk_Int_1, nbulk_Int_1])   
    nbot_Int_1=np.array([bot_Int_1[-1]+dt*(a*((bulk_Int_1[-1]+bot_Int_1[-1])/2-bot_Int_1[-1])/al+k_app*MB_bot[-1]*((OH_rad_s[-1]+dt*(v_p*Lambda_p*k_1*OH_ion_s[-1]/((v_p+h*(k_1*OH_ion_s[-1]+(k_rec_2*k_T_trap*n_s)/(k_rec_2*p_s+k_red*O2_bot[-1]*(6.02e23))))*(6.02e23))))/7.52692727587e-09)-FF*bot_OH_rad[-1]*k_gen*bot_Int_1[-1]/4)])
    bot_Int_1 = np.hstack([top_Int_1, nbot_Int_1])
    
    ntop_Int_2=np.array([top_Int_2[-1]+dt*(a*((top_Int_2[-1]+bulk_Int_2[-1])/2-top_Int_2[-1])/al+FF*top_OH_rad[-1]*k_gen*(top_Int_1[-1]/4-top_Int_2[-1]/3))])
    top_Int_2 = np.hstack([top_Int_2, ntop_Int_2])
    nbulk_Int_2=np.array([bulk_Int_2[-1]+dt*(a*((top_Int_2[-1]+bot_Int_2[-1])/2-bulk_Int_2[-1])/al+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_1[-1]/4-bulk_Int_2[-1]/3))])
    bulk_Int_2 = np.hstack([bulk_Int_2, nbulk_Int_2])
    nbot_Int_2=np.array([bot_Int_2[-1]+dt*(a*((bulk_Int_2[-1]+bot_Int_2[-1])/2-bot_Int_2[-1])/al+FF*bot_OH_rad[-1]*k_gen*(bot_Int_1[-1]/4-bot_Int_2[-1]/3))])
    bot_Int_2 = np.hstack([top_Int_2, nbot_Int_2])
    
    ntop_Int_3=np.array([top_Int_3[-1]+dt*(a*((top_Int_3[-1]+bulk_Int_3[-1])/2-top_Int_3[-1])/al+FF*top_OH_rad[-1]*k_gen*(top_Int_2[-1]/3-top_Int_3[-1]/4))])
    top_Int_3 = np.hstack([top_Int_3, ntop_Int_3])
    nbulk_Int_3=np.array([bulk_Int_3[-1]+dt*(a*((top_Int_3[-1]+bot_Int_3[-1])/2-bulk_Int_3[-1])/al+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_2[-1]/3-bulk_Int_3[-1]/4))])
    bulk_Int_3 = np.hstack([bulk_Int_3, nbulk_Int_3])
    nbot_Int_3=np.array([bot_Int_3[-1]+dt*(a*((bulk_Int_3[-1]+bot_Int_3[-1])/2-bot_Int_3[-1])/al+FF*bot_OH_rad[-1]*k_gen*(bot_Int_2[-1]/3-bot_Int_3[-1]/4))])
    bot_Int_3 = np.hstack([top_Int_3, nbot_Int_3])
    
    ntop_Int_4=np.array([top_Int_4[-1]+dt*(a*((top_Int_4[-1]+bulk_Int_4[-1])/2-top_Int_4[-1])/am+FF*top_OH_rad[-1]*k_gen*(top_Int_3[-1]/4-top_Int_4[-1]/21))])
    top_Int_4 = np.hstack([top_Int_4, ntop_Int_4])
    nbulk_Int_4=np.array([bulk_Int_4[-1]+dt*(a*((top_Int_4[-1]+bot_Int_4[-1])/2-bulk_Int_4[-1])/am+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_3[-1]/4-bulk_Int_4[-1]/21))])
    bulk_Int_4 = np.hstack([bulk_Int_4, nbulk_Int_4])
    nbot_Int_4=np.array([bot_Int_4[-1]+dt*(a*((bulk_Int_4[-1]+bot_Int_4[-1])/2-bot_Int_4[-1])/am+FF*bot_OH_rad[-1]*k_gen*(bot_Int_3[-1]/4-bot_Int_4[-1]/21))])
    bot_Int_4 = np.hstack([top_Int_4, nbot_Int_4])
    
    ntop_Int_5=np.array([top_Int_5[-1]+dt*(a*((top_Int_5[-1]+bulk_Int_5[-1])/2-top_Int_5[-1])/am+FF*top_OH_rad[-1]*k_gen*(top_Int_3[-1]-top_Int_5[-1]/22))])
    top_Int_5 = np.hstack([top_Int_5, ntop_Int_5])
    nbulk_Int_5=np.array([bulk_Int_5[-1]+dt*(a*((top_Int_5[-1]+bot_Int_5[-1])/2-bulk_Int_5[-1])/am+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_3[-1]-bulk_Int_5[-1]/22))])
    bulk_Int_5 = np.hstack([bulk_Int_5, nbulk_Int_5])
    nbot_Int_5=np.array([bot_Int_4[-1]+dt*(a*((bulk_Int_5[-1]+bot_Int_5[-1])/2-bot_Int_5[-1])/am+FF*bot_OH_rad[-1]*k_gen*(bot_Int_3[-1]-bot_Int_5[-1]/22))])
    bot_Int_5 = np.hstack([top_Int_5, nbot_Int_5])
   
    ntop_N_OH_3=np.array([top_N_OH_3[-1]+dt*(an*((top_N_OH_3[-1]+bulk_N_OH_3[-1])/2-top_N_OH_3[-1])+FF*top_OH_rad[-1]*k_gen*(top_Int_3[-1]-top_N_OH_3[-1]/22))])
    top_N_OH_3 = np.hstack([top_N_OH_3, ntop_N_OH_3])
    nbulk_N_OH_3=np.array([bulk_N_OH_3[-1]+dt*(an*((top_N_OH_3[-1]+bot_N_OH_3[-1])/2-bulk_N_OH_3[-1])+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_3[-1]-bulk_N_OH_3[-1]/22))])
    bulk_N_OH_3 = np.hstack([bulk_N_OH_3, nbulk_N_OH_3])
    nbot_N_OH_3=np.array([bot_N_OH_3[-1]+dt*(an*((bulk_N_OH_3[-1]+bot_N_OH_3[-1])/2-bot_N_OH_3[-1])+FF*bot_OH_rad[-1]*k_gen*(bot_Int_3[-1]-bot_N_OH_3[-1]/22))])
    bot_N_OH_3 = np.hstack([top_N_OH_3, nbot_N_OH_3])
    
    ntop_NH2_rad=np.array([top_NH2_rad[-1]+dt*(an*((top_NH2_rad[-1]+bulk_NH2_rad[-1])/2-top_NH2_rad[-1])+FF*top_OH_rad[-1]*k_gen*(top_Int_2[-1]-top_NH2_rad[-1]/7))])
    top_NH2_rad = np.hstack([top_NH2_rad, ntop_NH2_rad])
    nbulk_NH2_rad=np.array([bulk_NH2_rad[-1]+dt*(an*((top_NH2_rad[-1]+bot_NH2_rad[-1])/2-bulk_NH2_rad[-1])+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_2[-1]-bulk_NH2_rad[-1]/7))])
    bulk_NH2_rad = np.hstack([bulk_NH2_rad, nbulk_NH2_rad])
    nbot_NH2_rad=np.array([bot_NH2_rad[-1]+dt*(an*((bulk_NH2_rad[-1]+bot_NH2_rad[-1])/2-bot_NH2_rad[-1])+FF*bot_OH_rad[-1]*k_gen*(bot_Int_2[-1]-bot_NH2_rad[-1]/7))])
    bot_NH2_rad = np.hstack([top_NH2_rad, nbot_NH2_rad])
    
    ntop_NO3_ion=np.array([top_NO3_ion[-1]+dt*(an*((top_NO3_ion[-1]+bulk_NO3_ion[-1])/2-top_NO3_ion[-1])+FF*top_OH_rad[-1]*k_gen*(top_NH2_rad[-1]/7+top_N_OH_3[-1]/2))])
    top_NO3_ion = np.hstack([top_NO3_ion, ntop_NO3_ion])
    nbulk_NO3_ion=np.array([bulk_NO3_ion[-1]+dt*(an*((top_NO3_ion[-1]+bot_NO3_ion[-1])/2-bulk_NO3_ion[-1])+FF*bulk_OH_rad[-1]*k_gen*(bulk_NH2_rad[-1]/7+bulk_N_OH_3[-1]/2))])
    bulk_NO3_ion = np.hstack([bulk_NO3_ion, nbulk_NO3_ion])
    nbot_NO3_ion=np.array([bot_NO3_ion[-1]+dt*(an*((bulk_NO3_ion[-1]+bot_NO3_ion[-1])/2-bot_NO3_ion[-1])+FF*bot_OH_rad[-1]*k_gen*(bot_NH2_rad[-1]/7+bot_N_OH_3[-1]/2))])
    bot_NO3_ion = np.hstack([top_NO3_ion, nbot_NO3_ion])
    
    ntop_SO4_ion=np.array([top_SO4_ion[-1]+dt*(aS*((top_SO4_ion[-1]+bulk_SO4_ion[-1])/2-top_SO4_ion[-1])+FF*top_OH_rad[-1]*k_gen*(top_Int_3[-1]/3))])
    top_SO4_ion = np.hstack([top_SO4_ion, ntop_SO4_ion])
    nbulk_SO4_ion=np.array([bulk_SO4_ion[-1]+dt*(aS*((top_SO4_ion[-1]+bot_SO4_ion[-1])/2-bulk_SO4_ion[-1])+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_3[-1]/3))])
    bulk_SO4_ion = np.hstack([bulk_SO4_ion, nbulk_SO4_ion])
    nbot_SO4_ion=np.array([bot_SO4_ion[-1]+dt*(aS*((bulk_SO4_ion[-1]+bot_SO4_ion[-1])/2-bot_SO4_ion[-1])+FF*bot_OH_rad[-1]*k_gen*(bot_Int_3[-1]/3))])
    bot_SO4_ion = np.hstack([top_SO4_ion, nbot_SO4_ion])
    
    ntop_CH3OH=np.array([top_CH3OH[-1]+dt*(ach3oh*((top_CH3OH[-1]+bulk_CH3OH[-1])/2-top_CH3OH[-1])/am+FF*top_OH_rad[-1]*k_gen*(top_Int_1[-1]-top_CH3OH[-1]/4))])
    top_CH3OH = np.hstack([top_CH3OH, ntop_CH3OH])
    nbulk_CH3OH=np.array([bulk_CH3OH[-1]+dt*(ach3oh*((top_CH3OH[-1]+bot_CH3OH[-1])/2-bulk_CH3OH[-1])/am+FF*bulk_OH_rad[-1]*k_gen*(bulk_Int_1[-1]-bulk_CH3OH[-1]/4))])
    bulk_CH3OH = np.hstack([bulk_CH3OH, nbulk_CH3OH])
    nbot_CH3OH=np.array([bot_CH3OH[-1]+dt*(ach3oh*((bulk_CH3OH[-1]+bot_CH3OH[-1])/2-bot_CH3OH[-1])/am+FF*bot_OH_rad[-1]*k_gen*(bot_Int_1[-1]-bot_CH3OH[-1]/4))])
    bot_CH3OH = np.hstack([top_CH3OH, nbot_CH3OH])
    
    ntop_CO2=np.array([top_CO2[-1]+dt*(a*((top_CO2[-1]+bulk_CO2[-1])/2-top_CO2[-1])/aco2+FF*top_OH_rad[-1]*k_gen*(top_CH3OH[-1]/4+6*top_Int_4[-1]/21+6*top_Int_5[-1]/22))])
    top_CO2 = np.hstack([top_CO2, ntop_CO2])
    nbulk_CO2=np.array([bulk_CO2[-1]+dt*(a*((top_CO2[-1]+bot_CO2[-1])/2-bulk_CO2[-1])/aco2+FF*bulk_OH_rad[-1]*k_gen*(bulk_CH3OH[-1]/4+6*bulk_Int_4[-1]/21+6*bulk_Int_5[-1]/22))])
    bulk_CO2 = np.hstack([bulk_CO2, nbulk_CO2])
    nbot_CO2=np.array([bot_CO2[-1]+dt*(a*((bulk_CO2[-1]+bot_CO2[-1])/2-bot_CO2[-1])/aco2+FF*bot_OH_rad[-1]*k_gen*(bot_CH3OH[-1]/4+6*bot_Int_4[-1]/21+6*bot_Int_5[-1]/22))])
    bot_CO2 = np.hstack([top_CO2, nbot_CO2])
    
#    ntop_TOC=np.array([top_TOC[-1]-top_CO2[-1]])
#    top_TOC = np.hstack([top_TOC, ntop_TOC])
#    nbulk_TOC=np.array([bulk_TOC[-1]-bulk_CO2[-1]])
#    bulk_TOC = np.hstack([bulk_TOC, nbulk_TOC])
#    nbot_TOC=np.array([top_TOC[-1]-top_CO2[-1]])
#    bot_TOC = np.hstack([bot_TOC, nbot_TOC])
    
#    ntop_TOC=np.array([16*MB_top[-1]+12*(top_Int_1[-1]+top_Int_2[-1]+top_Int_3[-1])+6*(top_Int_4[-1]+top_Int_5[-1])+top_CH3OH[-1]])
#    top_TOC = np.hstack([top_TOC, ntop_TOC])
#    nbulk_TOC=np.array([16*MB_bulk[-1]+12*(bulk_Int_1[-1]+bulk_Int_2[-1]+bulk_Int_3[-1])+6*(bulk_Int_4[-1]+bulk_Int_5[-1])+bulk_CH3OH[-1]])
#    bulk_TOC = np.hstack([bulk_TOC, nbulk_TOC])
#    nbot_TOC=np.array([[16*MB_bulk[-1]+12*(bulk_Int_1[-1]+bot_Int_2[-1]+bot_Int_3[-1])+6*(bot_Int_4[-1]+bot_Int_5[-1])+bot_CH3OH[-1]]])
#    bot_TOC = np.hstack([bot_TOC, nbot_TOC])
#    
#    ntop_H_ion=np.array([top_H_ion[-1]+dt*(a*((top_H_ion[-1]+bulk_H_ion[-1])/2-top_H_ion[-1])+FF*top_OH_rad[-1]*k_gen*(top_NH2_rad[-1]/7+top_N_OH_3[-1]/2+2*top_Int_3[1]/3))])
#    top_H_ion = np.hstack([top_H_ion, ntop_H_ion])
#    nbulk_H_ion=np.array([bulk_H_ion[-1]+dt*(a*((top_H_ion[-1]+bot_H_ion[-1])/2-bulk_H_ion[-1])+FF*bulk_OH_rad[-1]*k_gen*(bulk_NH2_rad[-1]/7+bulk_N_OH_3[-1]/2+2*bulk_Int_3[1]/3))])
#    bulk_H_ion = np.hstack([bulk_H_ion, nbulk_H_ion])
#    nbot_H_ion=np.array([bot_H_ion[-1]+dt*(a*((bulk_H_ion[-1]+bot_H_ion[-1])/2-bot_H_ion[-1])+FF*bot_OH_rad[-1]*k_gen*(bulk_NH2_rad[-1]/7+bulk_N_OH_3[-1]/2+2*bulk_Int_3[1]/3))])
#    bot_H_ion = np.hstack([top_H_ion, nbot_H_ion])
    
    ntop_H2O2=np.array([top_H2O2[-1]+dt*(a*((top_H2O2[-1]+bulk_H2O2[-1])/2-top_H2O2[-1])+FF2*(k_e*top_OH_rad[-1]**2+k_g*top_HO2_rad[-1]**2-top_H2O2[-1]*(k_a*top_OH_rad[-1]+k_h*top_O2_rad[-1]-k_f*top_HO2_rad[-1])))])
    top_H2O2 = np.hstack([top_H2O2, ntop_H2O2])
    nbulk_H2O2=np.array([bulk_H2O2[-1]+dt*(a*((top_H2O2[-1]+bot_H2O2[-1])/2-bulk_H2O2[-1])+FF2*(k_e*bulk_OH_rad[-1]**2+k_g*bulk_HO2_rad[-1]**2-bulk_H2O2[-1]*(k_a*bulk_OH_rad[-1]+k_h*bulk_O2_rad[-1]-k_f*bulk_HO2_rad[-1])))])
    bulk_H2O2 = np.hstack([bulk_H2O2, nbulk_H2O2])
    nbot_H2O2=np.array([bot_H2O2[-1]+dt*(a*((bulk_H2O2[-1]+bot_H2O2[-1])/2-bot_H2O2[-1])+FF2*(k_e*bot_OH_rad[-1]**2+k_g*bot_HO2_rad[-1]**2-bot_H2O2[-1]*(k_a*bot_OH_rad[-1]+k_h*bot_O2_rad[-1]-k_f*bot_HO2_rad[-1])))])
    bot_H2O2 = np.hstack([top_H2O2, nbot_H2O2])
#    
#    ntop_HO2_rad=np.array([top_HO2_rad[-1]+dt*(a*((top_HO2_rad[-1]+bulk_HO2_rad[-1])/2-top_HO2_rad[-1])+FF2*(k_a*top_OH_rad[-1]*top_H2O2[-1]-top_HO2_rad[-1]*(k_f*top_H2O2[-1]+k_g*top_HO2_rad[-1]-k_i*top_O2_rad[-1])))])
#    top_HO2_rad = np.hstack([top_HO2_rad, ntop_HO2_rad])
#    nbulk_HO2_rad=np.array([bulk_HO2_rad[-1]+dt*(a*((top_HO2_rad[-1]+bot_HO2_rad[-1])/2-bulk_HO2_rad[-1])+FF2*(k_a*bulk_OH_rad[-1]*bulk_H2O2[-1]-bulk_HO2_rad[-1]*(k_f*bulk_H2O2[-1]+k_g*bulk_HO2_rad[-1]-k_i*bulk_O2_rad[-1])))])
#    bulk_HO2_rad = np.hstack([bulk_HO2_rad, nbulk_HO2_rad])
#    nbot_HO2_rad=np.array([bot_HO2_rad[-1]+dt*(a*((bulk_HO2_rad[-1]+bot_HO2_rad[-1])/2-bot_HO2_rad[-1])+FF2*(k_a*bot_OH_rad[-1]*bot_H2O2[-1]-bot_HO2_rad[-1]*(k_f*bot_H2O2[-1]+k_g*bot_HO2_rad[-1]-k_i*bot_O2_rad[-1])))])
#    bot_HO2_rad = np.hstack([top_HO2_rad, nbot_HO2_rad])
    
    ntop_HO2_ion=np.array([top_HO2_ion[-1]+dt*(a*((top_HO2_ion[-1]+bulk_HO2_ion[-1])/2-top_HO2_ion[-1])+FF2*(k_i*top_HO2_rad[-1]*top_O2_rad[-1]-k_b*top_HO2_ion[-1]*top_OH_rad[-1]))])
    top_HO2_ion = np.hstack([top_HO2_ion, ntop_HO2_ion])
    nbulk_HO2_ion=np.array([bulk_HO2_ion[-1]+dt*(a*((top_HO2_ion[-1]+bot_HO2_ion[-1])/2-bulk_HO2_ion[-1])+FF2*(k_i*bulk_HO2_rad[-1]*bulk_O2_rad[-1]-k_b*bulk_HO2_ion[-1]*bulk_OH_rad[-1]))])
    bulk_HO2_ion = np.hstack([bulk_HO2_ion, nbulk_HO2_ion])
    nbot_HO2_ion=np.array([bot_HO2_ion[-1]+dt*(a*((bulk_HO2_ion[-1]+bot_HO2_ion[-1])/2-bot_HO2_ion[-1])+FF2*(k_i*bot_HO2_rad[-1]*bot_O2_rad[-1]-k_b*bot_HO2_ion[-1]*bot_OH_rad[-1]))])
    bot_HO2_ion = np.hstack([top_HO2_ion, nbot_HO2_ion])
    
    ntop_O2_rad=np.array([top_O2_rad[-1]+dt*(a*((top_O2_rad[-1]+bulk_O2_rad[-1])/2-top_O2_rad[-1])+FF2*(k_b*top_O2_rad[-1]*top_HO2_ion[0]-top_O2_rad[-1]*(k_c*top_OH_rad[-1]+k_h*top_H2O2[-1]+k_i*top_HO2_rad[-1])))])
    top_O2_rad = np.hstack([top_O2_rad, ntop_O2_rad])
    nbulk_O2_rad=np.array([bulk_O2_rad[-1]+dt*(a*((top_O2_rad[-1]+bot_O2_rad[-1])/2-bulk_O2_rad[-1])+FF2*(k_b*bulk_O2_rad[-1]*bulk_HO2_ion[0]-bulk_O2_rad[-1]*(k_c*bulk_OH_rad[-1]+k_h*bulk_H2O2[-1]+k_i*bulk_HO2_rad[-1])))])
    bulk_O2_rad = np.hstack([bulk_O2_rad, nbulk_O2_rad])
    nbot_O2_rad=np.array([bot_O2_rad[-1]+dt*(a*((bulk_O2_rad[-1]+bot_O2_rad[-1])/2-bot_O2_rad[-1])+FF2*(k_b*bot_O2_rad[-1]*bot_HO2_ion[0]+k_red*oc_trap*O2_bot[-1]-bot_O2_rad[-1]*(k_c*bot_OH_rad[-1]+k_h*bot_H2O2[-1]+k_i*bot_HO2_rad[-1])))])
    bot_O2_rad = np.hstack([top_O2_rad, nbot_O2_rad])
    
    ntop_OH_ion=np.array([top_OH_ion[-1]+dt*(a*((top_OH_ion[-1]+bulk_OH_ion[-1])/2-top_OH_ion[-1])+FF2*(k_c*top_OH_rad[-1]*top_O2_rad[-1]+k_h*top_O2_rad[-1]*top_H2O2[-1]))])
    top_OH_ion = np.hstack([top_OH_ion, ntop_OH_ion])
    nbulk_OH_ion=np.array([bulk_OH_ion[-1]+dt*(a*((top_OH_ion[-1]+bot_OH_ion[-1])/2-bulk_OH_ion[-1])+FF2*(k_c*bulk_OH_rad[-1]*bulk_O2_rad[-1]+k_h*bulk_O2_rad[-1]*bulk_H2O2[-1]))])
    bulk_OH_ion = np.hstack([bulk_OH_ion, nbulk_OH_ion])
    nbot_OH_ion=np.array([bot_OH_ion[-1]+dt*(a*((bulk_OH_ion[-1]+bot_OH_ion[-1])/2-bot_OH_ion[-1])+FF2*(k_c*bot_OH_rad[-1]*bot_O2_rad[-1]+k_h*bot_O2_rad[-1]*bot_H2O2[-1]))])
    bot_OH_ion = np.hstack([top_OH_ion, nbot_OH_ion])
    
    nOH_rad_s_rate=np.array([v_p*Lambda_p*k_1*OH_ion_s[-1]/((v_p+h*(k_1*OH_ion_s[-1]+(k_rec_2*k_T_trap*n_s)/(k_rec_2*p_s+k_red*O2_bot[-1]*(6.02e23))))*(6.02e23))])
    OH_rad_s_rate = np.hstack([OH_rad_s_rate, nOH_rad_s_rate])

MB_top=MB_bulk*1e9
MB_bulk=MB_bulk*1e9   
MB_bot=MB_bulk*1e9
O2_top=MB_bulk*1e9
O2_bulk=MB_bulk*1e9   
O2_bot=MB_bulk*1e9 
data=open("data_first_run.txt", "r")
data_vals=data.readlines()
time=open("time_first_run.txt", "r")
time_vals=time.readlines()
data.close()
time.close()
T=np.zeros(len(time_vals))
for i in range(0,len(time_vals)):
    T[i]=time_vals[i]   
I=np.zeros(len(data_vals))
for k in range(0,len(data_vals)):
    I[k]=data_vals[k]
I=1/((3.3/1024)*I)
I=I-min(I)*np.ones(len(I))
I=I/max(I)
MB=np.zeros(len(MB_bulk))
for k in range(0,len(MB_bulk)):
    MB[k]=MB_bulk[k]
MB = MB*(1/max(MB))   
#plt.xlabel('Time [s]')
#plt.ylabel('Bulk MB [mu mol dm^-3]')
#plt.plot( t, MB_bulk,linewidth = 1, label= "Bulk MB Concentration") # Plots the concentration of free of intermediat 1 and CO2 at the photocatalyst surface vs time
#plt.show()
#plt.xlabel('Time [s]')
#plt.ylabel('Surface O2 [mu mol dm^-3]')
#plt.plot( t, O2_bot,linewidth = 1, label= "Surface O2 Concentration") # Plots the concentration of free of intermediat 1 and CO2 at the photocatalyst surface vs time
#plt.show()
#plt.xlabel('Time [s]')
#plt.ylabel('Bulk O2 [mu mol dm^-3]')
#plt.plot( t, O2_bulk,linewidth = 1, label= "Bulk O2 Concentration") # Plots the concentration of free of intermediat 1 and CO2 at the photocatalyst surface vs time
#plt.show()
#plt.xlabel('Time [s]')
#plt.ylabel('Top O2 [mu mol dm^-3]')
#plt.plot( t, O2_top,linewidth = 1, label= "Top O2 Concentration") # Plots the concentration of free of intermediat 1 and CO2 at the photocatalyst surface vs time
#plt.show()
##print(O2_top[0]-O2_top[-1])
#plt.xlabel('Time [s]')
#plt.ylabel('Surface Hydroxyl Radical Production Rate [mu mol dm^-3 s^-1]')
#plt.plot( t, OH_rad_s_rate,linewidth = 1, label= "Surface Hydroxyl Radical Production Rate") # Plots the concentration of free of intermediat 1 and CO2 at the photocatalyst surface vs time
#plt.show()

plt.xlabel('Time [s]')
plt.ylabel('Bulk Concentration of TOC and CO2 [ mol cm^-3]')
plt.plot( t, bulk_CO2, linewidth = 1, label= "Surface Hydroxyl Radical Production Rate") # Plots the concentration of free of intermediat 1 and CO2 at the photocatalyst surface vs time
plt.show()

#np.savetxt("Time Variable MB.txt", t)
#np.savetxt("Top Oxygen Concentration Vairable MB (1 mol cm^-3).txt", O2_top)
#np.savetxt("Bulk MB Concentration (1 mol cm^-3).txt", MB_bulk)

#np.savetxt("Time 1.txt", t)
#np.savetxt("Bulk MB Concentration 1.txt", MB_bulk)
#np.savetxt("Surface Oxygen Concentration 1.txt", O2_bot)
#np.savetxt("Bulk Oxygen Concentration 1.txt", O2_bulk)
#np.savetxt("Bulk H ion Concentration 1.txt", bulk_H_ion)
#np.savetxt("Bulk OH ion Concentration 1.txt", bulk_OH_ion)
#np.savetxt("Bulk NO3 ion Concentration 11.txt", bulk_NO3_ion)
#np.savetxt("Bulk SO4 ion Concentration 1.txt", bulk_SO4_ion)
#np.savetxt("Bulk Int_1 Concentration 1.txt", bulk_Int_1)
#np.savetxt("Bulk Int_2 Concentration 1.txt", bulk_Int_2)
#np.savetxt("Bulk Int_3 Concentration 1.txt", bulk_Int_3)
#np.savetxt("Bulk Int_4 Concentration 1.txt", bulk_Int_4)
#np.savetxt("Bulk Int_5 Concentration 1.txt", bulk_Int_5)
#np.savetxt("Bulk N_OH_3 Concentration 1.txt", bulk_N_OH_3)
#np.savetxt("Bulk CH3OH Concentration 1.txt", bulk_CH3OH)
#np.savetxt("Surface Hydroxyl Radical Production Rate 1.txt", OH_rad_s_rate)
#np.savetxt("Bulk TOC Concentration 1.txt", bulk_TOC)    

#np.savetxt("Time Constant Oxygen Concentration 1.txt", t)
#np.savetxt("Bulk MB Concentration Constant Oxygen Concentration 1.txt", MB_bulk)
#np.savetxt("Surface Oxygen Concentration Constant Oxygen Concentration 1.txt", O2_bot)
#np.savetxt("Bulk Oxygen Concentration Constant Oxygen Concentration 1.txt", O2_bulk)
#np.savetxt("Bulk H ion Concentration Constant Oxygen Concentration 1.txt", bulk_H_ion)
#np.savetxt("Bulk OH ion Concentration Constant Oxygen Concentration 1.txt", bulk_OH_ion)
#np.savetxt("Bulk NO3 ion Concentration Constant Oxygen Concentration 1.txt", bulk_NO3_ion)
#np.savetxt("Bulk SO4 ion Concentration Constant Oxygen Concentration 1.txt", bulk_SO4_ion)
#np.savetxt("Bulk Int_1 Concentration Constant Oxygen Concentration 1.txt", bulk_Int_1)
#np.savetxt("Bulk Int_2 Concentration Constant Oxygen Concentration 1.txt", bulk_Int_2)
#np.savetxt("Bulk Int_3 Concentration Constant Oxygen Concentration 1.txt", bulk_Int_3)
#np.savetxt("Bulk Int_4 Concentration Constant Oxygen Concentration 1.txt", bulk_Int_4)
#np.savetxt("Bulk Int_5 Concentration Constant Oxygen Concentration 1.txt", bulk_Int_5)
#np.savetxt("Bulk N_OH_3 Concentration Constant Oxygen Concentration 1.txt", bulk_N_OH_3)
#np.savetxt("Bulk CH3OH Concentration Constant Oxygen Concentration 1.txt", bulk_CH3OH)
#np.savetxt("Surface Hydroxyl Radical Production Rate Constant Oxygen Concentration 1.txt", OH_rad_s_rate)
#np.savetxt("Bulk TOC Concentration Constant Oxygen Concentration 3.txt", bulk_TOC)

#O2max=2.56e-7
#inc=1e-8
#O2_bot=np.arange(0,O2max,inc)
#OH_rad_rate=v_p*Lambda_p*k_1*OH_ion_s[-1]/((v_p+h*(k_1*OH_ion_s[-1]+(k_rec_2*k_T_trap*n_s)/(k_rec_2*p_s+k_red*O2_bot*(6.02e23))))*(6.02e23))
##fig, ax = plt.subplots()
##ax.plot(O2_bot, OH_rad_rate)
##ax.ticklabel_format(useOffset=False, style='plain')
#plt.xlabel('O2 concentration [mu mol dm^-3]')
#plt.ylabel('Top O2 [mu mol dm^-3 s^-1]')
#plt.plot(O2_bot, OH_rad_rate,linewidth = 1, label= "Bulk MB Concentration")
#plt.show()