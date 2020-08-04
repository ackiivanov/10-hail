import numpy as np
import pandas as pd
import scipy.signal as sgn
import matplotlib.pyplot as plt
from matplotlib import rc

c = 343 #m/s
deltac = 2 #m/s


#Rectangular geometry open
Lx = 0.16 #m
Ly = 0.08 #m
Lz = 0.07 #m
deltaLx = 0.001 #m
deltaLy = 0.001 #m
deltaLz = 0.001 #m
MAX = 2300 #Hz

#Theoretical prediction
thnu = []
thdeltanu = []
for i in range(10):
	for j in range(10):
		for k in range(10):
			fr = c/2 * ((i/Lx)**2 + (j/Ly)**2 + ((k + 1/2)/Lz)**2)**(1/2)
			deltafr = fr/c * deltac + c**2/(4*fr)*(((j**2)/(Lx**3))*deltaLx + ((j**2)/(Ly**3))*deltaLy + (((k + 1/2)**2)/(Lz**3))*deltaLz)
			if fr < MAX:
				thnu.append(fr)
				thdeltanu.append(deltafr)

#Import data for rectangular box
rectE1 = pd.read_csv('rectangular_empty1.txt', sep='\t')
rectE2 = pd.read_csv('rectangular_empty2.txt', sep='\t')
rectE3 = pd.read_csv('rectangular_empty3.txt', sep='\t')
rectE4 = pd.read_csv('rectangular_empty4.txt', sep='\t')
rectE5 = pd.read_csv('rectangular_empty5.txt', sep='\t')
rectE6 = pd.read_csv('rectangular_empty6.txt', sep='\t')

rectB1 = pd.read_csv('rectangular_blocked1.txt', sep='\t')
rectB2 = pd.read_csv('rectangular_blocked2.txt', sep='\t')
rectB3 = pd.read_csv('rectangular_blocked3.txt', sep='\t')

rectEnu1 = rectE1['Frequency (Hz)']
rectEnu2 = rectE2['Frequency (Hz)']
rectEnu3 = rectE3['Frequency (Hz)']
rectEnu4 = rectE4['Frequency (Hz)']
rectEnu5 = rectE5['Frequency (Hz)']
rectEnu6 = rectE6['Frequency (Hz)']

rectBnu1 = rectB1['Frequency (Hz)']
rectBnu2 = rectB2['Frequency (Hz)']
rectBnu3 = rectB3['Frequency (Hz)']

rectEint1 = 10**(rectE1['Level (dB)']/10)/max(10**(rectE1['Level (dB)']/10))
rectEint2 = 10**(rectE2['Level (dB)']/10)/max(10**(rectE2['Level (dB)']/10))
rectEint3 = 10**(rectE3['Level (dB)']/10)/max(10**(rectE3['Level (dB)']/10))
rectEint4 = 10**(rectE4['Level (dB)']/10)/max(10**(rectE4['Level (dB)']/10))
rectEint5 = 10**(rectE5['Level (dB)']/10)/max(10**(rectE5['Level (dB)']/10))
rectEint6 = 10**(rectE6['Level (dB)']/10)/max(10**(rectE6['Level (dB)']/10))

rectBint1 = 10**(rectB1['Level (dB)']/10)/max(10**(rectB1['Level (dB)']/10))
rectBint2 = 10**(rectB2['Level (dB)']/10)/max(10**(rectB2['Level (dB)']/10))
rectBint3 = 10**(rectB3['Level (dB)']/10)/max(10**(rectB3['Level (dB)']/10))

rectEnu = rectE1['Frequency (Hz)']
rectEint = (rectEint1 + rectEint2 + rectEint3 + rectEint4 + rectEint5 + rectEint6)/max(rectEint1 + rectEint2 + rectEint3 + rectEint4 + rectEint5 + rectEint6)

rectBnu = rectB1['Frequency (Hz)']
rectBint = (rectBint1 + rectBint2 + rectBint3)/max(rectBint1 + rectBint2 + rectBint3)

#Finding peaks
rectEp1 = sgn.find_peaks(rectEint1, prominence=0.05)
rectEp2 = sgn.find_peaks(rectEint2, prominence=0.05)
rectEp3 = sgn.find_peaks(rectEint3, prominence=0.05)
rectEp4 = sgn.find_peaks(rectEint4, prominence=0.05)
rectEp5 = sgn.find_peaks(rectEint5, prominence=0.05)
rectEp6 = sgn.find_peaks(rectEint6, prominence=0.05)

rectBp1 = sgn.find_peaks(rectBint1, prominence=0.05)
rectBp2 = sgn.find_peaks(rectBint2, prominence=0.05)
rectBp3 = sgn.find_peaks(rectBint3, prominence=0.05)

rectEps = list(rectEp1[0]) + list(rectEp2[0]) + list(rectEp3[0]) + list(rectEp4[0]) + list(rectEp5[0]) + list(rectEp6[0])
rectBps = list(rectBp1[0]) + list(rectBp2[0]) + list(rectBp3[0])

#Plotting for rectangular box
figure, axes = plt.subplots(1)
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

plt.xlim(0, MAX)
plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"Intensity [arb. unit]")
plt.title("Rectangular geometry (Lx=0.16m, Ly=0.08m, Lz=0.07m)")
plt.grid()

plt.plot(rectEnu, rectEint, label='Volume free')
plt.plot(rectBnu, rectBint, label='Volume blocked')

#for i in range(len(thnu)):
#	axes.axvspan(thnu[i] - thdeltanu[i], thnu[i] + thdeltanu[i], alpha=0.25, color='r')
#	plt.axvline(thnu[i], ls='--', c='r', lw=0.7)
#axes.axvspan(thnu[0] - thdeltanu[0], thnu[0] + thdeltanu[0], alpha=0.25, color='r', label='Confidence interval')
#plt.axvline(thnu[0], ls='--', c='r', lw=0.7, label='Prediction')

#for i in rectEps:
#	plt.axvline(rectEnu[i], ls='--', c='b', lw=0.7)
#plt.axvline(rectEnu[0], ls='--', c='b', lw=0.7, label='Volume free')

#for i in rectBps:
#	plt.axvline(rectBnu[i], ls='--', c='y', lw=0.7)
#plt.axvline(rectBnu[0], ls='--', c='y', lw=0.7, label='Volume blocked')

plt.legend(loc='upper right')
plt.savefig("rectangular.png", dpi=500)
plt.show()



#Cylindrical geometry
H = 0.11 + 0.6*0.0603 #m
R = 0.0603 #m
deltaH = 0.002 #m
deltaR = 0.001 #m
MAX = 7000 #Hz

#Theoretical prediction
thnu = []
thdeltanu = []
for i in {0, 2.4048, 3.8317, 5.1356, 5.5201, 6.3802, 7.0156, 7.5883}:
	for j in range(10):
		fr = c/2 * ((i/(np.pi*R))**2 + ((j + 1/2)/H)**2)**(1/2)
		deltafr = fr/c * deltac + c**2/(4*fr)*(((i**2)/(np.pi**2 * R**3))*deltaR + (((j + 1/2)**2)/(H**3))*deltaH)
		if fr < MAX:
			thnu.append(fr)
			thdeltanu.append(deltafr)

#Import data for cylindrical box
cylE1 = pd.read_csv('cylinderpot_empty1.txt', sep='\t')
cylE2 = pd.read_csv('cylinderpot_empty2.txt', sep='\t')
cylE3 = pd.read_csv('cylinderpot_empty3.txt', sep='\t')
cylE4 = pd.read_csv('cylinderpot_empty4.txt', sep='\t')
cylE5 = pd.read_csv('cylinderpot_empty5.txt', sep='\t')
cylE6 = pd.read_csv('cylinderpot_empty6.txt', sep='\t')
cylE7 = pd.read_csv('cylinderpot_empty7.txt', sep='\t')

cylB1 = pd.read_csv('cylinderpot_blocked1.txt', sep='\t')
cylB2 = pd.read_csv('cylinderpot_blocked2.txt', sep='\t')
cylB3 = pd.read_csv('cylinderpot_blocked3.txt', sep='\t')

cylEnu1 = cylE1['Frequency (Hz)']
cylEnu2 = cylE2['Frequency (Hz)']
cylEnu3 = cylE3['Frequency (Hz)']
cylEnu4 = cylE4['Frequency (Hz)']
cylEnu5 = cylE5['Frequency (Hz)']
cylEnu6 = cylE6['Frequency (Hz)']
cylEnu7 = cylE7['Frequency (Hz)']

cylBnu1 = cylB1['Frequency (Hz)']
cylBnu2 = cylB2['Frequency (Hz)']
cylBnu3 = cylB3['Frequency (Hz)']

cylEint1 = 10**(cylE1['Level (dB)']/10)/max(10**(cylE1['Level (dB)']/10))
cylEint2 = 10**(cylE2['Level (dB)']/10)/max(10**(cylE2['Level (dB)']/10))
cylEint3 = 10**(cylE3['Level (dB)']/10)/max(10**(cylE3['Level (dB)']/10))
cylEint4 = 10**(cylE4['Level (dB)']/10)/max(10**(cylE4['Level (dB)']/10))
cylEint5 = 10**(cylE5['Level (dB)']/10)/max(10**(cylE5['Level (dB)']/10))
cylEint6 = 10**(cylE6['Level (dB)']/10)/max(10**(cylE6['Level (dB)']/10))
cylEint7 = 10**(cylE7['Level (dB)']/10)/max(10**(cylE7['Level (dB)']/10))

cylBint1 = 10**(cylB1['Level (dB)']/10)/max(10**(cylB1['Level (dB)']/10))
cylBint2 = 10**(cylB2['Level (dB)']/10)/max(10**(cylB2['Level (dB)']/10))
cylBint3 = 10**(cylB3['Level (dB)']/10)/max(10**(cylB3['Level (dB)']/10))

cylEnu = cylE1['Frequency (Hz)']
cylEint = (cylEint1 + cylEint2 + cylEint3 + cylEint4 + cylEint5 + cylEint6 + cylEint7)/(max(cylEint1 + cylEint2 + cylEint3 + cylEint4 + cylEint5 + cylEint6 + cylEint7))

cylBnu = cylB1['Frequency (Hz)']
cylBint = (cylBint1 + cylBint2 + cylBint3)/(max(cylBint1 + cylBint2 + cylBint3))

#Finding peaks
cylEp1 = sgn.find_peaks(cylEint1, prominence=0.05)
cylEp2 = sgn.find_peaks(cylEint2, prominence=0.05)
cylEp3 = sgn.find_peaks(cylEint3, prominence=0.05)
cylEp4 = sgn.find_peaks(cylEint4, prominence=0.05)
cylEp5 = sgn.find_peaks(cylEint5, prominence=0.05)
cylEp6 = sgn.find_peaks(cylEint6, prominence=0.05)
cylEp7 = sgn.find_peaks(cylEint7, prominence=0.05)

cylBp1 = sgn.find_peaks(cylBint1, prominence=0.05)
cylBp2 = sgn.find_peaks(cylBint2, prominence=0.05)
cylBp3 = sgn.find_peaks(cylBint3, prominence=0.05)

cylEps = list(cylEp1[0]) + list(cylEp2[0]) + list(cylEp3[0]) + list(cylEp4[0]) + list(cylEp5[0]) + list(cylEp6[0]) + list(cylEp7[0])
cylBps = list(cylBp1[0]) + list(cylBp2[0]) + list(cylBp3[0])

#Plotting for cylindrical pot
figure, axes = plt.subplots(1)
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

plt.xlim(0, MAX)
plt.ylim(0, 1.01)
plt.xlabel(r"Frequency $\nu$ [Hz]")
axes.secondary_xaxis('top')
plt.ylabel(r"Intensity [arb. unit]")
#plt.title("Cylindrical geometry (H = 0.11m, R = 0.06m)")
plt.grid()

#plt.plot(cylEnu, cylEint, label='Volume free')
#plt.plot(cylBnu, cylBint, label='Volume blocked')

for i in range(len(thnu)):
	axes.axvspan(thnu[i] - thdeltanu[i], thnu[i] + thdeltanu[i], alpha=0.25, color='r')
	plt.axvline(thnu[i], ls='--', c='r', lw=0.7)
axes.axvspan(thnu[0] - thdeltanu[0], thnu[0] + thdeltanu[0], alpha=0.25, color='r', label='Confidence interval')
plt.axvline(thnu[0], ls='--', c='r', lw=0.7, label='Prediction')

for i in cylEps:
	plt.axvline(cylEnu[i], ls='--', c='b', lw=0.7)
plt.axvline(cylEnu[0], ls='--', c='b', lw=0.7, label='Volume free')

for i in cylBps:
	plt.axvline(cylBnu[i], ls='--', c='y', lw=0.7)
plt.axvline(cylBnu[0], ls='--', c='y', lw=0.7, label='Volume blocked')

#plt.legend(loc='upper right')
plt.savefig("cylindrical.png", dpi=500)
plt.show()



#Mystery box 2
MAX = 1400 #Hz

#Import data
myst2E1 = pd.read_csv('mystery2_empty1.txt', sep='\t')
myst2E2 = pd.read_csv('mystery2_empty2.txt', sep='\t')
myst2E3 = pd.read_csv('mystery2_empty3.txt', sep='\t')
myst2E4 = pd.read_csv('mystery2_empty4.txt', sep='\t')
myst2E5 = pd.read_csv('mystery2_empty5.txt', sep='\t')
myst2E6 = pd.read_csv('mystery2_empty6.txt', sep='\t')

myst2B1 = pd.read_csv('mystery2_blocked1.txt', sep='\t')
myst2B2 = pd.read_csv('mystery2_blocked2.txt', sep='\t')
myst2B3 = pd.read_csv('mystery2_blocked3.txt', sep='\t')
myst2B4 = pd.read_csv('mystery2_blocked4.txt', sep='\t')


myst2Enu1 = myst2E1['Frequency (Hz)']
myst2Enu2 = myst2E2['Frequency (Hz)']
myst2Enu3 = myst2E3['Frequency (Hz)']
myst2Enu4 = myst2E4['Frequency (Hz)']
myst2Enu5 = myst2E5['Frequency (Hz)']
myst2Enu6 = myst2E6['Frequency (Hz)']

myst2Bnu1 = myst2B1['Frequency (Hz)']
myst2Bnu2 = myst2B2['Frequency (Hz)']
myst2Bnu3 = myst2B3['Frequency (Hz)']
myst2Bnu4 = myst2B4['Frequency (Hz)']

myst2Eint1 = 10**(myst2E1['Level (dB)']/10)/max(10**(myst2E1['Level (dB)']/10))
myst2Eint2 = 10**(myst2E2['Level (dB)']/10)/max(10**(myst2E2['Level (dB)']/10))
myst2Eint3 = 10**(myst2E3['Level (dB)']/10)/max(10**(myst2E3['Level (dB)']/10))
myst2Eint4 = 10**(myst2E4['Level (dB)']/10)/max(10**(myst2E4['Level (dB)']/10))
myst2Eint5 = 10**(myst2E5['Level (dB)']/10)/max(10**(myst2E5['Level (dB)']/10))
myst2Eint6 = 10**(myst2E6['Level (dB)']/10)/max(10**(myst2E6['Level (dB)']/10))

myst2Bint1 = 10**(myst2B1['Level (dB)']/10)/max(10**(myst2B1['Level (dB)']/10))
myst2Bint2 = 10**(myst2B2['Level (dB)']/10)/max(10**(myst2B2['Level (dB)']/10))
myst2Bint3 = 10**(myst2B3['Level (dB)']/10)/max(10**(myst2B3['Level (dB)']/10))
myst2Bint4 = 10**(myst2B4['Level (dB)']/10)/max(10**(myst2B4['Level (dB)']/10))

myst2Enu = myst2E1['Frequency (Hz)']
myst2Eint = (myst2Eint1 + myst2Eint2 + myst2Eint3 + myst2Eint4 + myst2Eint5 + myst2Eint6)/max(myst2Eint1 + myst2Eint2 + myst2Eint3 + myst2Eint4 + myst2Eint5 + myst2Eint6)

myst2Bnu = myst2B1['Frequency (Hz)']
myst2Bint = (myst2Bint1 + myst2Bint2 + myst2Bint3 + myst2Bint4)/max(myst2Bint1 + myst2Bint2 + myst2Bint3 + myst2Bint4)

#Finding peaks
myst2Ep1 = sgn.find_peaks(myst2Eint1, prominence=0.05)
myst2Ep2 = sgn.find_peaks(myst2Eint2, prominence=0.05)
myst2Ep3 = sgn.find_peaks(myst2Eint3, prominence=0.05)
myst2Ep4 = sgn.find_peaks(myst2Eint4, prominence=0.05)
myst2Ep5 = sgn.find_peaks(myst2Eint5, prominence=0.05)
myst2Ep6 = sgn.find_peaks(myst2Eint6, prominence=0.05)

myst2Bp1 = sgn.find_peaks(myst2Bint1, prominence=0.05)
myst2Bp2 = sgn.find_peaks(myst2Bint2, prominence=0.05)
myst2Bp3 = sgn.find_peaks(myst2Bint3, prominence=0.05)
myst2Bp4 = sgn.find_peaks(myst2Bint4, prominence=0.05)

myst2Eps = list(myst2Ep1[0]) + list(myst2Ep2[0]) + list(myst2Ep3[0]) + list(myst2Ep4[0]) + list(myst2Ep5[0]) + list(myst2Ep6[0])
myst2Bps = list(myst2Bp1[0]) + list(myst2Bp2[0]) + list(myst2Bp3[0]) + list(myst2Bp4[0])

#Plotting for mystery box 2
figure, axes = plt.subplots(1)
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

plt.xlim(0, MAX)
plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"Intensity [arb. unit]")
plt.title("Blind test container 1")
plt.grid()

plt.plot(myst2Enu, myst2Eint, label='Volume free')
plt.plot(myst2Bnu, myst2Bint, label='Volume blocked')

#for i in myst2Eps:
#	plt.axvline(myst2Enu[i], ls='--', c='b', lw=0.7)
#plt.axvline(myst2Enu[0], ls='--', c='b', lw=0.7, label='Volume free')

#for i in myst2Bps:
#	plt.axvline(myst2Bnu[i], ls='--', c='y', lw=0.7)
#plt.axvline(myst2Bnu[0], ls='--', c='y', lw=0.7, label='Volume blocked')

plt.legend(loc='upper right')
plt.savefig("mystery2.png", dpi=500)
plt.show()



#Mystery box 3
MAX = 1400 #Hz

#Import data
myst3E1 = pd.read_csv('mystery3_empty1.txt', sep='\t')
myst3E2 = pd.read_csv('mystery3_empty2.txt', sep='\t')
myst3E3 = pd.read_csv('mystery3_empty3.txt', sep='\t')
myst3E4 = pd.read_csv('mystery3_empty4.txt', sep='\t')

myst3B1 = pd.read_csv('mystery3_blocked1.txt', sep='\t')
myst3B2 = pd.read_csv('mystery3_blocked2.txt', sep='\t')
myst3B3 = pd.read_csv('mystery3_blocked3.txt', sep='\t')

myst3Enu1 = myst3E1['Frequency (Hz)']
myst3Enu2 = myst3E2['Frequency (Hz)']
myst3Enu3 = myst3E3['Frequency (Hz)']
myst3Enu4 = myst3E4['Frequency (Hz)']

myst3Bnu1 = myst3B1['Frequency (Hz)']
myst3Bnu2 = myst3B2['Frequency (Hz)']
myst3Bnu3 = myst3B3['Frequency (Hz)']

myst3Eint1 = 10**(myst3E1['Level (dB)']/10)/max(10**(myst3E1['Level (dB)']/10))
myst3Eint2 = 10**(myst3E2['Level (dB)']/10)/max(10**(myst3E2['Level (dB)']/10))
myst3Eint3 = 10**(myst3E3['Level (dB)']/10)/max(10**(myst3E3['Level (dB)']/10))
myst3Eint4 = 10**(myst3E4['Level (dB)']/10)/max(10**(myst3E4['Level (dB)']/10))

myst3Bint1 = 10**(myst3B1['Level (dB)']/10)/max(10**(myst3B1['Level (dB)']/10))
myst3Bint2 = 10**(myst3B2['Level (dB)']/10)/max(10**(myst3B2['Level (dB)']/10))
myst3Bint3 = 10**(myst3B3['Level (dB)']/10)/max(10**(myst3B3['Level (dB)']/10))

myst3Enu = myst3E1['Frequency (Hz)']
myst3Eint = (myst3Eint1 + myst3Eint2 + myst3Eint3 + myst3Eint4)/max(myst3Eint1 + myst3Eint2 + myst3Eint3 + myst3Eint4)

myst3Bnu = myst3B1['Frequency (Hz)']
myst3Bint = (myst3Bint1 + myst3Bint2 + myst3Bint3)/max(myst3Bint1 + myst3Bint2 + myst3Bint3)

#Finding peaks
myst3Ep1 = sgn.find_peaks(myst3Eint1, prominence=0.05)
myst3Ep2 = sgn.find_peaks(myst3Eint2, prominence=0.05)
myst3Ep3 = sgn.find_peaks(myst3Eint3, prominence=0.05)
myst3Ep4 = sgn.find_peaks(myst3Eint4, prominence=0.05)

myst3Bp1 = sgn.find_peaks(myst3Bint1, prominence=0.05)
myst3Bp2 = sgn.find_peaks(myst3Bint2, prominence=0.05)
myst3Bp3 = sgn.find_peaks(myst3Bint3, prominence=0.05)

myst3Eps = list(myst3Ep1[0]) + list(myst3Ep2[0]) + list(myst3Ep3[0]) + list(myst3Ep4[0])
myst3Bps = list(myst3Bp1[0]) + list(myst3Bp2[0]) + list(myst3Bp3[0])

#Plotting for mystery box 3
figure, axes = plt.subplots(1)
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

plt.xlim(0, MAX)
plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"Intensity [arb. unit]")
plt.title("Blind test container 2")
plt.grid()

plt.plot(myst3Enu, myst3Eint, label='Volume free')
plt.plot(myst3Bnu, myst3Bint, label='Volume blocked')

#for i in myst3Eps:
#	plt.axvline(myst3Enu[i], ls='--', c='b', lw=0.7)
#plt.axvline(myst3Enu[0], ls='--', c='b', lw=0.7, label='Volume free')

#for i in myst3Bps:
#	plt.axvline(myst3Bnu[i], ls='--', c='y', lw=0.7)
#plt.axvline(myst3Bnu[0], ls='--', c='y', lw=0.7, label='Volume blocked')

plt.legend(loc='upper right')
plt.savefig("mystery3.png", dpi=500)
plt.show()



#Tea Box
x1 = pd.read_csv('cylinder_hardsurface_1.txt', sep='\t')
x2 = pd.read_csv('cylinder_bottleinside_1.txt', sep='\t')
x3 = pd.read_csv('cylinder_bottleinside_2.txt', sep='\t')
x4 = pd.read_csv('cylinder_bottleinside_3.txt', sep='\t')

#Plotting
figure, axes = plt.subplots(1)
#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})

plt.xlabel(r"Frequency $\nu$ [Hz]")
plt.ylabel(r"Intensity [arb. unit]")
plt.title("Cylindrical geometry")
plt.grid()

plt.plot(x1['Frequency (Hz)'][:1200], 10**(x1['Level (dB)'][:1200]/10)/8, label="No obstacle")
plt.plot(x2['Frequency (Hz)'][:1200], 10**(x2['Level (dB)'][:1200]/10), label="Volume obstacle")
plt.plot(x3['Frequency (Hz)'][:1200], 10**(x3['Level (dB)'][:1200]/10), label="Volume obstacle")
plt.plot(x4['Frequency (Hz)'][:1200], 10**(x4['Level (dB)'][:1200]/10), label="Volume obstacle")

plt.legend(loc='upper left')
plt.savefig("teapeaks.png", dpi=500)
plt.show()