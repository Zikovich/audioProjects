import sys
sys.path.append('../A2')

import A2Part1 as a2p1
import A3Part1 as a3p1
import numpy as np
from matplotlib import pyplot as plt

'''
Test case 1: For an input signal x sampled at fs = 10000 Hz that consists of sinusoids of frequency 
f1 = 80 Hz and f2 = 200 Hz, you need to select M = 250 samples of the signal to meet the required 
condition. In this case, output mX is 126 samples in length and has non-zero values at bin indices 2 
and 5 (corresponding to the frequency values of 80 and 200 Hz, respectively). You can create a test 
signal x by generating and adding two sinusoids of the given frequencies.
'''

A=1.0
f1 = 80
phi =0
fs = 10000 
t = 1
y1 = a2p1.genSine(A, f1, phi, fs, t)

f2 = 200
y2 = a2p1.genSine(A, f2, phi, fs, t)

x= y1 + y2

mX = a3p1.minimizeEnergySpreadDFT(x,fs,f1,f2)

print (mX)

plt.plot(mX)
plt.show()

## Second test 

'''
Test case 2: For an input signal x sampled at fs = 48000 Hz that consists of sinusoids of frequency 
f1 = 300 Hz and f2 = 800 Hz, you need to select M = 480 samples of the signal to meet the required 
condition. In this case, output mX is 241 samples in length and has non-zero values at bin indices 3 
and 8 (corresponding to the frequency values of 300 and 800 Hz, respectively). You can create a test 
signal x by generating and adding two sinusoids of the given frequencies.
'''

A=1.0
f1 = 300
phi =0
fs = 48000
t = 1
y1 = a2p1.genSine(A, f1, phi, fs, t)

f2 = 800
y2 = a2p1.genSine(A, f2, phi, fs, t)

x= y1 + y2

mX = a3p1.minimizeEnergySpreadDFT(x,fs,f1,f2)

print (mX)

plt.plot(mX)
plt.show()


######## PartA3 part 2

import A3Part2 as a3p2

'''
Test case 1: For a sinusoid x with f = 100 Hz, M = 25 samples and fs = 1000 Hz, you will need to 
zero-pad by 5 samples and compute an N = 30 point DFT. In the magnitude spectrum, you can see a 
maximum value at bin index 3 corresponding to the frequency of 100 Hz. The output mX you return is 
16 samples in length. 
'''

A=1.0
f1 = 100
phi =0
fs = 1000
t = 1
y1 = a2p1.genSine(A, f1, phi, fs, t)

mX = a3p2.optimalZeropad(y1[:25],fs,f1)

print (mX)

plt.plot(mX)
plt.show()

'''
Test case 2: For a sinusoid x with f = 250 Hz, M = 210 samples and fs = 10000 Hz, you will need to 
zero-pad by 30 samples and compute an N = 240 point DFT. In the magnitude spectrum, you can see a 
maximum value at bin index 6 corresponding to the frequency of 250 Hz. The output mX you return is 
121 samples in length. 
'''
A=1.0
f1 = 250
phi =0
fs = 10000
t = 1
y1 = a2p1.genSine(A, f1, phi, fs, t)

mX = a3p2.optimalZeropad(y1[:210],fs,f1)

print (mX)

plt.plot(mX)
plt.show()


######## A3Part3 

import A3Part3 as a3p3

'''
Test case 1: If x = np.array([ 2, 3, 4, 3, 2 ]), which is a real and even signal (after zero phase 
windowing), the function returns (True, array([ 4., 3., 2., 2., 3.]), array([14.0000+0.j, 2.6180+0.j, 
0.3820+0.j, 0.3820+0.j, 2.6180+0.j])) (values are approximate)
'''
x = np.array([ 2, 3, 4, 3, 2 ])

isRealEven, dftBuffer, XFft = a3p3.testRealEven(x)

outputReal = a3p3.testRealEven(x)

print ('outputReal = ', outputReal)

print (dftBuffer)
plt.plot(dftBuffer)
plt.show()

print(XFft)
plt.plot(XFft)
plt.show()


'''
Test case 2: If x = np.array([1, 2, 3, 4, 1, 2, 3]), which is not a even signal (after zero phase 
windowing), the function returns (False,  array([ 4.,  1.,  2.,  3.,  1.,  2.,  3.]), array([ 16.+0.j, 
2.+0.69j, 2.+3.51j, 2.-1.08j, 2.+1.08j, 2.-3.51j, 2.-0.69j])) (values are approximate)
'''

x = np.array([1, 2, 3, 4, 1, 2, 3])

isRealEven, dftBuffer, XFft = a3p3.testRealEven(x)

print (isRealEven)

print (dftBuffer)
plt.plot(dftBuffer)
plt.show()

print(XFft)
plt.plot(dftBuffer)
plt.show()


############ A3 Part4

import A3Part4 as a3p4
'''
Test case 1: For an input signal with 40 Hz, 100 Hz, 200 Hz, 1000 Hz components, yfilt will only contain
100 Hz, 200 Hz and 1000 Hz components. 
'''

N = 2**13

A=1.0
f1 = 40
phi =0
fs = 5000
t = 1
y1 = a2p1.genSine(A, f1, phi, fs, t)

f2 = 100
y2 = a2p1.genSine(A, f2, phi, fs, t)

f3 = 200
y3 = a2p1.genSine(A, f3, phi, fs, t)

f4 = 1000
y4 = a2p1.genSine(A, f4, phi, fs, t)

yTotal = y1 + y2 + y3 + y4

(y, yfilt) = a3p4.suppressFreqDFTmodel(yTotal, fs, N)

print(y)
plt.plot(y)
plt.show()

print(yfilt)
plt.plot(yfilt)
plt.show()



'''
Test case 2: For an input signal with 23 Hz, 36 Hz, 230 Hz, 900 Hz, 2300 Hz components, yfilt will only contain
230 Hz, 900 Hz and 2300 Hz components. 
''' 
