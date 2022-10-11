import A2Part1 as a2p1
import numpy as np
import A2Part2 as a2p2
import A2Part3 as a2p3
import A2Part4 as a2p4


A=1.0
f = 10.0
phi = 1.0
fs = 50.0 
t = 0.1
y = a2p1.genSine(A, f, phi, fs, t)

print (y)

print ("######## Part 2")


k = 1

N = 5

nsin = a2p2.genComplexSine(k,N)

print (nsin)

for x in np.nditer(nsin.T):
    print(x)

print ("###### Part 2 ex.1")

# import matplotlib.pyplot as plt
# import numpy as np

# N=500
# k=3

# n2 = np.arange(-N/2,N/2)
# s = a2p2.genComplexSine(k,N)
# plt.plot (n2, np.imag(s))
# plt.axis([-N/2,N/2-1,-1,1])
# plt.xlabel('n')
# plt.ylabel('amplitude')

# plt.show()


print ("########### Part 2 A3")

testVar = np.arange(1,5)


print (testVar)

dftOutput = a2p3.DFT(testVar)

print(dftOutput)


print("##### Part2 A4")

testIdft = np.array([1 ,1 ,1 ,1])

idftOutput = a2p4.IDFT(testIdft)

print (idftOutput)



print("## test DFT then switch to IDFT")

testVar = np.array([1,0,0,0])

print ("testVar1 = ",testVar)

dftOutput = a2p3.DFT(testVar)

print ("dftOutput1 = ",dftOutput)

idftOutput1 = a2p4.IDFT(testIdft)

print ("idftOutput1 = ", idftOutput1)

print ("result = ", np.array_equal(testVar,idftOutput1))


############


import A2Part5 as a2p5

test = np.array([1, 2, 3, 4])

y = a2p5.genMagSpec(test)

print (y)
