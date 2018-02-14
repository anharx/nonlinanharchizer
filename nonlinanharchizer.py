#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NonlinAnharChizer: Nonlinearity and Anharmonicity Characterizer, is a script that evaluates the anharmonic shape
#  of (quasi-)periodic input signals and measures their degree of nonlinearity derived from it. To this effect, it uses
#  a multiresolution representation based on P. Hanusse's /Nonlinear Trigonometric/ functions from which it derives the
#  dual and projects it on the signal; this is done at each pixel. Then, from it, it calculates a synthetic sparsity
#  (which describes amplitude fluctuations, analogous to the shimmer of Fourier representation), the optimal shape
#  parameter (which represents departure from Fourier) and a nonlinearity measure (as the gain in sparsity from Fourier);
#  all these calculated for windows, which should be at least as large as the fundamental period.

# The projectors are calculated once and then applied to the signal, so that we obtain the representation coefficients directly.
#  The shape parameter itself is window-varying, so it needs an optimization routine at the window level (a huge improvement
#  from running optimizers at the pixel level!). This is the slowest part of the processing. A LUT approach could help here if
#  a big corpus of files needed to be batch processed, run in a big-mem system. By default, and in the absence of tolerance-based
#  stopping criteria, we let users define a max number of iterations from the command line. A bit of testing shows when outputs
#  stop moving any significantly.

# There are two command line parameters: "Window length" and "maximum number of iterations", which are the ones likely requiring
#  a rather finer tuning. Internally, there is also the output interval (by default, the same as the window length, i.e., no overlap)
#  and the size of the multiresolution pyramid, N. It is important that N is large enough to capture the larger-scale dynamics,
#  while simultaneously small enough to avoid a sparsity being dominated by the nearly null low-frequency coefficients, which would
#  unfairly favor Fourier. N=32 seems to work well for 16kHz speech; check the matplots of lvlEnergy if you want to try other values.

# The representation basis is a multiresolution frame based on P. Hanusse's psin_1 and pcos_1 functions (for which the definition
#  series have closed expressions). Like with Fourier series', it is possible to use both psin and pcos, or just either. To complete
#  the basis at finite resolution, we add a unit function. The effect of this on the representation gets canceled if it is tuned to
#  the same shape of the analysis window. On the absence of any prior information about the fundamental period, a Hanning window
#  enforces the required periodicity; therefore, we add a Hanning kernel as the unit function for representation, which will be taking
#  the bulk of the /approximation/ component. Doing this, then the cosine-like functions are worse suited than the sine-like functions
#  for representing the /details/ component. In consequence, in this case, we will be using by default psin_1 exclusively (pcos_1 is
#  also implemented nevertheless, for testing purposes).



########################
#   CODE STARTS HERE   #
########################


# A handful of standard modules required:

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import wavfile
import argparse


# Parsing of command line arguments

parser = argparse.ArgumentParser(description='Evaluate anharmonic shape of input files and characterize a nonlinearity measure from it.')

def reasonableWindowType(x):
 if 1 <= int(x) <= 1E5:
  return int(x)
 else:
  raise argparse.ArgumentTypeError("Window should be larger than 1 and reasonably narrow")

parser.add_argument('-w', '--window', type=reasonableWindowType, metavar='WINDOW_LENGTH', help='length of the processing window (in pixels)')

parser.add_argument('--iter', type=int, choices=range(3, 100), metavar='MAX_ITERATIONS', help='max number of iterations')

parser.add_argument('-s', '--save-matrix', action='store_true', help='output the plot of the level energies as a PNG file instead of displaying it interactively')

parser.add_argument('wavfile', type=argparse.FileType('r'), nargs='?', help='name of the input wav file')

args = parser.parse_args()


# Global constants

# Length of the processing window over which we will extract the sparsity, shape and nonlinearity
# The default 4096 corresponds to 250ms at 16kHz sampling
WINDOW = args.window or 4096

# Length of the interval step for processing
# By default, the same as WINDOW, which makes non-overlapping windows (make them overlapping to see finer details)
INTERVAL = WINDOW

# Number-of-iterations limit used for the optimal-shape minimizer
# By default 10, which are not many, but probably enough for a no-gradients, 1D, smooth-landscape optimization
MAXITER = args.iter or 10

# Size of the dyadic multiresolution pyramid; a power of two
N = 32

# The anharmonic shape parameter r ranges from 0 to 1, with r=0 equivalent to standard trigonometry (psin -> sin, pcos -> cos)
# However, the numeric implementations do not converge at the extremes; use a tiny value in lieu of 0 and its 1-complement for 1
TINY = 1E-6

# Half to the half, or reciprocal of the Pythagoras' constant
recPytha = 1. / np.sqrt(2.)


# Function definitions

def log2int(x):
 "Fast integer part of the base-2 logarithm; mostly useful for powers of two but works for any real nonzero input" 
 return np.frexp(x)[1] - 1

def normalization(array):
 "Square root of the average square; a.k.a the L2-norm, when array values are the images of a function whose domain ranges (0, 1)"
 return np.sqrt(np.mean(np.square(array)))

# There is ambiguity in the translational shift and this can induce horizontal biases in the projector.
# To solve this, we use the fftshift trick, to cancel the shift in time-frequency space (even if we do nothing with Fourier).
# With this, the duals retain time-domain localization with no shift bias.

def psin(n=N, r=TINY):
 "Return an array with a period of the anharmonic sine at form r; very small for the trigonometric sine"
 if not TINY <= r <= 1.-TINY:
  raise ValueError('psin needs r well between 0 and 1')
 psin = np.arctan(r * np.sin(2. * np.pi / n * np.arange(n)) / (1. - r * np.cos(2. * np.pi / n * np.arange(n))))
 psin /= normalization(psin)
 return np.fft.fftshift(psin)

def pcos(n=N, r=TINY):
 "Return an array with a period of the anharmonic cosine at form r; very small for the trigonometric cosine"
 if not TINY <= r <= 1.-TINY:
  raise ValueError('pcos needs r well between 0 and 1')
 pcos = -.5 * np.log(1. + r * r - 2. * r * np.cos(2. * np.pi / n * np.arange(n)))
 pcos /= normalization(pcos)
 return np.fft.fftshift(pcos)

# Use a Hanning window as the unit function. This way, our "pixel" representation or PSF is synced with the filtering.
# As a consequence, we can get rid of the border effects without altering the representation.
# The window has Linf normalization by default
# The advantage of using L2 this is that projector and representer are like valued:

def unit(n=N):
 "Return an array with a unit function, consisting of a normalized Hanning window"
 unit = np.hanning(n)
 unit /= normalization(unit)
 return unit

# We will be representing the frames and the unit function as column vectors, and we will combine them in a matrix:

def basis(n=N, r=TINY):
 "Return a bidimensional array whose columns are the basis vectors"
 frame = psin(n, r)
 basis = unit(n)
 # Now let us iteratively add the multiple resolutions:
 halfN = n // 2
 width = n
 m = 1
 while m < halfN: # Leave out the last level (1-pixel-long resolution), to avoid semiperiod cancellation with the psin
  mass = np.copy(frame[::m]) * np.sqrt(m)
 # Normalization (not necessary in practice):
 # offset = np.mean(mass)
 # mass -= offset
 # weight = np.sqrt(np.sum(np.square(mass))/N) # L2
 # weight = np.max(mass) # Linf
 # mass /= weight
  for l in range(m): # m times from 0 to m-1
   preZ = l * width
   postZ = n - width - preZ
   atom = np.r_[np.zeros(preZ), mass, np.zeros(postZ)]
   basis = np.c_[basis, atom]
  width >>= 1
  m <<= 1
 # Correction for the last level
 mass = np.empty(2)
 mass[0] = +np.sqrt(n / 2.)
 mass[1] = -np.sqrt(n / 2.)
 for l in range(n // 2):
  preZ = l * 2
  postZ = n - 2 - preZ
  atom = np.r_[np.zeros(preZ), mass, np.zeros(postZ)]
  basis = np.c_[basis, atom]
 return basis

# Consider that the dual elements (covectors), <A'_jk| are such that they resonate with their respective pairs
# and they cancel with any other, i.e., <A'_jk|A_lm> = δ_jl δ_km. Now, it is possible to extend the same requirement
# to the unit function so that <dual_i|basis_j> = δ_ij and then the duals get defined simply as a matrix inversion.
# The matrix basis is nonsingular by construction, as we canceled the mode 0 for the detail elements and used
# a positive unit function.
# Reciprocal to the unit function we have a constant function, because all the Hanusse atoms average-cancel at each period.

def getDual(n=N, r=TINY):
 "Call the basis() procedure to get the basis column-vectors and get, through inversion, the dual line-covectors"
 return np.linalg.inv(basis(n, r))

#plt.plot(np.transpose(getDual()[32:64,:])); plt.show()
#plt.plot(basis()[:,16:32]); plt.show()

def getAnalysis(n=N, r=TINY):
 "For analysis purposes, normalize the dual"
 dual = getDual(n, r)
 for i in range(1, n):
  normalCovector = dual[i]
  normalCovector -= np.mean(normalCovector) # should already be close to 0 though
  normalCovector /= normalization(normalCovector)
 return dual

def project_old(signal, n=N, r=TINY):
 "Window-filter a signal, project it through the covectors and return the projections (coefficients) as a bidimensional array"
 window = unit(n)
 dual = getDual(n, r)
 length = signal.size - n
 projections = np.empty([n, length])
 for t in range(length):
  filtdata = np.multiply(signal[t:t+n], window)
  projections[:,t] = np.dot(dual, filtdata) # filtdata is 1D (not column) but nevertheless the result (also 1D) is correct
 return projections

def project(signal, n=N, r=TINY, matsignal=None):
 "Window-filter a signal, project it through the covectors and return the projections (coefficients) as a bidimensional array"
 # Window-filter the rows of the dual matrix; this is equivalent but faster than doing it on the signal segments. which are of size n anyways 
 filtdual = np.multiply(getDual(n, r), unit(n)) # Dual matrix, filtered
            #           #              # 1D window
            #           # 2D matrix with covectors as rows 
            # multiply rows element-wise
 if matsignal is None:
  length = signal.size - n
  matsignal = np.empty([n, length])
  for t in range(length):
   matsignal[:,t] = signal[t:t+n]
 return np.dot(filtdual, matsignal)

def lvlEnergy(signal, n=N, r=TINY, matsignal=None):
 "For each set of coefficients in a same level, return its /energy/, or square root of the sum of squares"
 projections = project(signal, n=n, r=r, matsignal=matsignal)
 averages = projections[0] # take the first component, which should be the same as all the others
 length = signal.size - n
 levels = log2int(n)
 lvlEnergy = np.empty([levels, length])
 for j in range(levels):
  lvlEnergy[j] = np.linalg.norm(projections[2**j:2*2**j], axis=0)
 return averages, lvlEnergy

#thing = lvlEnergy(wav[1000:2256], n=256)[1]
#plt.matshow(thing, origin='upper', aspect='auto'); plt.colorbar(); plt.show()

def getSparsity(r, signal, n=N, matsignal=None):
 "From the level energy vector, return the /sparsity/ of the representation defined as the 1-norm of this"
 return np.average(np.fabs(lvlEnergy(signal, n=n, r=r, matsignal=matsignal)[1]))


# Processing starts here

if args.wavfile is None:
 # The script was invoked without a filename; launch a GUI dialog to ask for one
 from tkinter import Tk
 from tkinter.filedialog import askopenfilename
 Tk().withdraw()
 in_filename = askopenfilename()
else:
 in_filename = args.wavfile.name

wav = wavfile.read(in_filename)[1]
out_filename = in_filename.rpartition('.')[0] + '.txt'
out_figure = in_filename.rpartition('.')[0] + '.svg'

print("\nProcessing results outputted to file: " + out_filename + "\n")
out = open(out_filename, 'wt')

for t in range(0, wav.size-WINDOW-N, INTERVAL):
 signal = wav[t:t+WINDOW+N].astype(float)
 matsignal = np.empty([N, WINDOW])
 for l in range(WINDOW):
  matsignal[:,l] = signal[l:l+N]
 fourierSparsity = getSparsity(TINY, signal, n=N, matsignal=matsignal)
 result = optimize.minimize_scalar(getSparsity, args=(signal, N, matsignal), method='bounded', bounds=(TINY, 1.-TINY), options={'maxiter': MAXITER})
 if fourierSparsity < result.fun:
  sparsity = fourierSparsity
  r = TINY
  nonlin = 0.
 else:
  sparsity = result.fun
  r = result.x
  nonlin = np.log(fourierSparsity / sparsity)
 sparsity /= np.average(np.fabs(signal))
 out.write('{} {} {}\n'.format(sparsity, r, nonlin))
 print('{:4.1f}%:  {:5.1f}  {:6.4f}  {:6.4f}'.format(100.*t/wav.size, sparsity, r, nonlin))

out.close()

lvlEnergies = lvlEnergy(wav)[1]
plt.matshow(lvlEnergies, origin='upper', aspect='auto')
plt.colorbar()

if args.save_matrix:
 plt.savefig(out_figure, dpi=2000)
 print("\nPlot of level energies saved to file: " + out_figure)
#else:
# plt.show(block=True) # this shows a dialogue to save the figure too
