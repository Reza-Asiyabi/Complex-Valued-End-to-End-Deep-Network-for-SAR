from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import scipy.fft
import scipy.io as sio

##################################### Import data

range_spacing = 3.246363 ## S1 (2.246363) ## RS2 (4.7330)
azimute_spacing = 4.555256 ## S1 (3.555256) ## RS2 (5.1816)
Fs = 67.7283950933333 #  RS2
PRF = 1718.1290
Azimuth_HM_coef = 1.75
Range_HM_coef = 1.75
img_VH = sio.loadmat("D:/Reza/VH_mat_complex/Complex_VH_2021_08.mat")
img_VH = img_VH['Img_VH_2021_08']

img = img_VH

# dB_im = 20*np.log10(abs(img)/np.max(abs(img)) + 1e-10)    #dB conversion
#
# plt.figure()
# plt.imshow(dB_im, cmap='jet', origin='lower', aspect='auto', vmin=-70, vmax=0, extent=[0, (np.size(img, 0)-1)*range_spacing, 0, (np.size(img, 1)-1)*azimute_spacing])
# plt.suptitle('dB_im data')
# plt.colorbar()
# plt.xlabel("Slant Range [m]")
# plt.ylabel("Azimuth [m]")
# plt.show()

# im = Image.fromarray(dB_im)
# im.save('FFT outputs/Santos/Subset02/HV/dB_im.tiff')

############################################ calculate the spectrum

im_fft2 = scipy.fft.fftshift(scipy.fft.fft2(img))
# dB_im_fft2 = 20*np.log10(abs(im_fft2)/np.max(abs(im_fft2)) + 1e-10)
# plt.figure()
# plt.imshow(dB_im_fft2, cmap='jet', vmin=-50, vmax=0, origin='lower', aspect='auto', extent=[-Fs/2/1e6, Fs/2/1e6, -PRF/2, PRF/2])
# plt.suptitle('2D Spectrum')
# plt.colorbar()
# plt.xlabel("Range Frequency [MHz]")
# plt.ylabel("Azimuth Frequency [Hz]")
# plt.show()

# im = Image.fromarray(dB_im_fft2)
# im.save('FFT outputs/Santos/Subset02/HV/dB_im_fft2.tiff')

# #################################################### Sinc function and Hamming window effect
# X = np.arange(-100, 100, 0.1)
# Sinc = np.sinc(X)
#
# Sinc_FFT = scipy.fft.fft(Sinc)
#
# ham = np.zeros(np.size(X))
# h = np.hamming(200)
# ham[900:1100] = h
#
# ham_Sinc_FFT = Sinc_FFT*ham
#
# iSinc = scipy.fft.fftshift(scipy.fft.ifft(ham_Sinc_FFT))
# iSinc = iSinc/np.max(iSinc)
#
# plt.figure()
# plt.plot(X, abs(iSinc), X, abs(Sinc))
# plt.gca().set_title("Sinc function before and after applying the hamming window")
# plt.show()


############################################## Create the Hamming window
M = np.size(img, 0)
h1 = []
for i in range(M):
    h1.append(Azimuth_HM_coef - (1-Azimuth_HM_coef)*np.cos(2*np.pi*(i/(M-1))))

M = np.size(img, 1)
h2 = []
for i in range(M):
    h2.append(Range_HM_coef - (1-Range_HM_coef)*np.cos(2*np.pi*(i/(M-1))))

# h1 = np.hamming(2273)
# h2 = np.hamming(4129)
ham2d = np.outer(h1, h2)

# plt.figure()
# plt.subplot(221)
# plt.plot(h1)
#
# plt.subplot(222)
# plt.plot(h2)
#
# plt.subplot(2, 2, (3, 4))
# plt.imshow(ham2d, cmap="jet")
# plt.suptitle('Hamming Window')
#
# plt.tight_layout()
# plt.show()

####################################################### Subtracting the Hamming window effect on the data
WN = im_fft2/ham2d
# dB_WN = 20*np.log10(abs(WN)/np.max(abs(WN)) + 1e-10)
#
# plt.figure()
# plt.subplot(211)
# plt.imshow(dB_WN, cmap='jet', vmin=-50, vmax=0, origin='lower', aspect='auto', extent=[-Fs/2/1e6, Fs/2/1e6, -PRF/2, PRF/2])
# plt.gca().set_title('2D Spectrum after dividing by the Hamming window')
# plt.colorbar()
# plt.xlabel("Range Frequency [MHz]")
# plt.ylabel("Azimuth Frequency [Hz]")
#
# plt.subplot(212)
# plt.imshow(dB_im_fft2, cmap='jet', vmin=-50, vmax=0, origin='lower', aspect='auto', extent=[-Fs/2/1e6, Fs/2/1e6, -PRF/2, PRF/2])
# plt.gca().set_title('2D Spectrum before dividing by the Hamming window')
# plt.colorbar()
# plt.xlabel("Range Frequency [MHz]")
# plt.ylabel("Azimuth Frequency [Hz]")
#
# plt.tight_layout()
# plt.show()

# im = Image.fromarray(dB_WN)
# im.save('FFT outputs/Santos/Subset02/HV/dB_WN.tiff')

############################################################# creating the image after subtracting the Hamming window
im_WN = scipy.fft.ifft2(WN)
# dB_im_WN = 20*np.log10(abs(im_WN)/np.max(abs(im_WN)) + 1e-10)
#
# plt.figure()
# plt.subplot(211)
# plt.imshow(dB_im_WN, cmap='jet', origin='lower', aspect='auto', vmin=-60, vmax=0, extent=[0, (np.size(img, 0)-1)*range_spacing, 0, (np.size(img, 1)-1)*azimute_spacing])
# plt.gca().set_title('Data after dividing by the Hamming window')
# plt.colorbar()
# plt.xlabel("Slant Range [m]")
# plt.ylabel("Azimuth [m]")
#
# plt.subplot(212)
# plt.imshow(dB_im, cmap='jet', origin='lower', aspect='auto', vmin=-60, vmax=0, extent=[0, (np.size(img, 0)-1)*range_spacing, 0, (np.size(img, 1)-1)*azimute_spacing])
# plt.gca().set_title('Data before dividing by the Hamming window')
# plt.colorbar()
# plt.xlabel("Slant Range [m]")
# plt.ylabel("Azimuth [m]")
#
# plt.tight_layout()
# plt.show()

# im = Image.fromarray(dB_im_WN)
# im.save('FFT outputs/Santos/Subset02/HV/dB_im_WN.tiff')


###################################################################### Subaperture Processing
##################################### Subaperture hamming window creation
azimuth_factor = 2
overlapping_percentage = 0
azimuth_M = int((np.size(img, 0)/azimuth_factor)*(1 + overlapping_percentage))
h1 = []
for i in range(azimuth_M):
    h1.append(Azimuth_HM_coef - (1-Azimuth_HM_coef)*np.cos(2*np.pi*(i/(azimuth_M-1))))

range_factor = 1
range_M = int(np.size(img, 1)/range_factor)
h2 = []
for i in range(range_M):
    h2.append(Range_HM_coef - (1-Range_HM_coef)*np.cos(2*np.pi*(i/(range_M-1))))

ham2d = np.outer(h1, h2)
# ham2d = np.hstack((ham2d, ham2d, ham2d, ham2d))
# ham2d = np.vstack((ham2d, ham2d))

# plt.figure()
# plt.imshow(ham2d)
# plt.gca().set_title('Subaperture Hamming window')
# plt.show()

############################################################# Applying the subaperture hamming window
sub_ham2d = []
sub_ham_FFT = []
sub_ham_FFT_im = []
dB_sub_ham_FFT_im = []

centered_sub_ham_FFT = []
centered_sub_ham_FFT_im = []
dB_centered_sub_ham_FFT_im = []

for i in range(azimuth_factor):
    for j in range(range_factor):
        subham2d_temp = np.zeros(np.shape(img))
        if i == 0:
            ix1 = int(i*int((np.size(img, 0)/azimuth_factor)*(1 - overlapping_percentage)))
            ix2 = int((i+1)*int((np.size(img, 0)/azimuth_factor)*(1 + overlapping_percentage)))
        elif i == azimuth_factor-1:
            ix1 = int(i*int((np.size(img, 0)/azimuth_factor))-((np.size(img, 0)/azimuth_factor)* overlapping_percentage))
            ix2 = int((i+1)*int((np.size(img, 0)/azimuth_factor)))
        else:
            ix1 = int(i*int((np.size(img, 0)/azimuth_factor))-((np.size(img, 0)/azimuth_factor)* overlapping_percentage/2))
            ix2 = int((i+1)*int((np.size(img, 0)/azimuth_factor))+((np.size(img, 0)/azimuth_factor)* overlapping_percentage/2))

        subham2d_temp[ix1:ix2, j*range_M:(j+1)*range_M] = ham2d
        sub_ham2d.append(subham2d_temp)
        ham_FFT_temp = subham2d_temp*WN
        sub_ham_FFT.append(ham_FFT_temp)
        ham_FFT_im_temp = scipy.fft.ifft2(ham_FFT_temp)
        sub_ham_FFT_im.append(ham_FFT_im_temp)
        dB_ham_FFT_im_temp = 20*np.log10(abs(ham_FFT_im_temp)/np.max(abs(ham_FFT_im_temp)) + 1e-10)
        dB_sub_ham_FFT_im.append(dB_ham_FFT_im_temp)

        centered_sub_ham_FFT_temp = np.zeros(np.shape(img), dtype="complex")
        centered_sub_ham_FFT_temp[int((np.size(img, 0)/2)-(np.size(ham2d, 0)/2)):int((np.size(img, 0)/2)+(np.size(ham2d, 0)/2)), :] = ham_FFT_temp[ix1:ix2, j*range_M:(j+1)*range_M]
        centered_sub_ham_FFT.append(centered_sub_ham_FFT_temp)
        centered_ham_FFT_im_temp = scipy.fft.ifft2(centered_sub_ham_FFT_temp)
        centered_sub_ham_FFT_im.append(centered_ham_FFT_im_temp)
        dB_centered_ham_FFT_im_temp = 20*np.log10(abs(centered_ham_FFT_im_temp)/np.max(abs(centered_ham_FFT_im_temp)) + 1e-10)
        dB_centered_sub_ham_FFT_im.append(dB_centered_ham_FFT_im_temp)

        del subham2d_temp, ham_FFT_temp, ham_FFT_im_temp, dB_ham_FFT_im_temp, centered_sub_ham_FFT_temp, centered_ham_FFT_im_temp, dB_centered_ham_FFT_im_temp

############################################ Showing subaperture images
plt.figure()
for i in range(len(sub_ham2d)):
    plt.subplot(azimuth_factor, range_factor, i+1)
    plt.imshow(sub_ham2d[i], cmap='jet', origin='lower', aspect='auto', extent=[-Fs/2/1e6, Fs/2/1e6, -PRF/2, PRF/2])
    plt.xlabel("Range Frequency [MHz]")
    plt.ylabel("Azimuth Frequency [Hz]")
    plt.colorbar()
    # im = Image.fromarray(sub_ham2d[i])
    # im.save('FFT outputs/Santos/Subset02/HV/sub_ham2d%s.tiff' %i)
plt.suptitle("Subaperture hamming windows")
plt.show()

plt.figure()
for i in range(len(centered_sub_ham_FFT)):
    plt.subplot(azimuth_factor, range_factor, i+1)
    dB_temp = 20*np.log10(abs(centered_sub_ham_FFT[i])/np.max(abs(centered_sub_ham_FFT[i])) + 1e-10)
    plt.imshow(dB_temp, cmap='jet', vmin=-50, vmax=0, origin='lower', aspect='auto', extent=[-Fs/2/1e6, Fs/2/1e6, -PRF/2, PRF/2])
    plt.xlabel("Range Frequency [MHz]")
    plt.ylabel("Azimuth Frequency [Hz]")
    plt.colorbar()
    # im = Image.fromarray(dB_temp)
    # im.save('FFT outputs/Santos/Subset02/HV/centered_sub_ham_FFT%s.tiff' %i)
plt.suptitle("Centered Subaperture spectrum")
plt.show()

plt.figure()
for i in range(len(dB_centered_sub_ham_FFT_im)):
    plt.subplot(azimuth_factor, range_factor, i+1)
    plt.imshow(dB_centered_sub_ham_FFT_im[i], cmap='jet', origin='lower', aspect='auto', vmin=-70, vmax=0)
    plt.xlabel("Slant Range [m]")
    plt.ylabel("Azimuth [m]")
    plt.colorbar()
    # im = Image.fromarray(dB_centered_sub_ham_FFT_im[i])
    # im.save('FFT outputs/Santos/Subset02/HV/3 Subapertures/dB_centered_sub_ham_FFT_im%s.tiff' %i)
plt.suptitle("Subaperture images")
plt.show()

dict = {"Sub1_VH_2021_08": centered_sub_ham_FFT_im[0], "Sub2_VH_2021_08": centered_sub_ham_FFT_im[1]}
sio.savemat("D:/Reza/VH_SAs/Sub_VH_2021_08.mat", dict)

