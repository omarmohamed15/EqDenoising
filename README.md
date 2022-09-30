# EqDenoising

"Unsupervised deep learning for single-channel earthquake data denoising and its applications in event detection and fully automatic location"
September 2022IEEE Transactions on Geoscience and Remote Sensing PP(99):1-1
DOI:10.1109/TGRS.2022.3209932


We propose to use unsupervised deep learning (DL) and attention networks to mute the unwanted components of the single-channel earthquake data. The proposed algorithm is an unsupervised technique that does not require any prior information about the input data, i.e., no need for the labeled data. The imaginary and real parts of the short-time frequency transform (STFT) are divided into several overlapped patches to be the input of the proposed DL network, while the output target is the absolute value of the STFT. The proposed DL network utilizes a customized loss function to reconstruct the signal mask, where the STFT components related to the seismic noise are muted. An adaptive thresholding technique is utilized to obtain the binary mask, which is multiplied by the real and imaginary parts of the input seismic data. The binary mask has zero values for the samples corresponding to the unwanted components and ones for the seismic signal components. Then, inverse STFT is used to reconstruct the denoised signal. The proposed algorithm is evaluated using samples from the STanford EArthquake Dataset (STEAD) and the results are compared to the benchmark denoising method, i.e., DeepDenoiser. As a result, the proposed algorithm shows a robust denoising performance and outperforms the DeepDenoiser method by 1.95 dB in terms of signal-to-noise ratio.



