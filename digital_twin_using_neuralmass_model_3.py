# Neural Laplace EEG Signal Simulation Pipeline from multi channel EEG signal
# Multichannel EEG analytic signal: E_i(t)=x_i(t)+j*Hilbert(x_i(t))
#	--> every channel instantaneous amplitude a_j(t)=|E_i(t)|
#		--> initial amplitude:R_j(0)=a_j(0)
#	--> every channel insantaneous phase p_j(t)=angle[E_i(t)] unwrap
#		-->initial phase: Q_j(0)=p_j(0)
#		--> time average of time-derivative of each channel instantaneous phase w_j
#		--> across channel differneces of instantaneous phases: pd_ij(t)=p_i(t)-p_j(t)
#			--> coupling strength: abs of time average of complex exponential of cross channel phase diff, e^{j*pd_ij(t)}: C_ij 
#				--> Adjacency matrix: A_ij=max(0, C_ij-k); k=0.95
# Neural Mass Model for each channel: With R_j(0),Q_j(0), A_ij, w_j, dt=match sampling rate 128:0.0078125
#	--> Amplitude evolution: dR_dt_j=f(R_j,Q_j,Q_i,A_ij)+noise 
#		--> Amplitude integration: R_(t+dt)_j=R_t_j+dt*dR_dt_j
#			-->min Amplitude R_(t+dt)_j=max(0.01,R_(t+dt)_j)
#	--> Phase evlution: dQ_dt_j=f(w_j,R_j,R_i,Q_j,Q_i,A_ij)+noise
#		--> Phase integration: Q_(t+dt)_j=Q_t_j+dt*dQ_dt_j 
#			-->Phase correction modulo with 2*pi is not required (ignore this step)
# EEG simulation for each channel:
#	--> sim_eeg_j=R_j*cos(Q_j)+ f(noisy volume conduction matrix,R_j,R_i,Q_j,Q_i)
#*********************************************************************
import numpy as np
import torch
from scipy.signal import hilbert, butter, filtfilt, welch
import mne
import matplotlib.pyplot as plt
from scipy.signal import resample


# === Parameters ===
time_dur=int(input('enter time duration for modeling in secs: '))
dt = 0.0078125 #0.002
alpha = 0.1
beta = 0.5
kappa = 2.0
sigma_r = 0.01
sigma_theta = 0.05
sigma_noise = 0.1
lambda_thresh = 0.95
eps_min = 0.01
gamma0 = 0.1

def match_sampling_rate(simulated, target_len=256):
    # Resample to match original EEG
    return resample(simulated, target_len, axis=1)
    
# === EEG Preprocessing ===
def preprocess_eeg(eeg):
    analytic = hilbert(eeg, axis=1)
    phase = np.unwrap(np.angle(analytic))
    amplitude = np.abs(analytic)
    return analytic, phase, amplitude

# === Natural frequency (Equation 3) ===
def compute_omega(phase):
    dphi_dt = np.gradient(phase, dt, axis=1)
    return np.mean(dphi_dt, axis=1)

# === Phase difference and coupling strength ===
def compute_phase_diff_and_coupling(phase):
    N = phase.shape[0]
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dphi = phase[i] - phase[j]
                R[i, j] = np.abs(np.mean(np.exp(1j * dphi)))
    return R

def adjacency_matrix(R):
    A = np.maximum(0, R - lambda_thresh)
    return A

# === Neural Mass Model Dynamics ===
def dR_dt(R, theta, A):
    N = len(R)
    dR = -alpha * R
    for i in range(N):
        dR[i] += beta * np.sum(A[i] * R * np.cos(theta - theta[i]))
        dR[i] += sigma_r * np.random.randn()
    return dR

def dtheta_dt(R, theta, omega, A):
    N = len(R)
    dtheta = np.zeros(N)
    for i in range(N):
        dtheta[i] = omega[i] + kappa * np.sum(A[i] * (R / R[i]) * np.sin(theta - theta[i]))
        dtheta[i] += sigma_theta * np.random.randn()
    return dtheta

# === Simulate Neural Mass ===
def simulate_neural_mass(R0, theta0, omega, A, T):
    N = len(R0)
    steps = int(T / dt)
    R = np.zeros((N, steps))
    theta = np.zeros((N, steps))
    R[:, 0], theta[:, 0] = R0, theta0

    for t in range(steps - 1):
        dR = dR_dt(R[:, t], theta[:, t], A)
        dtheta = dtheta_dt(R[:, t], theta[:, t], omega, A)

        R[:, t+1] = np.maximum(R[:, t] + dt * dR, eps_min)
        theta[:, t+1] = (theta[:, t] + dt * dtheta) # % (2 * np.pi)

    return R, theta

# === Volume Conduction ===
def compute_gamma(N):
    gamma = gamma0 * np.random.rand(N, N)
    np.fill_diagonal(gamma, 1.0)
    #gamma = np.exp(-np.abs(np.arange(N).reshape(-1, 1) - np.arange(N)) / 2.0)
    return gamma

# === Reconstruct EEG ===
def reconstruct_eeg(R, theta, gamma):
    N, T = R.shape
    eeg = np.zeros((N, T))
    for i in range(N):
        eeg[i] = R[i] * np.cos(theta[i]) + np.sum(gamma[i][:, None] * R * np.cos(theta), axis=0) - gamma[i,i]*R[i]*np.cos(theta[i])
    #print(R.shape,eeg.shape)
    return eeg

# === Add Measurement Noise ===
def add_noise(eeg):
    return eeg + sigma_noise * np.random.randn(*eeg.shape)

# === Full Pipeline ===
def neural_laplace_simulate(eeg_input, T=2.0):
    EA, phi, R0 = preprocess_eeg(eeg_input)
    theta0 = phi[:, 0]
    omega = compute_omega(phi)
    R = compute_phase_diff_and_coupling(phi)
    A = adjacency_matrix(R)
    R_out, theta_out = simulate_neural_mass(R0[:, 0], theta0, omega, A, T)
    gamma = compute_gamma(len(R0))
    eeg_syn = reconstruct_eeg(R_out, theta_out, gamma)
    eeg_noisy = add_noise(eeg_syn)
    return eeg_noisy, R_out, theta_out

# === Run from EDF File ===
def run_simulation_from_edf(edf_path, channel_names=None, duration_sec=2.0, fs=128):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(fs)

    #if channel_names:
        #raw.pick_channels(channel_names)
    #    raw.pick(['Fcz.'])
    #else:
    #    raw.pick_types(eeg=True)

    data, _ = raw[:, :int(fs * duration_sec)]  # shape (n_channels, samples)
    print(data.shape)
    eeg_input = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    simulated_eeg, R, theta = neural_laplace_simulate(eeg_input, T=duration_sec)
    #simulated_eeg = match_sampling_rate(simulated_eeg, target_len=eeg_input.shape[1])
    #simulated_eeg = (simulated_eeg - np.mean(simulated_eeg, axis=1, keepdims=True)) / np.std(simulated_eeg, axis=1, keepdims=True)
    return eeg_input,simulated_eeg
    
eeg_file = 'S001R14.edf'
real_eeg,simulated_eeg=run_simulation_from_edf(eeg_file, duration_sec=time_dur)
print(real_eeg.shape,simulated_eeg.shape)


# === Define EEG Frequency Bands (Hz) ===
bands = {
    'Delta (0.5–4 Hz)': (0.5, 4),
    'Theta (4–8 Hz)': (4, 8),
    'Alpha (8–13 Hz)': (8, 13),
    'Beta (13–30 Hz)': (13, 30),
    'Gamma (30–45 Hz)': (30, 45),
}

# === Bandpass Filter Function ===
def bandpass_filter(signal, lowcut, highcut):
    nyq = 0.5 * 128
    b, a = butter(N=4, Wn=[lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

# === Apply Filters to One Channel ===
# real_eeg: shape (samples,), simulated_eeg: shape (samples,)
# fs_real = 128 (real EEG sampling rate), fs_sim = 500 or resampled to 128

filtered_real = {
    band: bandpass_filter(real_eeg, *freq)
    for band, freq in bands.items()
}

filtered_sim = {
    band: bandpass_filter(simulated_eeg, *freq)
    for band, freq in bands.items()
}


def interactive_band_plot(filtered_real, filtered_sim, bands):
    num_channels = list(filtered_real.values())[0].shape[0]
    fs=128
    time_axis=np.arange(0,time_dur,dt)
    #print(len(time_axis));exit(0)
    while True:
        try:
            ch = int(input(f"\nEnter channel number (0 to {num_channels - 1}, or -1 to quit): "))
            if ch == -1:
                print("Exiting...")
                break
            if ch < 0 or ch >= num_channels:
                print(f"Invalid channel number. Please enter a value between 0 and {num_channels - 1}.")
                continue

            # Create subplot
            fig, axes = plt.subplots(5, 3, figsize=(14, 10))
            fig.suptitle(f"EEG Bands - Real vs Simulated (Channel {ch})", fontsize=16)

            for i, (band, _) in enumerate(bands.items()):
                # Real EEG
                axes[i, 0].plot(time_axis,filtered_real[band][ch], label="Real EEG", color='blue')
                #axes[i, 0].plot(filtered_sim[band][ch], label="Real EEG", color='green')
                axes[i, 0].set_ylabel(band)
                axes[i, 0].legend(loc="upper right")

                # Simulated EEG
                axes[i, 1].plot(time_axis,filtered_sim[band][ch], label="Simulated EEG", color='green')
                axes[i, 1].legend(loc="upper right")
                
                f_real, Pxx_real = welch(filtered_real[band][ch], fs=fs, nperseg=fs*2)
                f_sim, Pxx_sim = welch(filtered_sim[band][ch], fs=fs, nperseg=fs*2)
                axes[i, 2].semilogy(f_real, Pxx_real, label='Real', color='blue')
                axes[i, 2].semilogy(f_sim, Pxx_sim, label='Simulated', color='green')
                axes[i, 2].set_xlim(0, 50)
                axes[i, 2].set_title("PSD Comparison")
                axes[i, 2].legend()
                axes[i, 2].grid(True)

            axes[-1, 0].set_xlabel("time in sec")
            axes[-1, 1].set_xlabel("time in sec")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig('neural_mass_'+str(ch)+'_'+str(time_dur)+".eps",format='eps')
            plt.show()

        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break

interactive_band_plot(filtered_real, filtered_sim, bands)
