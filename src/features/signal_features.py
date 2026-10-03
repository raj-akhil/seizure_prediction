"""Deterministic features. No fitting across windows, subjects, or labels."""
import numpy as np
from scipy.signal import welch, butter, sosfilt, find_peaks
from scipy.ndimage import uniform_filter1d


def divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan), where=np.asarray(b)>0)


def channel_features(x, fs, native_fs, families, bands, segment_sec):
    """x: windows x channels x samples. Return named windows x channels arrays."""
    x = np.asarray(x, dtype=np.float64)
    result = {}
    variance = np.var(x,axis=-1)
    centered = x-np.mean(x,axis=-1,keepdims=True)
    derivative = np.diff(x,axis=-1)
    if 'time' in families:
        std = np.sqrt(variance)
        result.update(mean=np.mean(x,axis=-1), std=std, rms=np.sqrt(np.mean(x*x,axis=-1)),
            peak_to_peak=np.ptp(x,axis=-1), mean_abs=np.mean(np.abs(x),axis=-1),
            line_length_per_sec=np.sum(np.abs(derivative),axis=-1)/(x.shape[-1]/fs),
            zero_crossings_per_sec=np.sum(np.diff(np.signbit(centered),axis=-1),axis=-1)/(x.shape[-1]/fs),
            skewness=divide(np.mean(centered**3,axis=-1),std**3),
            excess_kurtosis=divide(np.mean(centered**4,axis=-1),variance**2)-3)
    if 'hjorth' in families:
        # Derivatives per sample: keep rate in provenance for comparisons.
        v1 = np.var(derivative,axis=-1)
        mobility = np.sqrt(divide(v1,variance))
        result.update(hjorth_activity=variance,hjorth_mobility=mobility,
            hjorth_complexity=divide(np.sqrt(divide(np.var(np.diff(derivative,axis=-1),axis=-1),v1)),mobility))
    if 'spectral' in families or 'entropy' in families:
        nperseg = min(x.shape[-1],max(4,round(segment_sec*fs)))
        f, psd = welch(x,fs=fs,nperseg=nperseg,axis=-1)
        keep = f<=min(fs,native_fs)/2+1e-9
        f, psd = f[keep], psd[...,keep]
        df = f[1]-f[0]
        total = psd.sum(axis=-1)*df
        weights = divide(psd,total[...,None]/df)
        if 'entropy' in families:
            result['spectral_entropy'] = -np.sum(np.where(weights>0,weights*np.log2(np.maximum(weights,1e-300)),0),axis=-1)/np.log2(len(f))
            result['spectral_entropy'][total<=0]=np.nan
        if 'spectral' in families:
            result.update(total_power=total,spectral_centroid_hz=np.sum(weights*f,axis=-1),
                dominant_frequency_hz=f[np.argmax(psd,axis=-1)],
                median_frequency_hz=f[np.argmax(np.cumsum(np.nan_to_num(weights),axis=-1)>=.5,axis=-1)])
            for name,(low,high) in bands.items():
                if high>min(native_fs,fs)/2+1e-8:
                    result[f'{name}_power']=np.full(total.shape,np.nan)
                else:
                    result[f'{name}_power']=psd[..., (f>=low)&(f<high)].sum(axis=-1)*df
                result[f'{name}_relative_power']=divide(result[f'{name}_power'],total)
    if 'muscle' in families:
        result['integrated_abs_emg']=np.sum(np.abs(x),axis=-1)/fs
        result['slope_sign_changes_per_sec']=np.sum(derivative[...,:-1]*derivative[...,1:]<0,axis=-1)/(x.shape[-1]/fs)
    if 'motion' in families:
        result['derivative_rms_per_sec']=np.sqrt(np.mean((derivative*fs)**2,axis=-1))
        result['dynamic_rms']=np.sqrt(np.mean(centered**2,axis=-1))
    return result


def cardiac_features(x, fs, cfg):
    """QRS-energy heuristic on observed history only; not a clinical beat annotation."""
    names=['heart_rate_bpm','rr_mean_ms','rr_sd_ms','rr_rmssd_ms','rr_pnn50',
           'rr_valid_fraction','detected_beats','hrv_available']
    result=dict.fromkeys(names,np.nan)
    result['hrv_available']=0.
    if not np.isfinite(x).all() or len(x)<fs*3 or np.ptp(x)<=0:
        return result
    z=sosfilt(butter(cfg['detector_order'],cfg['detector_band_hz'],btype='bandpass',fs=fs,output='sos'),x)
    energy=np.diff(z,prepend=z[0])**2
    # Trailing integrator implemented explicitly (no future samples at its timestamp).
    n=max(1,round(cfg['integration_sec']*fs))
    energy=np.convolve(energy,np.ones(n)/n,mode='full')[:len(z)]
    med=np.median(energy)
    threshold=med+cfg['threshold_mad']*np.median(np.abs(energy-med))
    peaks,_=find_peaks(energy,height=threshold,distance=max(1,round(cfg['refractory_sec']*fs)))
    # Refine each energy peak against observed, filtered ECG; trim detector startup.
    refined=[]
    for p in peaks:
        a,b=max(0,p-round(.15*fs)),min(len(z),p+round(.05*fs))
        if b>a:
            q=a+int(np.argmax(np.abs(z[a:b])))
            if q>=fs and (not refined or q-refined[-1]>=cfg['refractory_sec']*fs):
                refined.append(q)
    rr=np.diff(refined)/fs
    result['detected_beats']=float(len(refined))
    if not len(rr):
        return result
    mask=(rr>=cfg['rr_range_sec'][0])&(rr<=cfg['rr_range_sec'][1])
    result['rr_valid_fraction']=float(mask.mean())
    if mask.mean()<1-cfg['max_rejected_fraction'] or mask.sum()<2:
        return result
    result['rr_mean_ms']=float(np.mean(rr[mask])*1000)
    result['heart_rate_bpm']=float(60/np.mean(rr[mask]))
    if len(x)/fs>=cfg['min_hrv_sec'] and mask.sum()>=cfg['min_rr_count']:
        adjacent=np.diff(rr)[mask[:-1]&mask[1:]]
        if len(adjacent):
            result.update(rr_sd_ms=float(np.std(rr[mask],ddof=1)*1000),
                rr_rmssd_ms=float(np.sqrt(np.mean(adjacent**2))*1000),
                rr_pnn50=float(np.mean(np.abs(adjacent)>.05)),hrv_available=1.)
    return result
