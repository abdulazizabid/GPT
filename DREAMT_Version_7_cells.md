# DREAMT Version 7 — cell-by-cell code

Copy numbered code cells into separate Colab cells, in order. This is the same code as the notebook.

## Cell 1 — explanation

# DREAMT Version 7 — multimodal wearable student with actual PSG knowledge distillation

**Targets:** macro F1 ≥70% and accuracy around/above70%. These are measured goals, not promised scores. V6 reached53.91% macro F1 and21.97% REM F1 at67.91% accuracy after class offsets. Its manifest showed **no KD**. Raising macro F1 to70% needs much stronger REM discrimination; new sensors and reliable temporal context matter more than another threshold change.

This version uses the approved wearable modalities: **BVP, derived pulse intervals, ACC_X/Y/Z, EDA, temperature and HR**. PSG remains **teacher-only**. This changes the student input specification from V5/V6's IBI-only model; report that change in any comparison. Neither model receives labels, scoring masks, subject ID, clock time or ground-truth stage as an input. Both are offline bidirectional models, not causal real-time classifiers.

**Changes supported by your V6 results:** native64Hz wearables for all splits; labels separated from sensor context; N1↔N2↔N3 transitions retained as NREM; paired-stream alignment audits; three genuinely cross-fitted three-class PSG teachers; a supervised reference followed by two distilled students; validation-only ensemble selection; positive temperature calibration without class offsets; strict historical matched comparisons alongside new-cohort coverage.

Run top to bottom. Set paths below and choose a Colab GPU. Teacher training is real and can take several hours; compatible completed stages resume. Raw DREAMT CSVs are required when the new prepared arrays are absent. This notebook does not download or include the dataset.

## Cell 2 — code

```python
# Keep Colab's GPU-enabled PyTorch; install only small dependencies.
%pip -q install numpy pandas scipy scikit-learn tqdm matplotlib
```

## Cell 3 — code

```python
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('Set DATA and output paths for this non-Colab machine.')
```

## Cell 4 — code

```python
# Configuration. Keep OUT separate from Version-5 and Version-6.
DATA=Path('/content/drive/MyDrive/Dataset/dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.2.0')
DIR100,DIR64=DATA/'data_100Hz',DATA/'data_64Hz'
V5=Path('/content/drive/MyDrive/Dataset/Version-5')
V6=Path('/content/drive/MyDrive/Dataset/Version-6/v6_1')
OUT=Path('/content/drive/MyDrive/Dataset/Version-7')
LOCAL=Path('/content/dreamt_v7_local' if Path('/content').exists() else '/tmp/dreamt_v7_local')
RUN_TAG='v7_1'
SEED=42
STUDENT_MODE='multimodal'  # approved default; ibi_only is a separate explicit ablation
TEACHER_FOLDS=3
TEACHER_EPOCHS,TEACHER_MIN_EPOCHS,TEACHER_PATIENCE=30,8,8
BASELINE_EPOCHS,DISTILL_EPOCHS=45,30
STUDENT_MIN_EPOCHS,STUDENT_PATIENCE=12,12
DISTILL_SEEDS=[42,43]
LR_TEACHER,LR_BASELINE,LR_DISTILL=3e-4,3e-4,1e-4
WEIGHT_DECAY=1e-3
CORE,CONTEXT=32,32
SEQ=CORE+2*CONTEXT  # max48minutes; each32epoch core emits exactly once
BVP_FS,SLOW_FS,PSG_FS=32,4,50
MIN_STAGE_PURITY=1.0  # do not silently relax labels to improve scores
CLASS_WEIGHT_POWER=.35
KD_T,KD_WEIGHT,KD_WARMUP,KD_RAMP=2.,.20,3,8
REM_AUX_WEIGHT=.15  # REM-vs-NREM auxiliary loss; Wake is masked from this loss
EFFECTIVE_BATCH_TEACHER,EFFECTIVE_BATCH_STUDENT=4,16
BATCH=0  # automatic microbatch; lower to1/2 after an OOM and rerun to resume
NUM_WORKERS=2  # set0 for limited RAM or multiprocessing problems
SAVE_UPDATES=25
AUGMENT_NOISE=.01
CHUNK_ROWS=300000
LOCAL_CACHE=True
ID_PATTERN=r'(?i)(S\d+)'
COLUMN_OVERRIDE,LABEL_OVERRIDE,VALID_SECONDS={},{},{}
WEARABLE_ONLY_SUBJECTS={'S097'}
TARGET_MACRO_F1,TARGET_ACCURACY=.70,.70
SELECTION_ACCURACY_FLOOR=.68
CLASS_NAMES=['Wake','NREM','REM']
```

## Cell 5 — code

```python
import os,re,json,time,random,hashlib,shutil,gc,warnings,zipfile,pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from scipy.signal import butter,sosfiltfilt,find_peaks,resample_poly
from scipy.special import softmax
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import (accuracy_score,f1_score,classification_report,confusion_matrix,
    cohen_kappa_score,log_loss,average_precision_score,balanced_accuracy_score)
from functools import lru_cache
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

assert CORE>0 and CONTEXT>=0 and SEQ==CORE+2*CONTEXT
assert 0<MIN_STAGE_PURITY<=1 and STUDENT_MODE in {'multimodal','ibi_only'}
assert DISTILL_EPOCHS>KD_WARMUP and DISTILL_SEEDS
assert TEACHER_FOLDS>=2 and SAVE_UPDATES>0
assert BVP_FS==32 and SLOW_FS==4 and PSG_FS==50,'Model shapes are pinned to32/4/50Hz; use a new version for rate changes.'
OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');AMP=DEVICE.type=='cuda'
AMP_DTYPE=torch.bfloat16 if AMP and torch.cuda.is_bf16_supported() else torch.float16
torch.set_num_threads(min(8,os.cpu_count() or 2))
if AMP:
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if AMP:torch.cuda.manual_seed_all(seed)
seed_all(SEED)
print('PyTorch:',torch.__version__,'| device:',DEVICE,'| AMP:',AMP)
print('CPU workers:',NUM_WORKERS,'| free local disk GiB:',round(shutil.disk_usage(LOCAL).free/2**30,1))
if AMP:print('GPU:',torch.cuda.get_device_name(0),'| free GiB:',round(torch.cuda.mem_get_info()[0]/2**30,1))
else:warnings.warn('CPU fallback is functional; full teacher plus student training is slow. A GPU runtime is strongly recommended.')

def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x
def digest(x):return hashlib.sha256(json.dumps(jsonable(x),sort_keys=True).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save_json(x,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(jsonable(x),indent=2));os.replace(tmp,path)
def save_csv(x,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    x.to_csv(tmp,index=False);os.replace(tmp,path)
def save_torch(x,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(x,tmp);os.replace(tmp,path)
def load_own_checkpoint(path):
    # Only deserialize checkpoints produced in your own trusted output directory.
    return torch.load(path,map_location='cpu',weights_only=False)
PIPELINE_REV='21661c4a06c2122b180258ab2621e9480eb17040e7ee6203b3ee5469b786874b'
PREP_REV='bbb3826ccf0bfdb938379f289b03604e2bcaa91ae85f2df8e23ba86ee35f5676'
STORE,METADATA,QSTORE={},{},{}
KD_ID=None
TEACHER_RELIABILITY=np.ones(3,np.float32)
```

## Cell 6 — explanation

## Fixed subjects and scoring policy

The original80/20 subject lists and seeded64/16 development split are unchanged. Eight validation people select checkpoints/ensembles; the other eight fit temperature only. All64Hz training signals are from training participants; validation and test participants remain excluded from gradients and normalization. Teacher inner validation also comes entirely from the64 training people. This gives **three-class OOF teacher targets**, not five-class probabilities incorrectly re-tempered after merging.

Every complete wearable epoch stays in its true timeline. A valid label determines the training/scoring mask; it does not determine whether neighboring signals can supply context. Unknown/preparation/cross-class mixed labels are not scored. NREM substage transitions are NREM when the collapsed three-class label is pure. True missing timestamps still break runs; missing time is never compressed. No genuine Wake is removed using a guessed preparation boundary.

## Cell 7 — code

```python
# Fixed subject lists transcribed from your image.
test_subjects = [
    'S006', 'S008', 'S016', 'S019', 'S021',
    'S024', 'S028', 'S031', 'S039', 'S053',
    'S062', 'S074', 'S076', 'S077', 'S078',
    'S084', 'S088', 'S093', 'S096', 'S102',
]
train_subjects = [
    'S002', 'S003', 'S004', 'S005', 'S007',
    'S009', 'S010', 'S011', 'S012', 'S013',
    'S014', 'S015', 'S017', 'S018', 'S020',
    'S022', 'S023', 'S025', 'S026', 'S027',
    'S029', 'S030', 'S032', 'S033', 'S034',
    'S035', 'S036', 'S037', 'S038', 'S040',
    'S042', 'S043', 'S044', 'S045', 'S046',
    'S047', 'S048', 'S049', 'S050', 'S051',
    'S052', 'S054', 'S055', 'S056', 'S057',
    'S058', 'S059', 'S061', 'S063', 'S064',
    'S065', 'S066', 'S067', 'S068', 'S069',
    'S070', 'S071', 'S072', 'S073', 'S075',
    'S079', 'S080', 'S081', 'S082', 'S083',
    'S085', 'S086', 'S087', 'S089', 'S090',
    'S091', 'S092', 'S094', 'S095', 'S097',
    'S098', 'S099', 'S100', 'S101', 'S103',
]
```

## Cell 8 — code

```python
assert len(set(train_subjects))==80 and len(set(test_subjects))==20
assert not set(train_subjects)&set(test_subjects)
train_ids,valid_ids=train_test_split(train_subjects,test_size=16,random_state=SEED)
select_ids,calibrate_ids=train_test_split(sorted(valid_ids),test_size=8,random_state=SEED+100)
SPLIT=dict(train=sorted(train_ids),select=sorted(select_ids),calibrate=sorted(calibrate_ids),valid=sorted(valid_ids),test=sorted(test_subjects))
RUNROOT=OUT/RUN_TAG;RUNROOT.mkdir(parents=True,exist_ok=True)
print({k:len(v) for k,v in SPLIT.items()})
print('Model-selection people:',SPLIT['select']);print('Temperature-calibration people:',SPLIT['calibrate'])
```

## Cell 9 — code

```python
# V7 native-rate preprocessing. No raw files are read until the API is called.
# Labels decide which epochs can be scored; they never cut valid sensor context.
PSG_CHANNELS = ['C4-M1', 'F4-M1', 'O2-M1', 'E1', 'E2', 'CHIN', 'ECG']
SLOW_CHANNELS = ['IBI', 'ACC_X', 'ACC_Y', 'ACC_Z', 'EDA', 'TEMP', 'HR']
FEATURE_NAMES = ['bvp_log_scale', 'bvp_skew', 'bvp_kurtosis', 'bvp_log_power_0.5_1.5',
                 'bvp_log_power_1.5_3', 'bvp_log_power_3_5', 'bvp_log_power_5_8',
                 'ibi_mean', 'ibi_std', 'ibi_rmssd', 'ibi_median', 'ibi_iqr', 'ibi_pnn50']
FEATURE_NAMES += [f'{c.lower()}_{s}' for c in SLOW_CHANNELS[1:]
                  for s in ['mean', 'std', 'median', 'iqr', 'slope']]
FEATURE_NAMES += ['acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_change']
FEATURE_NAMES += [f'available_{c.lower()}' for c in ['BVP'] + SLOW_CHANNELS]


def prep_clean_name(value):
    return re.sub(r'[^a-z0-9]', '', re.sub(r'\[.*?\]|\(.*?\)', '', str(value)).lower())


def discover_files():
    """Discover headers only later; never silently choose duplicate subject CSVs."""
    found = []
    for directory in [Path(DIR100), Path(DIR64)]:
        if not directory.is_dir():
            raise FileNotFoundError(f'Missing data directory: {directory}. Set DIR100 and DIR64 in Configuration.')
        mapping = {}
        for path in sorted(directory.rglob('*.csv')):
            match = re.search(ID_PATTERN, path.name, flags=re.I)
            if match is None:
                continue
            sid = str(match.group(0)).upper()
            if sid in mapping:
                raise ValueError(f'Multiple CSV files for {sid} in {directory}: {mapping[sid]} and {path}. Select one source directory.')
            mapping[sid] = path
        found.append(mapping)
    if not found[1]:
        raise FileNotFoundError(f'No native-64 Hz subject CSV files found under {DIR64}.')
    return found[0], found[1]


def prep_source_signature(path):
    stat = Path(path).stat()
    # Source identity uses resolved path, exact byte count, mtime_ns and CSV header.
    # Never borrow a V5/V6 cache: native-rate modalities and masking changed.
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                header=list(pd.read_csv(path, nrows=0).columns))


def prep_atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def prep_atomic_npz(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def prep_mapping(path, domain):
    header = list(pd.read_csv(path, nrows=0).columns)
    aliases = {'TIMESTAMP': ['timestamp', 'time', 'time_seconds'],
               'Sleep_Stage': ['Sleep_Stage', 'sleep stage', 'stage'],
               'BVP': ['BVP', 'blood volume pulse'],
               'ACC_X': ['ACC_X', 'ACCX', 'ACC X', 'accelerometer_x', 'accelerometer x'],
               'ACC_Y': ['ACC_Y', 'ACCY', 'ACC Y', 'accelerometer_y', 'accelerometer y'],
               'ACC_Z': ['ACC_Z', 'ACCZ', 'ACC Z', 'accelerometer_z', 'accelerometer z'],
               'EDA': ['EDA', 'GSR', 'electrodermal activity'],
               'TEMP': ['TEMP', 'temperature', 'skin temperature'],
               'HR': ['HR', 'heart rate', 'heartrate']}
    required = ['TIMESTAMP', 'Sleep_Stage'] + (PSG_CHANNELS if domain == '100' else ['BVP'])
    if domain == '64' and STUDENT_MODE == 'multimodal':
        required += SLOW_CHANNELS[1:]
    overrides = dict(COLUMN_OVERRIDE)
    if isinstance(COLUMN_OVERRIDE.get(domain), dict):
        overrides.update(COLUMN_OVERRIDE[domain])
    result = {}
    for name in required:
        candidates = aliases.get(name, [name, 'EEG ' + name])
        override = overrides.get(name)
        matches = ([h for h in header if h == override] if isinstance(override, str)
                   else [h for h in header if prep_clean_name(h) in {prep_clean_name(v) for v in candidates}])
        if len(matches) != 1:
            raise ValueError(f'{path.name}: cannot uniquely resolve {name}. Set COLUMN_OVERRIDE[{name!r}] to its exact header. Available: {header}')
        result[name] = matches[0]
    return result


def prep_encode3(series):
    # Collapse N1/N2/N3 BEFORE epoch purity: valid NREM transitions remain NREM.
    mapping = {'w': 0, 'wake': 0, 'n1': 1, 'nrem1': 1, 'n2': 1, 'nrem2': 1,
               'n3': 1, 'nrem3': 1, 'nrem': 1, 'r': 2, 'rem': 2, 'remsleep': 2,
               'p': -100, 'preparation': -100, 'missing': -100, 'nan': -100, '': -100,
               'unknown': -100, 'unscored': -100, 'movement': -100, 'artifact': -100}
    # Overrides retain V5's five-class convention: Wake=0,N1=1,N2=2,N3=3,REM=4.
    for label, value in LABEL_OVERRIDE.items():
        value = int(value)
        if value not in [-100, -1, 0, 1, 2, 3, 4]:
            raise ValueError('LABEL_OVERRIDE values use five-class codes 0,1,2,3,4 or -1/-100.')
        mapping[prep_clean_name(label)] = -100 if value < 0 else [0, 1, 1, 1, 2][value]
    names = series.fillna('').map(prep_clean_name)
    # Unknown annotations are explicitly unscored, preserving their sensor context.
    encoded = names.map(mapping).fillna(-100).to_numpy(dtype=np.int64)
    return encoded, names


def prep_epoch_label(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0 or np.any(labels < 0):
        return -100
    counts = np.bincount(labels, minlength=3)
    winner = int(np.argmax(counts))
    return winner if counts[winner] / len(labels) >= float(MIN_STAGE_PURITY) else -100


def prep_epoch_blocks(path, mapping, sid, fs):
    """Stream complete 30-second groups; epoch IDs preserve true missing time."""
    carry = pd.DataFrame()
    previous = -np.inf
    inverse = {value: key for key, value in mapping.items()}
    for frame in pd.read_csv(path, usecols=list(mapping.values()), chunksize=int(CHUNK_ROWS)):
        frame = frame.rename(columns=inverse)
        label3, raw_names = prep_encode3(frame['Sleep_Stage'])
        frame['Sleep_Stage'] = label3
        frame['_label_name'] = raw_names
        for column in mapping:
            if column != 'Sleep_Stage':
                frame[column] = pd.to_numeric(frame[column], errors='coerce')
        times = frame['TIMESTAMP'].to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or (len(times) and times[0] <= previous):
            raise ValueError(f'{sid}: timestamps must be finite, strictly increasing seconds. Do not silently shift timestamps.')
        if len(times):
            previous = times[-1]
        frame['_epoch'] = np.floor((times + 1e-6) / 30).astype(np.int64)
        frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        last = frame['_epoch'].iloc[-1]
        ready, carry = frame[frame['_epoch'] != last], frame[frame['_epoch'] == last]
        if not ready.empty:
            yield ready
    if not carry.empty:
        yield carry


def prep_repair(values, fs, allow_flat=False, min_coverage=0.99):
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if not len(values) or not good[0] or not good[-1] or good.mean() < min_coverage:
        return None
    missing = np.flatnonzero(~good)
    if len(missing):
        runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
        if max(map(len, runs)) > max(1, int(0.25 * fs)):
            return None
        values = np.interp(np.arange(len(values)), np.flatnonzero(good), values[good])
    if not allow_flat and np.std(values) <= 1e-8:
        return None
    return values


@lru_cache(maxsize=32)
def prep_band(fs, low, high):
    return butter(3, [low, high], btype='bandpass', fs=fs, output='sos')


def prep_make_ibi(bvp, fs=64):
    """V5-compatible pulse intervals: interpolate at 25 Hz, then polyphase to 4 Hz."""
    clean = prep_repair(bvp, fs)
    if clean is None:
        return None
    filtered = sosfiltfilt(prep_band(fs, 0.5, 8.0), clean)
    peaks, _ = find_peaks(filtered, distance=max(1, int(0.30 * fs)), prominence=0.2 * np.std(filtered))
    if len(peaks) < 12:
        return None
    times, intervals = peaks / fs, np.diff(peaks / fs)
    good = (intervals >= 0.30) & (intervals <= 2.0)
    t, intervals = times[1:][good], intervals[good]
    if len(t) < 10 or t[0] > 3 or t[-1] < 27 or np.max(np.diff(t)) > 3:
        return None
    interpolated = np.interp(np.arange(750) / 25, t, intervals)
    return resample_poly(interpolated, int(SLOW_FS), 25).astype(np.float32), intervals.astype(np.float32)


def prep_features(bvp, ibi, slow, avail):
    out = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if bvp is not None:
        centered = bvp - np.median(bvp)
        scale = max(float(np.std(centered)), 1e-6)
        z = np.clip(centered / scale, -20, 20)
        out[:3] = [np.log1p(scale), float(np.mean(z ** 3)), float(np.mean(z ** 4)) - 3]
        power = np.abs(np.fft.rfft(centered)) ** 2 / max(1, len(centered))
        freq = np.fft.rfftfreq(len(centered), 1 / 64)
        for k, (lo, hi) in enumerate([(0.5, 1.5), (1.5, 3), (3, 5), (5, 8)]):
            out[3 + k] = np.log1p(power[(freq >= lo) & (freq < hi)].mean())
    if ibi is not None:
        d = np.diff(ibi)
        out[7:13] = [np.mean(ibi), np.std(ibi), np.sqrt(np.mean(d ** 2)) if len(d) else 0,
                     np.median(ibi), np.percentile(ibi, 75) - np.percentile(ibi, 25),
                     np.mean(np.abs(d) > 0.05) if len(d) else 0]
    for i in range(1, 7):
        if avail[i + 1]:
            x = slow[i]
            slope = float(np.polyfit(np.arange(len(x)) / SLOW_FS, x, 1)[0])
            out[13 + (i - 1) * 5:18 + (i - 1) * 5] = [x.mean(), x.std(), np.median(x),
                                                                         np.percentile(x, 75) - np.percentile(x, 25), slope]
    if np.all(avail[2:5] > 0):
        magnitude = np.sqrt(np.sum(slow[1:4] ** 2, axis=0))
        out[43:46] = [magnitude.mean(), magnitude.std(), np.mean(np.abs(np.diff(magnitude)))]
    out[46:54] = avail
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
```

## Cell 10 — code

```python
def prep_wearable_epoch(group, sid):
    times = group['TIMESTAMP'].to_numpy(float)
    if len(group) != 30 * 64:
        return None, 'incomplete'
    if not np.allclose(np.diff(times), 1 / 64, atol=0.0015, rtol=0):
        return None, 'timestamp_gap'
    epoch = int(group['_epoch'].iloc[0])
    y = prep_epoch_label(group['Sleep_Stage'].to_numpy(np.int64))
    lo, hi = VALID_SECONDS.get(sid, (float('-inf'), float('inf')))
    if epoch * 30 < lo or (epoch + 1) * 30 > hi:
        y = -100
    fast = np.zeros((2, 30 * int(BVP_FS)), dtype=np.float32)
    slow = np.zeros((7, 30 * int(SLOW_FS)), dtype=np.float32)
    avail = np.zeros(8, dtype=np.float32)
    clean = prep_repair(group['BVP'].to_numpy(float), 64)
    intervals = None
    if clean is not None:
        filtered = sosfiltfilt(prep_band(64, 0.5, 8), clean)
        down = resample_poly(filtered, int(BVP_FS), 64).astype(np.float32)
        scale = max(float(np.percentile(down, 75) - np.percentile(down, 25)), 1e-5)
        normalized = np.clip((down - np.median(down)) / scale, -12, 12)
        fast[0] = normalized
        fast[1] = np.clip(np.gradient(normalized) * 4.0, -12, 12)
        avail[0] = 1
        ibi = prep_make_ibi(clean)
        if ibi is not None:
            slow[0], intervals = ibi
            avail[1] = 1
    if STUDENT_MODE == 'multimodal':
        for index, column in enumerate(SLOW_CHANNELS[1:], start=1):
            values = prep_repair(group[column].to_numpy(float), 64, allow_flat=True, min_coverage=0.95)
            if values is not None:
                slow[index] = resample_poly(values, int(SLOW_FS), 64).astype(np.float32)
                avail[index + 1] = 1
    if STUDENT_MODE == 'ibi_only':
        fast[:] = 0
        avail[0] = 0
    usable = bool(avail[1] > 0) if STUDENT_MODE == 'ibi_only' else bool(avail[0] > 0 or np.any(avail[2:] > 0))
    row = dict(e=np.int64(epoch), y=np.int64(y), usable=np.bool_(usable), psg_valid=np.bool_(False),
               fast=fast, slow=slow, avail=avail, features=prep_features(None if STUDENT_MODE == 'ibi_only' else clean, intervals, slow, avail))
    return row, 'kept'


def prep_run_summary(epoch_ids):
    if not len(epoch_ids):
        return dict(epochs=0, runs=0, median_seconds=0, fraction_under_32min=0)
    runs = np.split(np.asarray(epoch_ids), np.flatnonzero(np.diff(epoch_ids) != 1) + 1)
    lengths = np.array([len(run) for run in runs])
    return dict(epochs=int(lengths.sum()), runs=int(len(lengths)), median_seconds=float(np.median(lengths) * 30),
                fraction_under_32min=float(lengths[lengths < 64].sum() / lengths.sum()))


def prep_local_array(source):
    """Mirror immutable prepared arrays to local SSD for efficient random reads."""
    import shutil
    source = Path(source)
    local_root = Path(LOCAL) / 'prepared_arrays'
    if not globals().get('LOCAL_CACHE', True):
        return source
    try:
        if source.resolve().is_relative_to(Path(LOCAL).resolve()):
            return source
    except AttributeError:  # Python <3.9 fallback (current Colab is newer).
        pass
    stat = source.stat()
    signature = dict(path=str(source.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    target = local_root / digest(signature)[:20] / source.name
    if target.exists() and target.stat().st_size == stat.st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < stat.st_size + 512 * 1024 ** 2:
        warnings.warn(f'Not enough local SSD space to mirror {source.name}; reading its persisted mmap directly. Set LOCAL to a larger disk or free obsolete local caches.')
        return source
    temporary = target.with_name(target.name + '.tmp')
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != stat.st_size:
            raise OSError(f'Incomplete local mirror of {source}')
        os.replace(temporary, target)
    except OSError as error:
        warnings.warn(f'Local cache copy failed for {source.name}; reading persisted mmap: {error}')
        if temporary.exists():
            temporary.unlink()
        return source
    return target
```

## Cell 11 — code

```python
def prep_open_cache(folder, meta):
    data = {}
    for key, record in meta['arrays'].items():
        source = Path(folder) / f'{key}.npy'
        path = prep_local_array(source)
        try:
            array = np.load(path, mmap_mode='r', allow_pickle=False)
        except (OSError, ValueError, EOFError):
            if path == source:
                raise
            prep_quarantine(path)
            array = np.load(prep_local_array(source), mmap_mode='r', allow_pickle=False)
        if list(array.shape) != record['shape'] or str(array.dtype) != record['dtype']:
            raise ValueError(f'Invalid saved array shape/dtype: {path}')
        data[key] = array
    return data


def prep_quarantine(path):
    path = Path(path)
    if path.exists():
        os.replace(path, path.with_name(path.name + f'.invalid_{time.time_ns()}'))


def prep_consolidate(blocks, folder):
    # Inspect headers one block at a time; avoid a list of all subject PSG arrays.
    keys = None
    total = 0
    shapes, dtypes = {}, {}
    for path in blocks:
        with np.load(path, allow_pickle=False) as archive:
            current = [key for key in archive.files if key != '_qc']
            if current:
                if keys is None:
                    keys = current
                    shapes = {key: archive[key].shape[1:] for key in keys}
                    dtypes = {key: archive[key].dtype for key in keys}
                total += len(archive['e'])
    if total == 0:
        raise ValueError(f'No complete native-64 Hz epochs in {folder}. Check timestamps and source schema.')
    maps, temporary = {}, {}
    for key in keys:
        temporary[key] = Path(folder) / f'{key}.npy.tmp'
        maps[key] = np.lib.format.open_memmap(temporary[key], mode='w+', dtype=dtypes[key], shape=(total,) + shapes[key])
    offset = 0
    for path in blocks:
        with np.load(path, allow_pickle=False) as archive:
            if 'e' not in archive:
                continue
            n = len(archive['e'])
            for key in keys:
                maps[key][offset:offset+n] = archive[key]
            offset += n
    for key in keys:
        maps[key].flush()
    records = {key: dict(shape=list(maps[key].shape), dtype=str(maps[key].dtype)) for key in keys}
    maps.clear()
    gc.collect()
    for key in keys:
        os.replace(temporary[key], Path(folder) / f'{key}.npy')
    return records


def prep_prepare_wearable(sid):
    if sid not in FILES64:
        raise FileNotFoundError(f'{sid}: native-64 Hz wearable data is required for every split; it was not found in DIR64.')
    source = prep_source_signature(FILES64[sid])
    settings = dict(revision=PREP_REV, student_mode=STUDENT_MODE, bvp_fs=BVP_FS, slow_fs=SLOW_FS,
                    min_stage_purity=MIN_STAGE_PURITY, columns=COLUMN_OVERRIDE, labels=LABEL_OVERRIDE,
                    valid_seconds=VALID_SECONDS.get(sid), chunk_rows=CHUNK_ROWS, feature_names=FEATURE_NAMES,
                    code_policy='native64_context_v7_4', exclude_teacher_subjects=sorted(WEARABLE_ONLY_SUBJECTS))
    wear_key = digest(dict(source=source, settings=settings))[:20]
    folder = Path(OUT) / 'prepared' / sid / wear_key
    manifest = folder / 'manifest.json'
    if manifest.exists():
        try:
            meta = json.loads(manifest.read_text())
            data = prep_open_cache(folder, meta)
            if not np.all(np.diff(data['e']) > 0):
                raise ValueError('Saved epoch IDs are not strictly increasing.')
            return data, meta
        except (OSError, ValueError, KeyError, EOFError) as error:
            warnings.warn(f'{sid}: own V7 manifest invalid; rebuilding: {error}')
            prep_quarantine(manifest)
    folder.mkdir(parents=True, exist_ok=True)
    block_folder = folder / 'blocks'
    block_folder.mkdir(exist_ok=True)
    mapping = prep_mapping(FILES64[sid], '64')
    blocks, totals, label_counts, label_epoch_counts = [], {}, {}, {}
    for block_index, frame in enumerate(prep_epoch_blocks(FILES64[sid], mapping, sid, 64)):
        path = block_folder / f'{block_index:05d}.npz'
        valid = False
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    qc = json.loads(str(saved['_qc']))
                    if 'e' in saved:
                        valid = (saved['fast'].shape == (len(saved['e']), 2, 30 * int(BVP_FS)) and
                                 saved['slow'].shape == (len(saved['e']), 7, 30 * int(SLOW_FS)) and
                                 saved['features'].shape == (len(saved['e']), len(FEATURE_NAMES)))
                    else:
                        valid = bool(qc.get('empty', False))
            except (OSError, ValueError, KeyError, EOFError):
                valid = False
            if not valid:
                prep_quarantine(path)
        if not valid:
            rows, qc = [], {'reasons': {}, 'label_epoch_status': {}, 'source_label_rows': {str(k): int(v) for k, v in frame['_label_name'].value_counts().items()}}
            for _, group in frame.groupby('_epoch', sort=True):
                row, reason = prep_wearable_epoch(group, sid)
                qc['reasons'][reason] = qc['reasons'].get(reason, 0) + 1
                if row is not None:
                    rows.append(row)
                    stage3 = group['Sleep_Stage'].to_numpy(np.int64)
                    names = set(group['_label_name'])
                    if int(row['y']) >= 0:
                        status = ['valid_wake', 'valid_nrem', 'valid_rem'][int(row['y'])]
                        if int(row['y']) == 1 and len(names) > 1:
                            qc['label_epoch_status']['retained_within_nrem_transition'] = qc['label_epoch_status'].get('retained_within_nrem_transition', 0) + 1
                    elif np.any(stage3 < 0):
                        status = 'unscored_preparation' if names.intersection({'p', 'preparation'}) else 'unscored_unknown_or_missing'
                    elif len(np.unique(stage3)) > 1:
                        status = 'unscored_mixed_three_class'
                    else:
                        status = 'unscored_outside_verified_interval'
                    qc['label_epoch_status'][status] = qc['label_epoch_status'].get(status, 0) + 1
            payload = {key: np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}
            qc['empty'] = not bool(rows)
            payload['_qc'] = np.array(json.dumps(qc))
            prep_atomic_npz(payload, path)
        for reason, count in qc['reasons'].items():
            totals[reason] = totals.get(reason, 0) + int(count)
        for label, count in qc['source_label_rows'].items():
            label_counts[label] = label_counts.get(label, 0) + int(count)
        for status, count in qc.get('label_epoch_status', {}).items():
            label_epoch_counts[status] = label_epoch_counts.get(status, 0) + int(count)
        blocks.append(path)
    records = prep_consolidate(blocks, folder)
    meta = dict(sid=sid, wear_key=wear_key, source64=source, settings=settings, arrays=records,
                qc=dict(epoch_reasons=totals, source_label_rows=label_counts, label_epoch_status=label_epoch_counts), feature_names=FEATURE_NAMES)
    data = prep_open_cache(folder, meta)
    scored = (data['y'] >= 0) & data['usable']
    meta['qc'].update(dict(complete_epochs=int(len(data['e'])), scorable_epochs=int(scored.sum()),
                          unscored_context_epochs=int((~scored).sum()),
                          class_counts={str(i): int(np.sum(data['y'][scored] == i)) for i in range(3)},
                          channel_coverage={name: float(data['avail'][:, i].mean()) for i, name in enumerate(['BVP'] + SLOW_CHANNELS)}))
    meta['run_lengths'] = prep_run_summary(data['e'])
    meta['scored_run_lengths'] = prep_run_summary(data['e'][scored])
    prep_atomic_json(meta, manifest)  # Completion marker is always written last.
    return data, meta
```

## Cell 12 — code

```python
def prep_add_psg(sid, data, base_meta):
    # Native wearable arrays and cache key remain identical when PSG is attached.
    if sid in WEARABLE_ONLY_SUBJECTS or sid not in FILES100:
        result = dict(data)
        result['teacher_available'] = np.zeros(len(data['e']), dtype=bool)
        result['psg_valid'] = result['teacher_available']
        meta = dict(base_meta)
        meta['psg_key'] = None
        meta['psg_note'] = 'Subject is wearable-only or paired PSG CSV is absent.'
        return result, meta
    source100 = prep_source_signature(FILES100[sid])
    psg_key = digest(dict(wear_key=base_meta['wear_key'], source100=source100, revision=PREP_REV,
                         fs=PSG_FS, columns=COLUMN_OVERRIDE, labels=LABEL_OVERRIDE, purity=MIN_STAGE_PURITY,
                         psg_policy='raw_amplitude_context_v7_3'))[:20]
    folder = Path(OUT) / 'prepared' / sid / base_meta['wear_key'] / 'psg' / psg_key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        try:
            pmeta = json.loads(manifest.read_text())
            attached = prep_open_cache(folder, pmeta)
            if len(attached['psg']) != len(data['e']):
                raise ValueError('PSG/native64 epoch counts differ.')
            result, meta = dict(data), dict(base_meta)
            result.update(attached)
            meta.update(psg_key=psg_key, source100=source100, psg_qc=pmeta['qc'])
            return result, meta
        except (OSError, ValueError, KeyError, EOFError) as error:
            warnings.warn(f'{sid}: own PSG manifest invalid; rebuilding: {error}')
            prep_quarantine(manifest)
    n = len(data['e'])
    # Save the PSG tensor directly to disk: at most a single epoch is filtered in RAM.
    psg_tmp = folder / 'psg.npy.tmp'
    psg = np.lib.format.open_memmap(psg_tmp, mode='w+', dtype=np.float32, shape=(n, 7, 30 * int(PSG_FS)))
    psg[:] = 0
    psg_valid = np.zeros(n, dtype=bool)
    teacher_available = np.zeros(n, dtype=bool)
    rows = {int(epoch): i for i, epoch in enumerate(data['e'])}
    mapping = prep_mapping(FILES100[sid], '100')
    qc = dict(matched_scorable=0, stage3_agree=0, complete_psg=0, valid_psg=0, teacher_epochs=0,
              incomplete=0, timestamp_gap=0, label_mismatch=0, invalid_psg=0)
    for frame in prep_epoch_blocks(FILES100[sid], mapping, sid, 100):
        for epoch, group in frame.groupby('_epoch', sort=True):
            index = rows.get(int(epoch))
            if index is None:
                continue
            if len(group) != 3000:
                qc['incomplete'] += 1
                continue
            if not np.allclose(np.diff(group['TIMESTAMP'].to_numpy(float)), 0.01, atol=0.0015, rtol=0):
                qc['timestamp_gap'] += 1
                continue
            qc['complete_psg'] += 1
            y100 = prep_epoch_label(group['Sleep_Stage'].to_numpy(np.int64))
            y64 = int(data['y'][index])
            if y100 >= 0 and y64 >= 0:
                qc['matched_scorable'] += 1
                qc['stage3_agree'] += int(y100 == y64)
                qc['label_mismatch'] += int(y100 != y64)
            channels = [prep_repair(group[channel].to_numpy(float), 100) for channel in PSG_CHANNELS]
            if any(channel is None for channel in channels):
                qc['invalid_psg'] += 1
                continue
            bands = [(0.3, 35)] * 3 + [(0.3, 15)] * 2 + [(10, 45), (0.5, 40)]
            x = np.stack([resample_poly(sosfiltfilt(prep_band(100, lo, hi), channel), int(PSG_FS), 100)
                          for channel, (lo, hi) in zip(channels, bands)]).astype(np.float32)
            # Preserve physiological amplitude, especially chin EMG for REM.
            # Training alone fits the global channel scales; no per-epoch division.
            x -= np.median(x, axis=1, keepdims=True)
            psg[index] = x
            psg_valid[index] = True
            # Signal context is independent of annotation and student usability.
            teacher_available[index] = bool(y100 >= 0 and y64 >= 0 and y100 == y64 and data['usable'][index])
            qc['valid_psg'] += 1
            qc['teacher_epochs'] += int(teacher_available[index])
    qc['stage3_agreement'] = float(qc['stage3_agree'] / max(1, qc['matched_scorable']))
    if qc['matched_scorable'] >= 20 and qc['stage3_agreement'] < 0.90:
        del psg
        gc.collect()
        prep_quarantine(psg_tmp)
        raise ValueError(f"{sid}: paired 100/64 Hz three-class label agreement is {qc['stage3_agreement']:.1%} over {qc['matched_scorable']} epochs. Check source version, timestamp origin and annotation alignment. No shift is applied automatically.")
    psg.flush()
    del psg
    os.replace(psg_tmp, folder / 'psg.npy')
    for name, array in [('psg_valid', psg_valid), ('teacher_available', teacher_available)]:
        temporary = folder / (name + '.npy.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, folder / (name + '.npy'))
    pmeta = dict(psg_key=psg_key, source100=source100, qc=qc,
                 arrays=dict(psg=dict(shape=[n, 7, 30 * int(PSG_FS)], dtype='float32'),
                             psg_valid=dict(shape=[n], dtype='bool'), teacher_available=dict(shape=[n], dtype='bool')))
    prep_atomic_json(pmeta, manifest)
    result, meta = dict(data), dict(base_meta)
    result.update(prep_open_cache(folder, pmeta))
    meta.update(psg_key=psg_key, source100=source100, psg_qc=qc)
    return result, meta


def prepare_subject(sid, include_psg=False):
    """Return mmap arrays + manifest; student-only reads never load PSG arrays."""
    data, meta = prep_prepare_wearable(str(sid))
    if include_psg:
        return prep_add_psg(str(sid), data, meta)
    return data, meta
```

## Cell 13 — code

```python
FILES100,FILES64=discover_files()
required=set(train_subjects+test_subjects)
assert required<=set(FILES64),f'Native64Hz files missing: {sorted(required-set(FILES64))}'
assert set(SPLIT['train'])-WEARABLE_ONLY_SUBJECTS<=set(FILES100),'Training PSG source files are missing.'
for sid in tqdm(SPLIT['train']+SPLIT['valid'],desc='Native64Hz development preprocessing'):
    STORE[sid],METADATA[sid]=prepare_subject(sid,include_psg=False)
support=[]
for role in ['train','select','calibrate']:
    for sid in SPLIT[role]:
        d=STORE[sid];mask=(d['y']>=0)&d['usable'];counts=np.bincount(np.asarray(d['y'][mask],int),minlength=3)
        support.append(dict(role=role,subject=sid,total_context_epochs=len(d['e']),scored_epochs=int(mask.sum()),
            **dict(zip(CLASS_NAMES,counts)),**{'timeline_'+k:v for k,v in METADATA[sid]['run_lengths'].items()}))
supports=pd.DataFrame(support);save_csv(supports,RUNROOT/'development_support.csv')
print(supports.groupby('role')[CLASS_NAMES+['total_context_epochs','scored_epochs']].sum())
for role in ['train','select','calibrate']:
    assert (supports[supports.role==role][CLASS_NAMES].sum().to_numpy()>0).all(),f'{role} lacks a class; inspect label encoding/coverage, never use test labels to fill it.'
SETTINGS=dict(version=PIPELINE_REV,tag=RUN_TAG,split=SPLIT,mode=STUDENT_MODE,features=FEATURE_NAMES,
    data={s:METADATA[s]['wear_key'] for s in sorted(STORE)},core=CORE,context=CONTEXT,
    teacher_folds=TEACHER_FOLDS,teacher_epochs=TEACHER_EPOCHS,teacher_minimum=TEACHER_MIN_EPOCHS,teacher_patience=TEACHER_PATIENCE,
    baseline_epochs=BASELINE_EPOCHS,distill_epochs=DISTILL_EPOCHS,student_minimum=STUDENT_MIN_EPOCHS,student_patience=STUDENT_PATIENCE,
    seeds=DISTILL_SEEDS,lr=[LR_TEACHER,LR_BASELINE,LR_DISTILL],weight_decay=WEIGHT_DECAY,
    kd=[KD_T,KD_WEIGHT,KD_WARMUP,KD_RAMP],rem_aux=REM_AUX_WEIGHT,class_power=CLASS_WEIGHT_POWER,
    effective_batch=[EFFECTIVE_BATCH_TEACHER,EFFECTIVE_BATCH_STUDENT],augmentation=AUGMENT_NOISE,
    targets=[TARGET_MACRO_F1,TARGET_ACCURACY],selection_accuracy_floor=SELECTION_ACCURACY_FLOOR)
EXPERIMENT_ID=digest(SETTINGS)
if (RUNROOT/'experiment.json').exists():
    assert digest(json.loads((RUNROOT/'experiment.json').read_text()))==EXPERIMENT_ID,'Experiment changed: use a new RUN_TAG. Existing artifacts are preserved.'
else:save_json(SETTINGS,RUNROOT/'experiment.json')
save_json(SPLIT,RUNROOT/'split.json')
```

## Cell 14 — explanation

## Models, normalization and optimizer steps

The student uses fast BVP morphology, slow wearable streams and54 summary/quality features. Availability masks handle an unusable modality without dropping the entire timeline. Pulse-interval summaries use detected intervals from a short segment; they are not long-recording LF/HF HRV estimates. The teacher preserves PSG amplitude before fitting training-only normalization, including chin EMG information useful for REM.

48-minute windows use16-minute cores plus left/right context. Each scored core is emitted once, avoiding uncertain edge averaging. Packed GRUs ignore padding. GroupNorm avoids participant/minibatch-dependent batch statistics. Effective-batch gradient accumulation, AMP, pinned transfers, local SSD mirrors and memory-mapped prepared arrays bound memory and Drive traffic. One available GPU is used; this is not distributed multi-GPU training.

## Cell 15 — code

```python
# Compact multiscale epoch encoders and explicit sequence padding for V7.
# Student inputs are wearable signals only; PSG is read only by TeacherV7.

class _V7ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=7, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else nn.Conv1d(in_channels, out_channels, 1, stride=stride,
                                    bias=False))

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))


class _V7EpochCNN(nn.Module):
    def __init__(self, in_channels, kernels, stem_channels, stride,
                 widths=(64, 96), output_dim=96):
        super().__init__()
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, stem_channels, k, stride=stride,
                          padding=k // 2, bias=False),
                nn.GroupNorm(8, stem_channels), nn.GELU(),
            ) for k in kernels
        ])
        width = stem_channels * len(kernels)
        blocks = []
        for i, next_width in enumerate(widths):
            blocks.append(_V7ConvBlock(width, next_width, stride=3 if i == 0 else 2))
            width = next_width
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Sequential(
            nn.Linear(2 * width, output_dim), nn.LayerNorm(output_dim),
            nn.GELU(), nn.Dropout(0.10),
        )

    def forward(self, x):
        # Per-epoch GroupNorm has no dependency on other epochs or participants.
        x = torch.cat([stem(x) for stem in self.stems], dim=1)
        x = self.blocks(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.project(pooled)


class _V7TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 3, padding=dilation,
                                   dilation=dilation, groups=width, bias=False)
        self.mix = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Dropout(0.15), nn.Linear(2 * width, width))
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, valid):
        # LayerNorm is per epoch. Invalid steps cannot enter a later convolution.
        m = valid.unsqueeze(-1).to(x.dtype)
        z = self.norm(x) * m
        z = self.depthwise(z.transpose(1, 2)).transpose(1, 2)
        return (x + self.dropout(self.mix(F.gelu(z)))) * m


class _V7TemporalHead(nn.Module):
    def __init__(self, width=160, hidden=128):
        super().__init__()
        self.context = nn.ModuleList([_V7TemporalBlock(width, d) for d in (1, 2, 4)])
        self.rnn = nn.GRU(width, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.15)
        self.residual = nn.Linear(width, 2 * hidden)
        self.norm = nn.LayerNorm(2 * hidden)
        self.dropout = nn.Dropout(0.20)
        self.stage = nn.Linear(2 * hidden, 3)
        # Auxiliary target is REM vs NREM conditional on sleep (root loss masks Wake).
        self.rem = nn.Linear(2 * hidden, 1)

    def forward(self, epoch_embeddings, lengths_cpu, valid):
        x = epoch_embeddings
        for block in self.context:
            x = block(x, valid)
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True,
                                      enforce_sorted=False)
        packed, _ = self.rnn(packed)
        x_rnn, _ = pad_packed_sequence(packed, batch_first=True,
                                       total_length=x.shape[1])
        embedding = self.norm(x_rnn + self.residual(x))
        embedding = embedding * valid.unsqueeze(-1).to(embedding.dtype)
        z = self.dropout(embedding)
        logits = self.stage(z) * valid.unsqueeze(-1).to(z.dtype)
        rem_logit = self.rem(z).squeeze(-1) * valid.to(z.dtype)
        return {"logits": logits, "rem_logit": rem_logit, "embedding": embedding}


def _v7_sequence_layout(lengths, batch_size, steps, device):
    lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
    if lengths_cpu.ndim != 1 or lengths_cpu.numel() != batch_size:
        raise ValueError("length must have one positive sequence length per batch item")
    if bool((lengths_cpu < 1).any()) or bool((lengths_cpu > steps).any()):
        raise ValueError("All sequence lengths must lie between 1 and padded T")
    valid = (torch.arange(steps, device=device)[None, :]
             < lengths_cpu.to(device)[:, None])
    return lengths_cpu, valid


class StudentV7(nn.Module):
    """Wearable-only student; forward never accesses PSG, labels, or score masks.

    fast: robust BVP and its derivative, 32 Hz, 960 samples/epoch.
    slow: IBI, ACC_X/Y/Z, EDA, TEMP, HR, 4 Hz, 120 samples/epoch.
    avail: valid fractions for BVP, IBI, ACC_X/Y/Z, EDA, TEMP, HR.
    All temporal context is wearable context; this is an offline bidirectional model.
    """
    def __init__(self, feature_dim=None):
        super().__init__()
        self.feature_dim = len(FEATURE_NAMES) if feature_dim is None else int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("StudentV7 requires at least one engineered wearable feature")
        self.fast_encoder = _V7EpochCNN(2, (7, 15, 31), 16, 4,
                                        widths=(64, 96), output_dim=96)
        self.slow_encoder = _V7EpochCNN(7, (5, 15), 32, 2,
                                        widths=(64, 96), output_dim=96)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim + 8, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
        )
        self.fusion = nn.Sequential(nn.Linear(96 + 96 + 64 + 8, 160),
                                     nn.LayerNorm(160), nn.GELU(), nn.Dropout(0.15))
        self.temporal = _V7TemporalHead(width=160, hidden=128)

    def forward(self, batch):
        fast, slow = batch["fast"], batch["slow"]
        features, available = batch["features"], batch["avail"]
        batch_size, steps = features.shape[:2]
        if features.shape[-1] != self.feature_dim:
            raise ValueError("Feature count differs from this StudentV7 checkpoint")
        lengths_cpu, valid = _v7_sequence_layout(batch["length"], batch_size,
                                                steps, features.device)
        # Encode actual epochs only. No epoch CNN spends memory on sequence padding.
        avail = torch.nan_to_num(available[valid], nan=0.0, posinf=0.0,
                                  neginf=0.0).clamp(0, 1)
        fast_valid = torch.nan_to_num(fast[valid], nan=0.0, posinf=0.0, neginf=0.0)
        slow_valid = torch.nan_to_num(slow[valid], nan=0.0, posinf=0.0, neginf=0.0)
        feature_valid = torch.nan_to_num(features[valid], nan=0.0,
                                          posinf=0.0, neginf=0.0)
        fast_valid = fast_valid * (avail[:, :1] > 0).to(fast.dtype).unsqueeze(-1)
        slow_valid = slow_valid * (avail[:, 1:] > 0).to(slow.dtype).unsqueeze(-1)
        fast_emb = self.fast_encoder(fast_valid) * avail[:, :1]
        slow_emb = self.slow_encoder(slow_valid) * avail[:, 1:].mean(dim=1, keepdim=True)
        feature_emb = self.feature_encoder(torch.cat([feature_valid, avail], dim=-1))
        fused = self.fusion(torch.cat([fast_emb, slow_emb, feature_emb, avail], dim=-1))
        epoch_embeddings = fused.new_zeros((batch_size, steps, fused.shape[-1]))
        epoch_embeddings[valid] = fused
        return self.temporal(epoch_embeddings, lengths_cpu, valid)


class TeacherV7(nn.Module):
    """Offline PSG teacher; forward reads only PSG and sequence lengths."""
    def __init__(self, feature_dim=None):
        super().__init__()
        # Matching constructor signatures simplify factory/checkpoint handling;
        # feature_dim is intentionally unused by this PSG-only teacher.
        self.epoch_encoder = _V7EpochCNN(7, (15, 31, 63), 24, 5,
                                         widths=(96, 128), output_dim=160)
        self.temporal = _V7TemporalHead(width=160, hidden=128)

    def forward(self, batch):
        psg = batch["psg"]
        batch_size, steps = psg.shape[:2]
        lengths_cpu, valid = _v7_sequence_layout(batch["length"], batch_size,
                                                steps, psg.device)
        psg_valid = torch.nan_to_num(psg[valid], nan=0.0, posinf=0.0, neginf=0.0)
        encoded = self.epoch_encoder(psg_valid)
        epoch_embeddings = encoded.new_zeros((batch_size, steps, encoded.shape[-1]))
        epoch_embeddings[valid] = encoded
        return self.temporal(epoch_embeddings, lengths_cpu, valid)
```

## Cell 16 — code

```python
# Train-only normalization, masked core windows, and honest out-of-fold KD.
def _moments_finish(total, square, count, floor=1e-3):
    count=np.maximum(count,1)
    mean=total/count
    std=np.sqrt(np.maximum(square/count-mean*mean,floor**2))
    return mean.astype(np.float32),std.astype(np.float32)

def normalization(subjects, kind):
    if kind=='teacher':
        total=np.zeros(7,np.float64); square=total.copy(); count=np.zeros(7,np.float64)
        for sid in subjects:
            d=STORE[sid]
            if 'psg' not in d: continue
            valid=np.flatnonzero(d['psg_valid'])
            for a in range(0,len(valid),32):
                x=np.asarray(d['psg'][valid[a:a+32]],np.float64)
                total+=x.sum((0,2));square+=(x*x).sum((0,2));count+=x.shape[0]*x.shape[-1]
        assert count.min()>0,'No aligned, valid PSG epochs for teacher normalization.'
        mean,std=_moments_finish(total,square,count)
        return dict(psg_mean=mean.tolist(),psg_std=std.tolist())
    total=np.zeros(7,np.float64);square=total.copy();count=total.copy()
    ft=np.zeros(len(FEATURE_NAMES),np.float64);fs=ft.copy();fn=0
    for sid in subjects:
        d=STORE[sid]
        for a in range(0,len(d['e']),128):
            x=np.asarray(d['slow'][a:a+128],np.float64)
            available=np.asarray(d['avail'][a:a+128,1:],bool)
            total+=(x*available[:,:,None]).sum((0,2))
            square+=(x*x*available[:,:,None]).sum((0,2))
            count+=available.sum(0)*x.shape[-1]
            valid=np.asarray(d['usable'][a:a+128],bool)
            f=np.asarray(d['features'][a:a+128],np.float64)[valid]
            ft+=f.sum(0);fs+=(f*f).sum(0);fn+=len(f)
    mean,std=_moments_finish(total,square,count)
    fm,fd=_moments_finish(ft,fs,np.full(len(ft),fn))
    return dict(slow_mean=mean.tolist(),slow_std=std.tolist(),feature_mean=fm.tolist(),feature_std=fd.tolist())

def contiguous_runs(epochs):
    boundaries=np.r_[0,np.flatnonzero(np.diff(epochs)!=1)+1,len(epochs)]
    return [(int(a),int(b)) for a,b in zip(boundaries[:-1],boundaries[1:]) if b>a]

class CoreWindows(Dataset):
    """Unlabeled epochs remain context. Every core epoch is emitted exactly once."""
    def __init__(self,subjects,norm,kind='student',training=False):
        self.norm,self.kind=norm,kind;self.windows=[]
        for sid in subjects:
            d=STORE[sid]
            for left,right in contiguous_runs(d['e']):
                for cs in range(left,right,CORE):
                    ce=min(cs+CORE,right);a=max(left,cs-CONTEXT);b=min(right,ce+CONTEXT)
                    valid=(d['y'][cs:ce]>=0)&d['usable'][cs:ce]
                    if kind=='teacher': valid=valid&d['teacher_available'][cs:ce]
                    if training and not np.any(valid): continue
                    if kind=='teacher' and not np.any(d['psg_valid'][cs:ce]): continue
                    self.windows.append((sid,a,b,cs,ce))
    def __len__(self): return len(self.windows)
    def __getitem__(self,index):
        sid,a,b,cs,ce=self.windows[index];d=STORE[sid];n=b-a
        y=np.full(SEQ,-100,np.int64);y[:n]=d['y'][a:b]
        valid=np.zeros(SEQ,bool);valid[:n]=(d['y'][a:b]>=0)&d['usable'][a:b]
        core=np.zeros(SEQ,bool);core[cs-a:ce-a]=True
        if self.kind=='teacher': valid[:n]&=d['teacher_available'][a:b]
        item=dict(y=torch.from_numpy(y),loss_mask=torch.from_numpy(valid&core),
                  core_mask=torch.from_numpy(core),length=n,index=index)
        if self.kind=='teacher':
            x=np.zeros((SEQ,7,30*PSG_FS),np.float32)
            mean=np.asarray(self.norm['psg_mean'],np.float32)[None,:,None]
            std=np.asarray(self.norm['psg_std'],np.float32)[None,:,None]
            v=np.asarray(d['psg_valid'][a:b],bool)
            raw=np.asarray(d['psg'][a:b],np.float32)
            x[:n]=np.clip((raw-mean)/std,-12,12)*v[:,None,None]
            item['psg']=torch.from_numpy(x)
        else:
            fast=np.zeros((SEQ,2,30*BVP_FS),np.float32)
            slow=np.zeros((SEQ,7,30*SLOW_FS),np.float32)
            avail=np.zeros((SEQ,8),np.float32);features=np.zeros((SEQ,len(FEATURE_NAMES)),np.float32)
            fast[:n]=d['fast'][a:b];avail[:n]=d['avail'][a:b]
            mean=np.asarray(self.norm['slow_mean'],np.float32)[None,:,None]
            std=np.asarray(self.norm['slow_std'],np.float32)[None,:,None]
            slow[:n]=np.clip((np.asarray(d['slow'][a:b],np.float32)-mean)/std,-10,10)*avail[:n,1:,None]
            fm=np.asarray(self.norm['feature_mean'],np.float32);fd=np.asarray(self.norm['feature_std'],np.float32)
            features[:n]=np.clip((np.asarray(d['features'][a:b],np.float32)-fm)/fd,-10,10)
            q=np.zeros((SEQ,3),np.float32);qvalid=np.zeros(SEQ,bool)
            if sid in QSTORE:
                q[:n]=QSTORE[sid]['logits'][a:b];qvalid[:n]=QSTORE[sid]['valid'][a:b]
            item.update(fast=torch.from_numpy(fast),slow=torch.from_numpy(slow),avail=torch.from_numpy(avail),
                        features=torch.from_numpy(features),q=torch.from_numpy(q),qvalid=torch.from_numpy(qvalid))
        return item
```

## Cell 17 — code

```python
def _batch_size(kind):
    effective=EFFECTIVE_BATCH_TEACHER if kind=='teacher' else EFFECTIVE_BATCH_STUDENT
    if BATCH:
        requested=BATCH
    elif not AMP:
        requested=1 if kind=='teacher' else 2
    else:
        free,_=torch.cuda.mem_get_info()
        requested=(1 if free<8*2**30 else 2 if free<18*2**30 else 4) if kind=='teacher' else (2 if free<8*2**30 else 4 if free<18*2**30 else 8)
    # Exact divisors retain the effective window batch when microbatch size changes on resume.
    return max(v for v in range(1,effective+1) if effective%v==0 and v<=max(1,requested))

def make_loader(data,order=None,batch=None):
    workers=NUM_WORKERS if os.name!='nt' else 0
    return DataLoader(data,batch_size=batch or _batch_size(data.kind),sampler=order,
        num_workers=workers,pin_memory=AMP,generator=torch.Generator().manual_seed(SEED),
        **({'prefetch_factor':2} if workers else {}))

def move_batch(raw):
    # Keep packed-RNN lengths and sample indices on CPU.
    return {k:(v.to(DEVICE,non_blocking=AMP) if torch.is_tensor(v) and k not in ['length','index'] else v) for k,v in raw.items()}

def metrics(y,p):
    y=np.asarray(y,np.int64);p=np.asarray(p,np.float64)
    assert len(y)>0 and p.shape==(len(y),3) and np.isfinite(p).all()
    p=np.maximum(p,1e-12);p/=p.sum(1,keepdims=True);pred=p.argmax(1)
    f=f1_score(y,pred,labels=[0,1,2],average=None,zero_division=0)
    result=dict(epochs=len(y),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f.mean()),
        weighted_f1=float(f1_score(y,pred,labels=[0,1,2],average='weighted',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),kappa=float(cohen_kappa_score(y,pred,labels=[0,1,2])),
        nll=float(log_loss(y,p,labels=[0,1,2])),brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))
    for i,name in enumerate(['wake','nrem','rem']):
        result[name+'_f1']=float(f[i]);result[name+'_support']=int((y==i).sum());result[name+'_predictions']=int((pred==i).sum())
    result['rem_ap']=float(average_precision_score(y==2,p[:,2])) if (y==2).any() else None
    return result

@torch.inference_mode()
def predict(model,subjects,norm,kind='student',batch=None):
    model.eval();data=CoreWindows(subjects,norm,kind,False);tables=[]
    for raw in make_loader(data,batch=batch):
        b=move_batch(raw)
        with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=AMP): z=model(b)['logits']
        z=z.float().cpu().numpy()
        for j,index in enumerate(raw['index'].tolist()):
            sid,a,end,cs,ce=data.windows[index];i0,i1=cs-a,ce-a;d=STORE[sid]
            t=pd.DataFrame(z[j,i0:i1],columns=['z_wake','z_nrem','z_rem'])
            t.insert(0,'epoch',np.asarray(d['e'][cs:ce],np.int64));t.insert(0,'subject',sid)
            t['true']=np.asarray(d['y'][cs:ce],np.int64)
            t['score']=(d['y'][cs:ce]>=0)&d['usable'][cs:ce]
            if kind=='teacher': t['score']=t['score'].to_numpy()&d['teacher_available'][cs:ce]
            tables.append(t)
    columns=['subject','epoch','z_wake','z_nrem','z_rem','true','score']
    table=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(columns=columns)
    assert not table.duplicated(['subject','epoch']).any(),'Duplicate core predictions'
    if kind=='student':
        assert len(table)==sum(len(STORE[s]['e']) for s in subjects),'Missing timeline predictions'
    return table

def table_metrics(table,temperature=1.):
    t=table[table.score.astype(bool)]
    return metrics(t.true,softmax(t[['z_wake','z_nrem','z_rem']].to_numpy(np.float64)/temperature,axis=1))

def fit_temperature(table):
    t=table[table.score.astype(bool)];z=t[['z_wake','z_nrem','z_rem']].to_numpy(np.float64);y=t.true.to_numpy(int)
    grid=np.unique(np.r_[1.,np.geomspace(.4,5.,70)])
    return float(min(grid,key=lambda v:log_loss(y,softmax(z/v,axis=1),labels=[0,1,2])))
```

## Cell 18 — code

```python
def class_weights_for(subjects,teacher=False):
    counts=np.zeros(3,np.int64)
    for sid in subjects:
        d=STORE[sid];valid=(d['y']>=0)&d['usable']
        if teacher: valid=valid&d['teacher_available']
        counts+=np.bincount(np.asarray(d['y'][valid],int),minlength=3)
    assert (counts>0).all(),f'Training lacks a class: {counts}'
    w=(counts.sum()/(3*counts))**CLASS_WEIGHT_POWER
    w=np.clip(w,.6,2.0);w/=w.mean()
    return counts,torch.tensor(w,dtype=torch.float32,device=DEVICE)

def loss_for(model,b,run,weights,epoch):
    output=model(b);mask=b['loss_mask'];y=b['y'][mask];z=output['logits'].float()[mask]
    assert len(y)>0,'A training window has no scored core labels.'
    supervised=F.cross_entropy(z,y,weight=weights)
    sleep=y!=0;aux=z.new_zeros(())
    if run['rem_weight'] and sleep.any():
        terms=F.binary_cross_entropy_with_logits(output['rem_logit'].float()[mask][sleep],(y[sleep]==2).float(),reduction='none')
        w=weights[y[sleep]];aux=(terms*w).sum()/w.sum()
    kd=z.new_zeros(());gated=0
    ramp=0. if epoch<KD_WARMUP else min(1.,(epoch-KD_WARMUP+1)/KD_RAMP)
    if run['kind']=='student' and run['kd_weight']>0 and ramp>0:
        q=b['q'][mask].float();valid=b['qvalid'][mask]
        gate=valid&(q.argmax(-1)==y)
        terms=[]
        for c in range(3):
            use=gate&(y==c)
            if use.any():
                target=F.softmax(q[use]/KD_T,-1)
                kl=F.kl_div(F.log_softmax(z[use]/KD_T,-1),target,reduction='none').sum(-1).mean()*KD_T**2
                terms.append(kl*float(TEACHER_RELIABILITY[c]))
        if terms: kd=torch.stack(terms).mean()
        gated=int(gate.sum().detach())
    loss=supervised+run['rem_weight']*aux+run['kd_weight']*ramp*kd
    return loss,gated

def rng_state():
    return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state_all() if AMP else [])
def restore_rng(state):
    random.setstate(state['py']);np.random.set_state(state['np']);torch.set_rng_state(state['cpu'])
    if AMP and len(state['gpu'])==torch.cuda.device_count():torch.cuda.set_rng_state_all(state['gpu'])
```

## Cell 19 — code

```python
def train_model(folder,run,train_ids,valid_ids,init_path=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);seed_all(run['seed'])
    kind=run['kind'];norm=normalization(train_ids,kind);counts,weights=class_weights_for(train_ids,kind=='teacher')
    model=(TeacherV7() if kind=='teacher' else StudentV7()).to(DEVICE)
    identity=digest(dict(pipeline=PIPELINE_REV,run=run,train=sorted(train_ids),valid=sorted(valid_ids),norm=norm,
        data={s:dict(wear=METADATA[s]['wear_key'],psg=METADATA[s].get('psg_key') if kind=='teacher' else None) for s in sorted(set(train_ids+valid_ids))},
        init=file_sha(init_path) if init_path else None,architecture=str(model),core=CORE,context=CONTEXT,
        weights=weights.detach().cpu().tolist(),kd=KD_ID if run['kd_weight'] else None,
        kd_settings=[KD_T,KD_WARMUP,KD_RAMP],effective_batch=EFFECTIVE_BATCH_TEACHER if kind=='teacher' else EFFECTIVE_BATCH_STUDENT))
    optimizer=torch.optim.AdamW(model.parameters(),lr=run['lr'],weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=run['epochs'],eta_min=run['lr']*.1)
    scaler=torch.amp.GradScaler('cuda',enabled=AMP and AMP_DTYPE==torch.float16)
    initial_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_rng=rng_state()
    state=dict(epoch=0,cursor=0,updates=0,best=-1.,bad=0,done=False,history=[],loss_sum=0.,loss_n=0,kd_examples=0)
    last=folder/'last.pt';best_path=folder/'best.pt'
    if last.exists():
        try:
            cp=load_own_checkpoint(last)
            assert isinstance(cp,dict)
            for field in ['identity','model','optimizer','scheduler','scaler','state','rng']:cp[field]
        except (OSError,ValueError,EOFError,RuntimeError,KeyError,TypeError,AssertionError,pickle.UnpicklingError) as exc:
            warnings.warn(f'{folder}: own resume checkpoint is unreadable; preserving it and retraining this stage: {exc}')
            prep_quarantine(last)
            if best_path.exists():prep_quarantine(best_path)
            cp=None
        if cp is not None:
            if cp['identity']!=identity:
                warnings.warn(f'{folder}: upstream model/target dependencies changed; preserving this stage and retraining it. Declared experiment settings are checked separately.')
                for stale in ['last.pt','best.pt','manifest.json','history.csv','selection_logits.csv','selection_logits.json']:
                    prep_quarantine(folder/stale)
                cp=None
            elif cp['state']['best']>=0 and not best_path.exists():
                warnings.warn(f'{folder}: selected weights are missing. Preserving the resume checkpoint and retraining this stage.')
                prep_quarantine(last);cp=None
        if cp is not None:
            try:
                selected=load_own_checkpoint(best_path) if best_path.exists() else None
                if selected is not None:
                    assert selected['identity']==identity
                    model.load_state_dict(selected['model'])
            except (OSError,ValueError,EOFError,RuntimeError,KeyError,TypeError,AssertionError,pickle.UnpicklingError) as exc:
                warnings.warn(f'{folder}: own selected weights are invalid; preserving artifacts and retraining: {exc}')
                prep_quarantine(last);prep_quarantine(best_path);cp=None
        if cp is not None:
            model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);scaler.load_state_dict(cp['scaler'])
            state=cp['state'];restore_rng(cp['rng'])
    if not last.exists():
        restore_rng(initial_rng)
        if init_path:
            cp=load_own_checkpoint(init_path);model.load_state_dict(cp['model'])
        else:model.load_state_dict(initial_model)
    del initial_model,initial_rng
    data=CoreWindows(train_ids,norm,kind,True);assert len(data),'No labeled training windows.'
    batch=_batch_size(kind);effective=EFFECTIVE_BATCH_TEACHER if kind=='teacher' else EFFECTIVE_BATCH_STUDENT
    accumulation=effective//batch
    def snapshot():
        save_torch(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),state=state,rng=rng_state(),norm=norm,run=run,train=train_ids,valid=valid_ids),last)
    if not last.exists():snapshot()
    while not state['done'] and state['epoch']<run['epochs']:
        order=np.random.default_rng(run['seed']+state['epoch']).permutation(len(data)).tolist()
        remaining=order[state['cursor']:];loader=make_loader(data,remaining,batch)
        n_batches=len(loader);model.train();optimizer.zero_grad(set_to_none=True);t0=time.time()
        try:
            for step,raw in enumerate(tqdm(loader,desc=f"{folder.name} epoch {state['epoch']+1}",leave=False)):
                b=move_batch(raw)
                if kind=='student' and AUGMENT_NOISE:
                    b['fast']=b['fast']+AUGMENT_NOISE*torch.randn_like(b['fast'])*b['avail'][:,:,0,None,None]
                group_start=(step//accumulation)*accumulation;group_size=min(accumulation,n_batches-group_start)
                with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=AMP):loss,gated=loss_for(model,b,run,weights,state['epoch'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss: inspect sensor scaling/availability.')
                scaler.scale(loss/group_size).backward()
                state['cursor']+=len(raw['y']);state['loss_sum']+=float(loss.detach());state['loss_n']+=1;state['kd_examples']+=gated
                if (step+1)%accumulation==0 or step+1==n_batches:
                    scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);state['updates']+=1
                    if state['updates']%SAVE_UPDATES==0:snapshot()
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError('GPU memory exhausted. Lower BATCH and rerun this cell with the same RUN_TAG; resume uses the last saved completed optimizer update and keeps the effective batch size.') from None
        table=predict(model,valid_ids,norm,kind,batch);m=table_metrics(table)
        eligible=not run['kd_weight'] or state['kd_examples']>0
        improved=eligible and m['macro_f1']>state['best']+1e-6;state['bad']=0 if improved else state['bad']+1
        if improved:
            state['best']=m['macro_f1']
            save_torch(dict(identity=identity,model=model.state_dict(),norm=norm,run=run,epoch=state['epoch']+1,metrics=m,
                train=train_ids,valid=valid_ids,kd_examples=state['kd_examples']),best_path)
            save_csv(table,folder/'selection_logits.csv')
            save_json(dict(best_sha=file_sha(best_path),identity=identity),folder/'selection_logits.json')
        row=dict(epoch=state['epoch']+1,loss=state['loss_sum']/max(state['loss_n'],1),seconds=time.time()-t0,kd_examples=state['kd_examples'],**m)
        state['history'].append(row)
        print(folder.name,{k:round(row[k],4) if isinstance(row[k],float) else row[k] for k in ['epoch','loss','accuracy','macro_f1','rem_f1','rem_predictions','kd_examples']})
        scheduler.step();state.update(epoch=state['epoch']+1,cursor=0,loss_sum=0.,loss_n=0)
        state['done']=state['epoch']>=run['epochs'] or (state['epoch']>=run['minimum'] and state['bad']>=run['patience'])
        snapshot();save_csv(pd.DataFrame(state['history']),folder/'history.csv')
    save_csv(pd.DataFrame(state['history']),folder/'history.csv')
    if not best_path.exists():
        if run['kd_weight'] and state['kd_examples']==0:
            raise RuntimeError('No correct OOF teacher examples contributed KD gradients. Inspect teacher_oof_audit.json and paired alignment; this run cannot produce a distilled checkpoint.')
        raise RuntimeError('No valid selected checkpoint was created; inspect training/validation support.')
    best=load_own_checkpoint(best_path);model.load_state_dict(best['model'])
    selection_path=folder/'selection_logits.csv';selection_meta=folder/'selection_logits.json'
    selection_key=dict(best_sha=file_sha(best_path),identity=identity)
    valid_selection=False
    if selection_path.exists() and selection_meta.exists():
        try:
            existing=pd.read_csv(selection_path)
            valid_selection=json.loads(selection_meta.read_text())==selection_key and not existing.duplicated(['subject','epoch']).any()
            valid_selection=valid_selection and {'subject','epoch','true','score','z_wake','z_nrem','z_rem'}<=set(existing.columns)
        except (OSError,ValueError,KeyError):valid_selection=False
    if not valid_selection:
        save_csv(predict(model,valid_ids,norm,kind,batch),selection_path);save_json(selection_key,selection_meta)
    save_json(dict(identity=identity,run=run,train=train_ids,valid=valid_ids,norm=norm,class_counts=counts,metrics=best['metrics']),folder/'manifest.json')
    if run['kd_weight'] and (state['kd_examples']==0 or best['kd_examples']==0):
        raise RuntimeError('No gated KD examples contributed to the chosen checkpoint. Inspect the teacher OOF audit; this model cannot be described as distilled.')
    del optimizer,scheduler,scaler,data;gc.collect()
    return model,best,norm
```

## Cell 20 — code

```python
def _teacher_support(subjects):
    counts=np.zeros(3,np.int64)
    for sid in subjects:
        d=STORE[sid];m=d['teacher_available']&(d['y']>=0)&d['usable'];counts+=np.bincount(d['y'][m],minlength=3)
    return counts

def teacher_targets():
    """Every training subject receives targets from a teacher that never fit/selected on them."""
    target_dir=RUNROOT/'oof_targets';target_dir.mkdir(parents=True,exist_ok=True)
    ids=np.array(SPLIT['train']);provenance=[];tables=[]
    folds=KFold(TEACHER_FOLDS,shuffle=True,random_state=SEED)
    for fold,(fit_index,held_index) in enumerate(folds.split(ids),1):
        pool=ids[fit_index].tolist();held=ids[held_index].tolist()
        pool=[s for s in pool if np.any(STORE[s]['psg_valid'])]
        for attempt in range(50):
            fit_ids,stop_ids=train_test_split(pool,test_size=max(3,int(np.ceil(.15*len(pool)))),random_state=SEED+fold+attempt)
            if (_teacher_support(fit_ids)>0).all() and (_teacher_support(stop_ids)>0).all():break
        else:raise ValueError('Cannot create inner teacher validation with all classes; inspect PSG/label availability.')
        assert not (set(fit_ids)|set(stop_ids))&set(held)
        folder=RUNROOT/'teachers'/f'fold_{fold:02d}'
        run=dict(name=f'teacher_{fold}',kind='teacher',seed=SEED+fold,lr=LR_TEACHER,epochs=TEACHER_EPOCHS,
                 minimum=TEACHER_MIN_EPOCHS,patience=TEACHER_PATIENCE,rem_weight=.05,kd_weight=0.)
        model,best,norm=train_model(folder,run,fit_ids,stop_ids)
        model_sha=file_sha(folder/'best.pt')
        temperature_path=folder/'temperature.json'
        temp_key=digest(dict(model=model_sha,stop=sorted(stop_ids),norm=norm))
        cached_temperature=False
        if temperature_path.exists():
            try:
                tj=json.loads(temperature_path.read_text())
                cached_temperature=tj.get('key')==temp_key and np.isfinite(tj['value']) and tj['value']>0
                if cached_temperature:temperature=tj['value']
            except (OSError,ValueError,KeyError,TypeError):cached_temperature=False
            if not cached_temperature:prep_quarantine(temperature_path)
        if not cached_temperature:
            temperature=fit_temperature(predict(model,stop_ids,norm,'teacher'))
            save_json(dict(key=temp_key,value=temperature),temperature_path)
        fold_meta=dict(fold=fold,fit=sorted(fit_ids),inner_stop=sorted(stop_ids),held_out=sorted(held),model_sha=model_sha,temperature=temperature)
        save_json(fold_meta,folder/'fold.json');provenance.append(fold_meta)
        for sid in held:
            key=digest(dict(fold=fold_meta,data=METADATA[sid]['psg_key'],wear=METADATA[sid]['wear_key'],pipeline=PIPELINE_REV))
            path=target_dir/f'{sid}.csv';mp=target_dir/f'{sid}.json'
            target_cached=False
            if path.exists() and mp.exists():
                try:
                    t=pd.read_csv(path)
                    target_cached=json.loads(mp.read_text()).get('key')==key
                    target_cached=target_cached and {'subject','epoch','true','score','z_wake','z_nrem','z_rem'}<=set(t.columns)
                    if target_cached and len(t):
                        target_cached=(set(t.subject)=={sid} and not t.duplicated(['subject','epoch']).any()
                            and np.isfinite(t[['z_wake','z_nrem','z_rem']].to_numpy(float)).all())
                except (OSError,ValueError,KeyError,TypeError):target_cached=False
                if not target_cached:
                    prep_quarantine(path);prep_quarantine(mp)
            if not target_cached:
                t=predict(model,[sid],norm,'teacher')
                # Calibrated logits remain logits; apply KD_T only in the KL loss.
                t[['z_wake','z_nrem','z_rem']]=t[['z_wake','z_nrem','z_rem']]/temperature
                save_csv(t,path);save_json(dict(key=key,fold=fold_meta,subject=sid),mp)
            assert not t.duplicated(['subject','epoch']).any()
            if len(t):
                assert set(t.subject)=={sid};tables.append(t)
        del model;gc.collect()
        if AMP:torch.cuda.empty_cache()
    all_targets=pd.concat(tables,ignore_index=True)
    assert not all_targets.duplicated(['subject','epoch']).any()
    for sid in SPLIT['train']:
        d=STORE[sid];q=np.zeros((len(d['e']),3),np.float32);valid=np.zeros(len(d['e']),bool)
        lookup={int(e):i for i,e in enumerate(d['e'])};t=all_targets[all_targets.subject==sid]
        for row in t.itertuples(index=False):
            if bool(row.score):
                i=lookup[int(row.epoch)];assert d['y'][i]==row.true
                q[i]=[row.z_wake,row.z_nrem,row.z_rem];valid[i]=True
        required=d['teacher_available']&(d['y']>=0)&d['usable']
        assert np.all(valid[required]),f'{sid}: missing valid OOF teacher targets'
        QSTORE[sid]=dict(logits=q,valid=valid)
    m=table_metrics(all_targets)
    scored=all_targets[all_targets.score.astype(bool)];p=softmax(scored[['z_wake','z_nrem','z_rem']].to_numpy(float),axis=1)
    report=classification_report(scored.true,p.argmax(1),labels=[0,1,2],target_names=CLASS_NAMES,output_dict=True,zero_division=0)
    reliability=np.clip([report[name]['recall'] for name in CLASS_NAMES],.05,1.)
    save_json(dict(metrics=m,report=report,reliability=reliability,provenance=provenance),RUNROOT/'teacher_oof_audit.json')
    save_csv(all_targets,RUNROOT/'teacher_oof_logits.csv');print('OOF teacher:',m)
    if m['rem_f1']<.4:warnings.warn('OOF teacher REM F1 is below .40. KD is running, but weak teacher targets may limit the target score; inspect alignment/PSG teacher quality.')
    return digest(dict(provenance=provenance,targets={s:file_sha(target_dir/f'{s}.csv') for s in SPLIT['train']})),reliability
```

## Cell 21 — explanation

## Train or reuse the PSG teachers and generate OOF targets

This is the expensive stage. For every fold, the held-out student-training people are excluded from teacher fitting, normalization and checkpoint selection. Teacher stopping/temperature people come from the remaining training pool. A wearable-only person or unavailable PSG epoch receives hard-label training but no KD target.

Missing or incompatible targets are **regenerated by training/resuming the corresponding PSG teacher**. KD is not silently disabled. Three-class teacher logits are temperature-calibrated on inner validation, saved by `(subject, epoch)`, and audited out of fold. A weak teacher causes a warning and lower class reliability; no score is fabricated.

## Cell 22 — code

```python
for sid in tqdm(SPLIT['train'],desc='Attach training-only paired PSG'):
    STORE[sid],METADATA[sid]=prepare_subject(sid,include_psg=True)
psg_rows=[]
for sid in SPLIT['train']:
    meta=METADATA[sid]
    psg_rows.append(dict(subject=sid,psg_key=meta.get('psg_key'),**meta.get('psg_qc',{})))
save_csv(pd.DataFrame(psg_rows),RUNROOT/'psg_alignment_audit.csv')
KD_ID,TEACHER_RELIABILITY=teacher_targets()
print('KD targets ready:',sum(int(q['valid'].sum()) for q in QSTORE.values()))
print('OOF per-class reliability:',dict(zip(CLASS_NAMES,TEACHER_RELIABILITY)))
```

## Cell 23 — explanation

## Supervised reference and distilled students

The supervised multimodal reference provides a same-input comparison. Each distilled student starts from that reference's best weights, warms up for3 epochs and ramps KD over8 epochs. A moderate class-balanced CE remains primary; the auxiliary REM-vs-NREM loss excludes Wake. KD uses only teacher-correct OOF targets and averages class terms, preventing abundant NREM targets from dominating. Every chosen distilled checkpoint must have received actual KD gradients. This comparison combines additional fine-tuning, KD and possibly ensembling; a matched-budget no-KD fine-tuning control would be needed to isolate KD's causal contribution.

Both seeds and their equal-probability ensemble are selected on the8 model-selection people. The preferred set must retain selection accuracy within1 point of the supervised reference and at least68%; if none meets that guard, the fallback is reported explicitly. No model is chosen using the20 historical test subjects. A high target cannot justify test-set threshold searches.

## Cell 24 — code

```python
baseline_run=dict(name='supervised_reference',kind='student',seed=SEED,lr=LR_BASELINE,epochs=BASELINE_EPOCHS,
    minimum=STUDENT_MIN_EPOCHS,patience=STUDENT_PATIENCE,rem_weight=REM_AUX_WEIGHT,kd_weight=0.)
BASE_FOLDER=RUNROOT/'students'/'supervised_reference'
model,best,base_norm=train_model(BASE_FOLDER,baseline_run,SPLIT['train'],SPLIT['select'])
base_selection=best['metrics'];del model,best;gc.collect()
if AMP:torch.cuda.empty_cache()
MEMBERS=[]
for seed in DISTILL_SEEDS:
    run=dict(name=f'distilled_seed_{seed}',kind='student',seed=seed,lr=LR_DISTILL,epochs=DISTILL_EPOCHS,
        minimum=max(STUDENT_MIN_EPOCHS,KD_WARMUP+KD_RAMP),patience=STUDENT_PATIENCE,rem_weight=REM_AUX_WEIGHT,kd_weight=KD_WEIGHT)
    folder=RUNROOT/'students'/run['name']
    model,best,norm=train_model(folder,run,SPLIT['train'],SPLIT['select'],BASE_FOLDER/'best.pt')
    MEMBERS.append(dict(name=run['name'],folder=str(folder),model_sha=file_sha(folder/'best.pt'),norm=norm,
        selection_metrics=best['metrics'],kd_examples=best['kd_examples']))
    del model,best;gc.collect()
    if AMP:torch.cuda.empty_cache()

def combine_tables(tables):
    reference=tables[0].reset_index(drop=True).copy();prob=[]
    for t in tables:
        t=t.reset_index(drop=True)
        assert reference[['subject','epoch','true','score']].equals(t[['subject','epoch','true','score']]),'Ensemble target alignment mismatch'
        prob.append(softmax(t[['z_wake','z_nrem','z_rem']].to_numpy(np.float64),axis=1))
    reference[['z_wake','z_nrem','z_rem']]=np.log(np.maximum(np.mean(prob,axis=0),1e-12))
    return reference

selection_tables=[pd.read_csv(Path(m['folder'])/'selection_logits.csv') for m in MEMBERS]
candidates=[]
for i,m in enumerate(MEMBERS):candidates.append(dict(name=m['name'],members=[i],**table_metrics(selection_tables[i])))
if len(MEMBERS)>1:
    ensemble=combine_tables(selection_tables)
    candidates.append(dict(name='distilled_ensemble',members=list(range(len(MEMBERS))),**table_metrics(ensemble)))
guard=max(SELECTION_ACCURACY_FLOOR,base_selection['accuracy']-.01)
for c in candidates:
    c['accuracy_guard_passed']=c['accuracy']>=guard
    c['joint_target_score']=min(c['macro_f1']/TARGET_MACRO_F1,c['accuracy']/TARGET_ACCURACY)
eligible=[c for c in candidates if c['accuracy_guard_passed']]
if eligible:chosen=max(eligible,key=lambda c:(c['macro_f1'],c['rem_f1'],c['accuracy']))
else:
    warnings.warn('No distilled candidate passed the validation accuracy guard. Reporting the best joint-target candidate; improvement is not established.')
    chosen=max(candidates,key=lambda c:(c['joint_target_score'],c['macro_f1'],c['accuracy']))
PRIMARY=[MEMBERS[i] for i in chosen['members']]
save_csv(pd.DataFrame([dict(c,members=json.dumps(c['members'])) for c in candidates]),RUNROOT/'candidate_selection.csv')
save_json(dict(chosen=chosen,primary_members=PRIMARY,supervised_reference=base_selection,accuracy_guard=guard),RUNROOT/'model_selection.json')
print(pd.DataFrame(candidates).drop(columns='members').to_string(index=False))
print('Frozen KD choice:',chosen['name'])
```

## Cell 25 — code

```python
@torch.inference_mode()
def predict_member(member,subjects):
    checkpoint=load_own_checkpoint(Path(member['folder'])/'best.pt')
    model=StudentV7().to(DEVICE);model.load_state_dict(checkpoint['model'])
    table=predict(model,subjects,checkpoint['norm'],'student')
    del model,checkpoint;gc.collect()
    if AMP:torch.cuda.empty_cache()
    return table
def predict_primary(subjects):return combine_tables([predict_member(m,subjects) for m in PRIMARY])
BASE_MEMBER=dict(name='supervised_reference',folder=str(BASE_FOLDER),model_sha=file_sha(BASE_FOLDER/'best.pt'),norm=base_norm)
calibration_key=digest(dict(experiment=EXPERIMENT_ID,primary=[m['model_sha'] for m in PRIMARY],baseline=BASE_MEMBER['model_sha'],
    calibration={s:METADATA[s]['wear_key'] for s in SPLIT['calibrate']},policy='temperature-only-no-offsets-v7'))
decoder_path=RUNROOT/'decoder.json'
cached_decoder=False
if decoder_path.exists():
    try:
        DECODER=json.loads(decoder_path.read_text())
        cached_decoder=DECODER.get('key')==calibration_key and DECODER['temperature']>0 and DECODER['baseline_temperature']>0
    except (OSError,ValueError,KeyError,TypeError):cached_decoder=False
    if not cached_decoder:prep_quarantine(decoder_path)
if not cached_decoder:
    primary_cal=predict_primary(SPLIT['calibrate']);base_cal=predict_member(BASE_MEMBER,SPLIT['calibrate'])
    DECODER=dict(key=calibration_key,temperature=fit_temperature(primary_cal),baseline_temperature=fit_temperature(base_cal),
        offsets=[0.,0.,0.],chosen=chosen['name'],selection=chosen,calibration_subjects=SPLIT['calibrate'])
    save_csv(primary_cal,RUNROOT/'calibration_logits.csv');save_csv(base_cal,RUNROOT/'baseline_calibration_logits.csv')
    save_json(DECODER,decoder_path)
save_json(dict(experiment=EXPERIMENT_ID,chosen=chosen,primary_members=PRIMARY,baseline=BASE_MEMBER,decoder=DECODER,
    test_subjects=SPLIT['test'],policy='Frozen before historical test preprocessing'),RUNROOT/'frozen_selection.json')
print('Temperature-only decoder:',DECODER['temperature'],'| class offsets:',DECODER['offsets'])
```

## Cell 26 — explanation

## Final historical benchmark — no further model selection

The20 test people are processed only now. Full-cohort scores describe all confidently scored native64Hz epochs with usable wearables. That coverage differs from V5/V6 because valid NREM transitions and wearable epochs are no longer rejected by PSG/IBI-only rules. The notebook therefore also reports **matched historical subject/epoch scores** when prior predictions exist. Compare versions on those matched rows, not mismatched aggregate cohorts. The supervised reference and selected distilled pipeline are both frozen and reported; neither is chosen after seeing test performance.

## Cell 27 — code

```python
PRED=RUNROOT/'predictions';PRED.mkdir(parents=True,exist_ok=True)
for sid in tqdm(SPLIT['test'],desc='Frozen historical test'):
    STORE[sid],METADATA[sid]=prepare_subject(sid,include_psg=False)
    key=digest(dict(experiment=EXPERIMENT_ID,decoder=DECODER,source=METADATA[sid]['wear_key'],
        primary=[m['model_sha'] for m in PRIMARY],baseline=BASE_MEMBER['model_sha']))
    path=PRED/f'{sid}.csv';mp=PRED/f'{sid}.json'
    if path.exists() and mp.exists() and json.loads(mp.read_text()).get('key')==key:
        cached=pd.read_csv(path)
        assert np.array_equal(cached.epoch.to_numpy(),STORE[sid]['e']),'Cached prediction timeline mismatch'
        continue
    t=predict_primary([sid]);b=predict_member(BASE_MEMBER,[sid])
    assert t[['subject','epoch','true','score']].equals(b[['subject','epoch','true','score']])
    prob=softmax(t[['z_wake','z_nrem','z_rem']].to_numpy(float)/DECODER['temperature'],axis=1)
    base_prob=softmax(b[['z_wake','z_nrem','z_rem']].to_numpy(float)/DECODER['baseline_temperature'],axis=1)
    for j,name in enumerate(['wake','nrem','rem']):t['p_'+name]=prob[:,j];t['baseline_p_'+name]=base_prob[:,j]
    t['predicted']=prob.argmax(1);t['baseline_predicted']=base_prob.argmax(1);t['epoch_start_seconds']=30*t.epoch
    save_csv(t,path);save_json(dict(key=key),mp)
timeline=pd.concat([pd.read_csv(PRED/f'{s}.csv') for s in SPLIT['test']],ignore_index=True)
assert not timeline.duplicated(['subject','epoch']).any()
assert len(timeline)==sum(len(STORE[s]['e']) for s in SPLIT['test'])
all_predictions=timeline[timeline.score.astype(bool)].copy()
save_csv(timeline,RUNROOT/'timeline_predictions.csv');save_csv(all_predictions,RUNROOT/'predictions.csv')
print('Timeline epochs:',len(timeline),'| scored epochs:',len(all_predictions),'| score fraction:',len(all_predictions)/len(timeline))
```

## Cell 28 — code

```python
def export_metrics(table,label,prob_prefix='',pred_col='predicted'):
    y=table.true.to_numpy(int);p=table[[prob_prefix+'p_'+c for c in ['wake','nrem','rem']]].to_numpy(np.float64)
    result=dict(method=label,**metrics(y,p))
    result.update(chosen_pipeline=chosen['name'] if label=='distilled_student' else 'supervised_reference',
                  member_count=len(PRIMARY) if label=='distilled_student' else 1)
    report=pd.DataFrame(classification_report(y,table[pred_col],labels=[0,1,2],target_names=CLASS_NAMES,output_dict=True,zero_division=0)).T
    save_csv(report.reset_index(names='class'),RUNROOT/f'{label}_class_metrics.csv')
    cm=pd.DataFrame(confusion_matrix(y,table[pred_col],labels=[0,1,2]),columns=['pred_'+c for c in CLASS_NAMES]);cm.insert(0,'true_class',CLASS_NAMES)
    save_csv(cm,RUNROOT/f'{label}_confusion_matrix.csv')
    return result
summary=pd.DataFrame([export_metrics(all_predictions,'supervised_reference','baseline_','baseline_predicted'),export_metrics(all_predictions,'distilled_student')])
save_csv(summary,RUNROOT/'metrics.csv');print(summary.to_string(index=False))
subjects=[]
for sid,g in all_predictions.groupby('subject'):
    subjects.append(dict(subject=sid,**metrics(g.true,g[['p_wake','p_nrem','p_rem']].to_numpy())))
save_csv(pd.DataFrame(subjects),RUNROOT/'subject_metrics.csv')
quality=[]
for sid,meta in METADATA.items():
    d=STORE[sid];scored=(d['y']>=0)&d['usable'];role=next(k for k in ['train','select','calibrate','test'] if sid in SPLIT[k])
    quality.append(dict(subject=sid,role=role,source='native64',total_context_epochs=len(d['e']),scored_epochs=int(scored.sum()),
        score_fraction=float(scored.mean()),**meta['run_lengths'],**{'scored_run_'+k:v for k,v in meta['scored_run_lengths'].items()},
        **meta['qc'].get('label_epoch_status',{})))
save_csv(pd.DataFrame(quality),RUNROOT/'quality_control.csv')
save_json({s:METADATA[s] for s in METADATA},RUNROOT/'preprocessing_audit.json')

def subject_bootstrap(table,pred_col='predicted',samples=1000):
    rng=np.random.default_rng(SEED)
    mats=np.stack([confusion_matrix(g.true,g[pred_col],labels=[0,1,2]) for _,g in table.groupby('subject')]);values=[]
    for _ in range(samples):
        cm=mats[rng.integers(0,len(mats),len(mats))].sum(0);den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0)
        values.append([np.trace(cm)/cm.sum(),f.mean(),f[2]])
    lo,hi=np.quantile(values,[.025,.975],axis=0)
    return pd.DataFrame(dict(metric=['accuracy','macro_f1','rem_f1'],lower_95=lo,upper_95=hi,samples=samples))
save_csv(subject_bootstrap(all_predictions),RUNROOT/'confidence_intervals.csv')
```

## Cell 29 — code

```python
# Historical comparison uses previous predictions for reporting only, never training or tuning.
comparison=[]
for version,path in [('V5',V5/'predictions.csv'),('V6',V6/'predictions.csv')]:
    if not path.exists():
        print(version,'historical predictions absent: matched comparison unavailable. New training/evaluation still completed.')
        continue
    old=pd.read_csv(path)
    if 'epoch' not in old:
        assert np.allclose(old.epoch_start_seconds/30,np.rint(old.epoch_start_seconds/30))
        old['epoch']=np.rint(old.epoch_start_seconds/30).astype(np.int64)
    assert not old.duplicated(['subject','epoch']).any()
    old=old[old.subject.isin(SPLIT['test'])]
    pair=all_predictions.merge(old[['subject','epoch','true','predicted']],on=['subject','epoch'],suffixes=('','_previous'),validate='one_to_one')
    assert len(pair)>0 and np.array_equal(pair.true,pair.true_previous),'Historical label/epoch mismatch: investigate source alignment before comparing.'
    for col,name in [('predicted_previous',version),('baseline_predicted','V7 supervised'),('predicted','V7 distilled')]:
        y=pair.true.to_numpy();pred=pair[col].to_numpy();f=f1_score(y,pred,labels=[0,1,2],average=None,zero_division=0)
        comparison.append(dict(reference=version,model=name,epochs=len(pair),previous_coverage=len(pair)/len(old),
            accuracy=accuracy_score(y,pred),macro_f1=f.mean(),weighted_f1=f1_score(y,pred,labels=[0,1,2],average='weighted',zero_division=0),
            wake_f1=f[0],nrem_f1=f[1],rem_f1=f[2]))
if comparison:
    comparison=pd.DataFrame(comparison);save_csv(comparison,RUNROOT/'historical_matched_comparison.csv');print(comparison.to_string(index=False))
primary=summary[summary.method=='distilled_student'].iloc[0]
target=dict(cohort='full_native64_scorable',chosen_pipeline=chosen['name'],member_count=len(PRIMARY),macro_f1_target=TARGET_MACRO_F1,accuracy_target=TARGET_ACCURACY,
    measured_macro_f1=float(primary.macro_f1),measured_accuracy=float(primary.accuracy),measured_rem_f1=float(primary.rem_f1),
    macro_target_met=bool(primary.macro_f1>=TARGET_MACRO_F1),accuracy_target_met=bool(primary.accuracy>=TARGET_ACCURACY),
    both_met=bool(primary.macro_f1>=TARGET_MACRO_F1 and primary.accuracy>=TARGET_ACCURACY))
if len(comparison):
    target['historical_matched_targets']=[dict(reference=row.reference,epochs=int(row.epochs),macro_f1=float(row.macro_f1),
        accuracy=float(row.accuracy),both_met=bool(row.macro_f1>=TARGET_MACRO_F1 and row.accuracy>=TARGET_ACCURACY))
        for row in comparison[comparison.model=='V7 distilled'].itertuples(index=False)]
save_json(target,RUNROOT/'target_attainment.json');print(json.dumps(target,indent=2))
if not target['both_met']:print('The target was NOT achieved in this run. Inspect teacher quality, modality coverage, alignment and subject metrics; do not claim a70% result.')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for folder in [BASE_FOLDER]+[Path(m['folder']) for m in MEMBERS]:
    h=pd.read_csv(folder/'history.csv');axes[0].plot(h.epoch,h.macro_f1,label=folder.name);axes[1].plot(h.epoch,h.rem_f1,label=folder.name)
axes[0].set_title('Selection macro F1');axes[1].set_title('Selection REM F1')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(RUNROOT/'training_diagnostics.png',dpi=150);plt.show()
print('Saved:',RUNROOT)
```

## Cell 30 — explanation

## Resume, reuse and interpret

- Rerun top to bottom after reconnecting. Raw-source/settings identities protect prepared arrays; completed teacher folds, OOF targets, supervised/KD checkpoints, temperature and per-person predictions are reused when compatible. Missing items are recalculated/retrained. CPU preprocessing and live processes must restart, but persisted files remain on Drive. Unknown/corrupt own cache artifacts are preserved before rebuilding when detected.
- The old V5/V6 wearable caches omit the newly required modalities and rejected context; their model architectures also differ. **Do not import their student weights or incomplete input caches.** They remain read only and can supply historical predictions for comparison. This one-time native64/PSG rebuild is necessary; future V7 runs reuse V7 artifacts. Exact model/source/config changes need a new `RUN_TAG`. The generated source revision does not automatically change if you edit a notebook cell by hand.
- A model checkpoint includes optimizer, scheduler, scaler, completed-window cursor and RNG state. Checkpoints are saved only after complete optimizer updates; unsaved work can repeat. Effective-batch gradient accumulation is preserved if a smaller microbatch is needed. Never load unknown downloaded pickle checkpoints.
- The final pipeline is `StudentV7` weights plus training-only normalization, modality preparation, frozen ensemble membership and `decoder.json`. Teachers and ground-truth labels are not required for deployed student forward inference. This notebook's evaluation wrapper uses labels solely to define scoring masks.
- Inspect `teacher_oof_audit.json`, `psg_alignment_audit.csv`, `development_support.csv`, both stage class reports/confusion matrices, `metrics.csv`, `subject_metrics.csv`, `quality_control.csv`, `historical_matched_comparison.csv` and `target_attainment.json`. A70% claim requires real measured output, with the cohort definition stated. The historical test has already been inspected; new independent participants provide stronger confirmation of a later final claim.

## Cell 31 — code

```python
# Optional: collect small result files after a real run for review; exclude caches/weights/raw data.
from zipfile import ZipFile,ZIP_DEFLATED
archive=Path('/content/DREAMT_V7_analysis.zip') if Path('/content').exists() else OUT/'DREAMT_V7_analysis.zip'
report_files=list(RUNROOT.glob('*.csv'))+list(RUNROOT.glob('*.json'))
report_files+=list(RUNROOT.glob('students/*/history.csv'))+list(RUNROOT.glob('teachers/*/history.csv'))
report_files+=list(RUNROOT.glob('students/*/selection_logits.csv'))
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for path in report_files:z.write(path,path.relative_to(RUNROOT))
print('Review archive:',archive)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    pass
```
