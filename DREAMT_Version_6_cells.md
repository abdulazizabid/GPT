# DREAMT Version 6 — cell-by-cell code

Copy each numbered code cell into a separate Colab cell, in order. Markdown cells provide explanations.

## Cell 1 — explanation

# DREAMT Version 6 — diagnose and address student REM collapse

Run top to bottom in Colab, preferably with a GPU. Set the paths in **Configuration**.
The original V5 files are read only; new caches, checkpoints, diagnostics and results go to `Version-6`.

**What the supplied results establish.** V5 accuracy is **72.05%**, macro F1 **47.22%** and REM F1 **0%** on 13,829 retained epochs (1,693 REM). Its rounded class metrics imply approximately **zero REM predictions**. The saved student log peaks at validation macro F1 **0.398** at epoch 6 and stops at epoch 11. The label map `REM: 4 → 2` is consistent. Imbalance is a plausible cause, but training counts and teacher quality must be measured before assigning causality.

**Probable remedies implemented here:** moderate train-only class weights; a supervised-only baseline plus a separate REM-auxiliary candidate; 32-minute context with overlapping prediction aggregation; GroupNorm; absolute and relative IBI features; training primarily on the simulated 64 Hz view; longer patience; full class diagnostics; and validation-only class decision offsets. Positive scalar temperature alone cannot change argmax, so V5's calibration cannot rescue a missing predicted class. KD is off by default; verified V5 OOF targets can optionally support a small, delayed, teacher-correct-gated KD candidate.

**No promised scores.** A new training/evaluation run is needed to establish improvement. These files do not include the raw DREAMT CSVs, V5 caches or model weights. Synthetic checks verify code paths, not physiological performance. The V5 test results have already been inspected, so the fixed 20-person test is a **historical benchmark**, not a new untouched confirmation set. Use another subject cohort for a stronger final confirmation.

The student still uses **BVP-derived IBI only**, without PSG, labels, timestamps or extra sensors as model inputs. It is an **offline bidirectional** model; it is not a causal real-time model.


## Cell 2 — code

```python
# Keep the runtime's existing GPU-enabled PyTorch. This cell installs small dependencies only.
%pip -q install numpy pandas scipy scikit-learn tqdm matplotlib
```

## Cell 3 — code

```python
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print('Not running in Colab: set the dataset/output paths below.')
```

## Cell 4 — code

```python
# Configuration: edit paths before running the remaining cells.
DATA = Path('/content/drive/MyDrive/Dataset/dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.2.0')
DIR100, DIR64 = DATA / 'data_100Hz', DATA / 'data_64Hz'
V5 = Path('/content/drive/MyDrive/Dataset/Version-5')
OUT = Path('/content/drive/MyDrive/Dataset/Version-6')
LOCAL = Path('/content/dreamt_v6_local' if Path('/content').exists() else '/tmp/dreamt_v6_local')

SEED = 42
SEQ, TRAIN_STRIDE, EVAL_STRIDE = 64, 32, 32  # 64 × 30 seconds = 32 minutes
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 45, 15, 12
LR, WEIGHT_DECAY, SAVE_STEPS = 3e-4, 1e-3, 100
BATCH = 0  # automatic conservative size; reduce if another process occupies GPU memory
NUM_WORKERS = 2  # arrays are staged in RAM; use 0 on Windows / constrained runtimes
CHUNK_ROWS = 300000
ID_PATTERN = r'(?i)(S\d+)'
COLUMN_OVERRIDE, LABEL_OVERRIDE, VALID_SECONDS = {}, {}, {}
WEARABLE_ONLY_SUBJECTS = {'S097'}

# Comparable mode reproduces V5's epoch acceptance rules, including PSG quality for 100 Hz.
# 'wearable_only' removes PSG-dependent training rejection and always rebuilds from raw CSVs.
COHORT_MODE = 'v5_comparable'
REUSE_V5_CACHE = True
ENABLE_KD_CANDIDATE = False  # requires verified V5 OOF provenance; otherwise skipped, not invented
RUN_TAG = 'v6_1'  # use a NEW tag/OUT when changing code, architecture or training settings
RUNS = [dict(name='balanced_ce', seed=SEED, rem_weight=0.0, kd_weight=0.0),
        dict(name='rem_aux', seed=SEED, rem_weight=0.20, kd_weight=0.0)]
CLASS_NAMES = ['Wake', 'NREM', 'REM']
```

## Cell 5 — code

```python
import os, re, json, random, hashlib, shutil, time, gc, warnings, zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.signal import butter, sosfiltfilt, find_peaks, resample_poly
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix,
    cohen_kappa_score, log_loss, average_precision_score, balanced_accuracy_score)
from functools import lru_cache
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

assert COHORT_MODE in {'v5_comparable', 'wearable_only'}
assert SEQ > 0 and 0 < TRAIN_STRIDE <= SEQ and 0 < EVAL_STRIDE <= SEQ
OUT.mkdir(parents=True, exist_ok=True)
LOCAL.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
AMP = DEVICE.type == 'cuda'
AMP_DTYPE = torch.bfloat16 if AMP and torch.cuda.is_bf16_supported() else torch.float16
torch.set_num_threads(min(8, os.cpu_count() or 2))
if AMP:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if AMP: torch.cuda.manual_seed_all(seed)
seed_all(SEED)
print('Device:', DEVICE, '| AMP:', AMP, '| PyTorch:', torch.__version__)
if AMP:
    print(torch.cuda.get_device_name(0), '— this notebook uses one GPU efficiently.')
else:
    warnings.warn('CPU fallback works; full DREAMT training will be substantially slower.')

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return x
def digest(x):
    return hashlib.sha256(json.dumps(jsonable(x),sort_keys=True).encode()).hexdigest()
def save_json(x, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+'.tmp'); temp.write_text(json.dumps(jsonable(x),indent=2)); os.replace(temp,path)
def save_csv(df,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp'); df.to_csv(temp,index=False); os.replace(temp,path)
def save_npz(path,**data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name(path.name+'.tmp').open('wb') as f: np.savez(f,**data)
    os.replace(path.with_name(path.name+'.tmp'),path)
def save_torch(x,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp'); torch.save(x,temp); os.replace(temp,path)
def load_own_checkpoint(path):
    # Only load checkpoints produced by this notebook in your own trusted OUT directory.
    return torch.load(path,map_location='cpu',weights_only=False)
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
```

## Cell 6 — explanation

## Data, split and coverage policy

The original fixed 80/20 subject lists and V5's seeded 64/16 development split are preserved. The 16 validation subjects are further split into **8 model-selection** and **8 calibration/decision** subjects. Neither subset contributes to gradients or normalization. Check their class support below; if a class is absent, the notebook stops instead of silently tuning on the test.

By default the same V5 epoch rules are retained for comparison. That includes rejecting PSG-unusable 100 Hz epochs even for student training, while 64 Hz test uses wearable quality only. This is a limitation of V5's cohort definition, not required by the student. `wearable_only` is an optional separate experiment; do not compare its unpaired full-cohort score directly to V5. Mixed labels, preparation/missing labels and other exclusions are logged separately when preprocessing from raw files. A reused V5 cache retains V5's coarser exclusion reasons.


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
def scan(folder):
    files={}
    for p in sorted(folder.glob('*.csv')):
        m=re.search(ID_PATTERN,p.stem)
        if not m: raise ValueError(f'Cannot identify participant: {p.name}')
        sid=m.group(1).upper()
        if sid in files: raise ValueError(f'Duplicate {sid} in {folder}')
        files[sid]=p
    return files
FILES100, FILES64 = scan(DIR100), scan(DIR64)
required=set(train_subjects+test_subjects)
assert len(set(train_subjects))==80 and len(set(test_subjects))==20
assert not set(train_subjects)&set(test_subjects)
assert required <= set(FILES100), f'100 Hz files missing: {sorted(required-set(FILES100))}. Set DATA first.'
assert set(test_subjects) <= set(FILES64), 'Missing 64 Hz test files.'
train_ids, valid_ids = train_test_split(train_subjects,test_size=16,random_state=SEED)
select_ids, calibrate_ids = train_test_split(sorted(valid_ids),test_size=8,random_state=SEED+100)
SPLIT=dict(train=sorted(train_ids),select=sorted(select_ids),calibrate=sorted(calibrate_ids),
           valid=sorted(valid_ids),test=sorted(test_subjects))
save_json(SPLIT,OUT/'split.json')
print({k:len(v) for k,v in SPLIT.items()})
print('Selection:',SPLIT['select']); print('Calibration:',SPLIT['calibrate'])

PSG=['C4-M1','F4-M1','O2-M1','E1','E2','CHIN','ECG']
def clean_name(name):
    return re.sub(r'[^a-z0-9]','',re.sub(r'\[.*?\]|\(.*?\)','',str(name)).lower())
LABELS={'w':0,'wake':0,'n1':1,'nrem1':1,'n2':2,'nrem2':2,'n3':3,'nrem3':3,
        'r':4,'rem':4,'remsleep':4,'p':-1,'preparation':-1,'missing':-1,'nan':-1,'':-1}
LABELS.update({clean_name(k):int(v) for k,v in LABEL_OVERRIDE.items()})
TO3=np.array([0,1,1,1,2],np.int64)
assert TO3[4]==2
def columns(path,need_psg):
    header=list(pd.read_csv(path,nrows=0).columns); result={}
    for name in ['TIMESTAMP','Sleep_Stage','BVP']+(PSG if need_psg else []):
        wanted=COLUMN_OVERRIDE.get(name)
        match=[h for h in header if h==wanted] if wanted else [h for h in header if clean_name(h)==clean_name(name)]
        if not match and name in PSG: match=[h for h in header if clean_name(h)=='eeg'+clean_name(name)]
        if len(match)!=1: raise ValueError(f'{path.name}: specify COLUMN_OVERRIDE[{name!r}]. Headers={header}')
        result[name]=match[0]
    return result
def raw_signature(sid,rate):
    p=(FILES100 if rate==100 else FILES64)[sid]; s=p.stat()
    return dict(subject=sid,rate=rate,path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)
PIPELINE_REV='cd840bd742c81e4c7ac6bc4b3dc28f63492589989ad607050bdea581dfe4c3ba'
PREP=dict(revision=PIPELINE_REV,mode=COHORT_MODE,chunk=CHUNK_ROWS,columns=COLUMN_OVERRIDE,
          labels=LABEL_OVERRIDE,intervals=VALID_SECONDS,wearable_only=sorted(WEARABLE_ONLY_SUBJECTS))
def cache_key(sid,rate): return digest(dict(raw=raw_signature(sid,rate),prep=PREP))
```

## Cell 9 — code

```python
# V5 cache compatibility is checked, not inferred from a filename alone.
def read_legacy_json(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text())
    except (OSError,ValueError) as exc:
        warnings.warn(f'V5 metadata unavailable; use raw preprocessing: {path.name}: {exc}')
        return None
LEGACY_CONFIG=read_legacy_json(V5/'config.json')
LEGACY_SPLIT=read_legacy_json(V5/'split.json')
def legacy_compatible(sid,rate):
    if not REUSE_V5_CACHE or COHORT_MODE!='v5_comparable' or not LEGACY_CONFIG: return False
    c=LEGACY_CONFIG
    if c.get('version')!=2 or c.get('chunk')!=CHUNK_ROWS: return False
    if any(c.get(k)!=v for k,v in [('columns',COLUMN_OVERRIDE),('labels',LABEL_OVERRIDE),('intervals',VALID_SECONDS)]): return False
    if LEGACY_SPLIT!={k:SPLIT[k] for k in ['train','valid','test']}: return False
    policy=V5/'psg_policy.json'
    if not policy.exists() or json.loads(policy.read_text())!={'version':1,'wearable_only':sorted(WEARABLE_ONLY_SUBJECTS)}: return False
    p=(FILES100 if rate==100 else FILES64)[sid]; stat=p.stat()
    return [sid,str(p),stat.st_size,stat.st_mtime_ns] in c.get('files',[])

def validate_payload(d):
    n=len(d['e'])
    assert n>0 and d['w'].shape==d['v'].shape==(n,750), 'Invalid IBI cache shape'
    assert len(np.unique(d['e']))==n and np.all(np.diff(d['e'])>0), 'Duplicate/unordered epochs'
    assert d['y5'].shape==(n,) and np.isin(d['y5'],np.arange(5)).all(), 'Invalid stage labels'
    assert np.isfinite(d['w']).all() and np.isfinite(d['v']).all(), 'Nonfinite IBI cache'
    assert np.all((d['w']>=0.30-1e-5)&(d['w']<=2.0+1e-5)), 'IBI seconds expected'
    assert np.all((d['v']>=0.30-1e-5)&(d['v']<=2.0+1e-5)), 'IBI seconds expected'

def import_legacy(sid,rate):
    folder=V5/'cache'/str(rate)/sid
    if not legacy_compatible(sid,rate) or not (folder/'done.json').exists(): return None
    parts=[]
    for p in sorted(folder.glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            if 'e' in z:
                parts.append({k:np.array(z[k]) for k in ['w','v','y','e']})
    if not parts: return None
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ix=np.argsort(d['e']); d={k:v[ix] for k,v in d.items()}; d['y5']=d.pop('y')
    validate_payload(d)
    qc=json.loads((folder/'done.json').read_text())
    assert qc.get('kept')==len(d['e']), 'Incomplete V5 cache'
    return d,qc
```

## Cell 10 — code

```python
# CPU preprocessing matches V5 IBI extraction; no TLS/checksum bypasses or raw-file edits.
def repair(values,fs):
    values=np.asarray(values,dtype=np.float64); good=np.isfinite(values)
    if not len(values) or not good[0] or not good[-1] or good.mean()<0.99: return None
    missing=np.flatnonzero(~good)
    if len(missing):
        runs=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
        if max(map(len,runs))>int(0.25*fs): return None
        values=np.interp(np.arange(len(values)),np.flatnonzero(good),values[good])
    return values if np.std(values)>1e-8 else None
@lru_cache(maxsize=32)
def band(fs,low,high): return butter(3,[low,high],btype='bandpass',fs=fs,output='sos')
def make_ibi(bvp,fs):
    x=repair(bvp,fs)
    if x is None: return None
    x=sosfiltfilt(band(fs,0.5,8.0),x)
    peaks,_=find_peaks(x,distance=max(1,int(0.30*fs)),prominence=0.2*np.std(x))
    if len(peaks)<12: return None
    times=peaks/fs; intervals=np.diff(times); good=(intervals>=0.30)&(intervals<=2.0)
    t,ibi=times[1:][good],intervals[good]
    if len(t)<10 or t[0]>3 or t[-1]<27 or np.max(np.diff(t))>3: return None
    return np.interp(np.arange(750)/25,t,ibi).astype(np.float32)
def encode_labels(series):
    names=series.fillna('').map(clean_name); unknown=sorted(set(names)-set(LABELS))
    if unknown: raise ValueError(f'Unknown stage labels {unknown}; verify LABEL_OVERRIDE.')
    y=names.map(LABELS).to_numpy(np.int64)
    assert np.isin(y,[-1,0,1,2,3,4]).all()
    return y
def epoch_blocks(path,mapping,sid):
    carry=pd.DataFrame(); previous=-np.inf
    for frame in pd.read_csv(path,usecols=list(mapping.values()),chunksize=CHUNK_ROWS):
        frame=frame.rename(columns={v:k for k,v in mapping.items()})
        for c in frame.columns.drop('Sleep_Stage'): frame[c]=pd.to_numeric(frame[c],errors='coerce')
        frame['Sleep_Stage']=encode_labels(frame.Sleep_Stage)
        t=frame.TIMESTAMP.to_numpy(float)
        if not np.isfinite(t).all() or np.any(np.diff(t)<=0) or t[0]<=previous:
            raise ValueError(f'{sid}: timestamps must be finite, increasing seconds; verify units.')
        previous=t[-1]; frame['epoch']=np.floor((t+1e-6)/30).astype(np.int64)
        frame=pd.concat([carry,frame],ignore_index=True); last=frame.epoch.iloc[-1]
        ready,carry=frame[frame.epoch!=last],frame[frame.epoch==last]
        if len(ready): yield ready
    if len(carry): yield carry
def prepare_epoch(group,fs,need_psg):
    if len(group)!=30*fs: return None,'incomplete'
    if not np.allclose(np.diff(group.TIMESTAMP.to_numpy(float)),1/fs,atol=.0015,rtol=0): return None,'timestamp_gap'
    y=group.Sleep_Stage.to_numpy(np.int64)
    if np.any(y<0): return None,'preparation_or_missing_label'
    if np.any(y!=y[0]): return None,'mixed_label'
    bvp=group.BVP.to_numpy(float); w=make_ibi(bvp,fs)
    if w is None: return None,'bad_bvp'
    v=make_ibi(resample_poly(repair(bvp,fs),16,25),64) if fs==100 else w.copy()
    if v is None: return None,'bad_resampled_bvp'
    if need_psg and any(repair(group[c].to_numpy(float),fs) is None for c in PSG): return None,'bad_psg'
    return dict(w=w,v=v,y5=int(y[0])),'kept'

def prepare_subject(sid,rate):
    key=cache_key(sid,rate); folder=OUT/'cache'/str(rate)/sid/key[:16]
    payload=folder/'subject.npz'; manifest=folder/'manifest.json'
    if payload.exists() and manifest.exists():
        try:
            meta=json.loads(manifest.read_text())
            if meta.get('key')==key:
                with np.load(payload,allow_pickle=False) as z: d={k:np.array(z[k]) for k in ['w','v','y5','e']}
                validate_payload(d); return d,meta
        except (OSError,ValueError,KeyError,AssertionError,EOFError,zipfile.BadZipFile) as exc:
            warnings.warn(f'{sid}: invalid V6 cache, rebuilding: {exc}')
            for bad in [payload,manifest]:
                if bad.exists(): bad.rename(bad.with_name(bad.name+f'.invalid_{time.time_ns()}'))
    try:
        imported=import_legacy(sid,rate)
    except (OSError,ValueError,KeyError,AssertionError,EOFError,zipfile.BadZipFile) as exc:
        warnings.warn(f'{sid}: V5 cache not usable; rebuilding from raw CSV: {exc}')
        imported=None
    if imported is not None:
        d,qc=imported; source='validated V5 cache'
    else:
        need_psg=rate==100 and COHORT_MODE=='v5_comparable' and sid not in WEARABLE_ONLY_SUBJECTS
        path=(FILES100 if rate==100 else FILES64)[sid]; mapping=columns(path,need_psg)
        blockdir=folder/'blocks'; blockdir.mkdir(parents=True,exist_ok=True)
        blockpaths=[]
        for b,frame in enumerate(epoch_blocks(path,mapping,sid)):
            bp=blockdir/f'{b:05d}.npz'; blockpaths.append(bp)
            if bp.exists():
                try:
                    with np.load(bp,allow_pickle=False) as z:
                        json.loads(str(z['qc']))
                        if 'e' in z: validate_payload({k:np.array(z[k]) for k in ['w','v','y5','e']})
                    continue  # source/config-specific folder; repeat only unfinished blocks
                except (OSError,ValueError,KeyError,AssertionError,EOFError,zipfile.BadZipFile):
                    bp.rename(bp.with_name(bp.name+f'.invalid_{time.time_ns()}'))
            rows=[]; counts={}; lo,hi=VALID_SECONDS.get(sid,(-np.inf,np.inf))
            for e,g in frame.groupby('epoch',sort=True):
                if e*30<lo or (e+1)*30>hi: result,reason=None,'outside_verified_interval'
                else: result,reason=prepare_epoch(g,rate,need_psg)
                counts[reason]=counts.get(reason,0)+1
                if result is not None: result['e']=int(e); rows.append(result)
            p={k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
            save_npz(bp,**p,qc=np.array(json.dumps(counts)))
        parts=[]; qc={}
        for bp in blockpaths:
            with np.load(bp,allow_pickle=False) as z:
                for k,v in json.loads(str(z['qc'])).items(): qc[k]=qc.get(k,0)+v
                if 'e' in z: parts.append({k:np.array(z[k]) for k in ['w','v','y5','e']})
        if not parts: raise ValueError(f'{sid}: no usable epochs. Inspect raw data and QC.')
        d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}; source='raw CSV'
    validate_payload(d)
    meta=dict(key=key,raw=raw_signature(sid,rate),prep=PREP,source=source,qc=qc,
              epochs=len(d['e']),class_counts=np.bincount(TO3[d['y5']],minlength=3))
    save_npz(payload,**d); save_json(meta,manifest)
    return d,meta
```

## Cell 11 — code

```python
# Stage compact wearable arrays into RAM; avoid tiny random reads from Google Drive per batch.
STORE, METADATA = {}, {}
for sid in tqdm(SPLIT['train']+SPLIT['valid'],desc='Prepare development subjects'):
    STORE[sid],METADATA[sid]=prepare_subject(sid,100)
def support_table():
    rows=[]
    for split in ['train','select','calibrate']:
        for sid in SPLIT[split]:
            counts=np.bincount(TO3[STORE[sid]['y5']],minlength=3)
            rows.append(dict(split=split,subject=sid,**dict(zip(CLASS_NAMES,counts))))
    return pd.DataFrame(rows)
supports=support_table(); save_csv(supports,OUT/'development_class_support.csv')
print(supports.groupby('split')[CLASS_NAMES].sum())
for split in ['train','select','calibrate']:
    assert (supports[supports.split==split][CLASS_NAMES].sum().to_numpy()>0).all(), f'{split}: a class is absent; choose a predeclared new development split, never use test labels.'
train_counts=supports[supports.split=='train'][CLASS_NAMES].sum().to_numpy(float)
weights=np.sqrt(train_counts.sum()/(3*train_counts)); weights=np.clip(weights,.5,3.0); weights/=weights.mean()
CLASS_WEIGHTS=torch.tensor(weights,dtype=torch.float32,device=DEVICE)
REM_POS_WEIGHT=torch.tensor(float(np.clip(np.sqrt((train_counts[0]+train_counts[1])/train_counts[2]),1,4)),device=DEVICE)
print('Train-only CE weights:',dict(zip(CLASS_NAMES,weights.round(3))))
print('No class oversampling is applied together with these weights.')
```

## Cell 12 — code

```python
# Optional teacher reuse: require matching cache, split and OOF fold exclusion evidence.
TARGETS={}
def load_verified_oof():
    qmap={}; provenance=[]
    for sid in SPLIT['train']:
        if sid in WEARABLE_ONLY_SUBJECTS: continue
        if not legacy_compatible(sid,100): raise ValueError(f'{sid}: V5 preprocessing/config mismatch')
        matches=[]
        for p in V5.glob('teacher_fold_*.json'):
            fold=json.loads(p.read_text())
            if sid in fold.get('held_out',[]): matches.append((p,fold))
        if len(matches)!=1: raise ValueError(f'{sid}: missing/ambiguous OOF fold manifest')
        p,fold=matches[0]; fitted=set(fold['train'])|set(fold['calibration']); held=set(fold['held_out'])
        assert not fitted&held and (fitted|held)<=set(SPLIT['train']), 'Teacher subject leakage'
        table=pd.read_csv(V5/'teacher_targets'/f'{sid}.csv')
        assert not table.duplicated(['subject','epoch']).any()
        assert set(table.subject)=={sid}
        p3=table[['wake','nrem','rem']].to_numpy(float)
        assert np.isfinite(p3).all() and (p3>=0).all() and np.allclose(p3.sum(1),1,atol=1e-4)
        allowed=set(STORE[sid]['e'].tolist())
        for row,q in zip(table.itertuples(index=False),p3):
            if int(row.epoch) in allowed: qmap[(sid,int(row.epoch))]=q.astype(np.float32)
        assert all((sid,int(e)) in qmap for e in STORE[sid]['e'])
        provenance.append(dict(subject=sid,fold=p.name,target_sha=file_sha(V5/'teacher_targets'/f'{sid}.csv')))
    return qmap,provenance
if ENABLE_KD_CANDIDATE:
    try:
        TARGETS,OOF_PROVENANCE=load_verified_oof()
    except (AssertionError,ValueError,FileNotFoundError,KeyError) as exc:
        warnings.warn(f'KD candidate skipped: {exc}. Supervised candidates still train.'); TARGETS={}
    if TARGETS:
        pairs=[(TO3[y],TARGETS[(sid,int(e))]) for sid in SPLIT['train'] for e,y in zip(STORE[sid]['e'],STORE[sid]['y5']) if (sid,int(e)) in TARGETS]
        y=np.array([v[0] for v in pairs]); q=np.stack([v[1] for v in pairs])
        audit=classification_report(y,q.argmax(1),labels=[0,1,2],target_names=CLASS_NAMES,output_dict=True,zero_division=0)
        save_json(audit,OUT/'oof_teacher_audit.json'); print(pd.DataFrame(audit).T)
        if audit['REM']['f1-score']>0 and audit['REM']['recall']>=.10:
            RUNS.append(dict(name='rem_aux_kd',seed=SEED,rem_weight=.20,kd_weight=.05))
        else:
            warnings.warn('Teacher REM recall is too weak for the predeclared KD gate; KD candidate skipped.')
            TARGETS={}
```

## Cell 13 — explanation

## Student model and training

The CNN sees globally normalized absolute IBI plus relative within-epoch IBI. An additional small feature branch uses IBI mean, spread, quantiles and one-second variability summaries. These are **interpolated-series features**, not clinical beat-to-beat HRV measurements; LF/HF is not estimated from a 30-second segment.

Training windows overlap, never cross people or rejected-epoch gaps, and use random window order without oversampling. Evaluation averages overlapping **logits** and then emits each retained epoch exactly once. Normalization and class weights use only the 64 training people. Both supervised candidates start from the same seed/architecture, so the REM auxiliary loss can be assessed on selection subjects. No V5 student/optimizer state is imported into this new architecture.


## Cell 14 — code

```python
FEATURE_NAMES=['mean_ibi','std_ibi','q10','median','q90','iqr','mean_hr',
               'std_one_second','rms_one_second_difference','abs_one_second_difference',
               'linear_trend','robust_range']
def ibi_features(x):
    x=np.asarray(x,np.float32); one=x.reshape(-1,30,25).mean(-1); d=np.diff(one,axis=-1)
    q10,q25,q50,q75,q90=np.quantile(x,[.10,.25,.50,.75,.90],axis=-1)
    t=np.linspace(-1,1,30,dtype=np.float32); trend=(one*t).mean(-1)/(t*t).mean()
    return np.stack([x.mean(-1),x.std(-1),q10,q50,q90,q75-q25,(60/np.clip(x,.3,2)).mean(-1),
        one.std(-1),np.sqrt((d*d).mean(-1)),np.abs(d).mean(-1),trend,q90-q10],axis=-1).astype(np.float32)
for d in STORE.values():
    d['fw'],d['fv']=ibi_features(d['w']),ibi_features(d['v'])
    d['q']=np.zeros((len(d['e']),3),np.float32)
for sid,d in STORE.items():
    for i,e in enumerate(d['e']): d['q'][i]=TARGETS.get((sid,int(e)),np.zeros(3,np.float32))

def fit_normalization():
    # The simulated 64 Hz training view matches the inference pipeline most closely.
    n=sum(STORE[s]['v'].size for s in SPLIT['train'])
    mean=sum(STORE[s]['v'].sum(dtype=np.float64) for s in SPLIT['train'])/n
    second=sum(np.square(STORE[s]['v'].astype(np.float64)).sum() for s in SPLIT['train'])/n
    features=np.concatenate([STORE[s]['fv'] for s in SPLIT['train']])
    return dict(ibi_mean=float(mean),ibi_std=float(max(np.sqrt(max(second-mean*mean,0)),1e-3)),
                feature_mean=features.mean(0).tolist(),feature_std=np.maximum(features.std(0),1e-3).tolist())
NORM=fit_normalization(); save_json(NORM,OUT/'normalization.json')

def continuous_runs(epochs):
    edges=[0]+(np.flatnonzero(np.diff(epochs)!=1)+1).tolist()+[len(epochs)]
    return list(zip(edges[:-1],edges[1:]))
class SleepWindows(Dataset):
    def __init__(self,subjects,norm,training=False,store=None):
        self.store=STORE if store is None else store; self.norm=norm; self.windows=[]
        stride=TRAIN_STRIDE if training else EVAL_STRIDE
        for sid in subjects:
            for a,b in continuous_runs(self.store[sid]['e']):
                starts=list(range(a,max(a+1,b-SEQ+1),stride))
                if b-a>SEQ and starts[-1]!=b-SEQ: starts.append(b-SEQ)
                self.windows.extend((sid,i,min(i+SEQ,b)) for i in starts)
        self.mean=np.asarray(norm['feature_mean'],np.float32); self.std=np.asarray(norm['feature_std'],np.float32)
    def __len__(self): return len(self.windows)
    def __getitem__(self,index):
        sid,a,b=self.windows[index]; d=self.store[sid]; n=b-a
        x=np.zeros((2,SEQ,2,750),np.float32); f=np.zeros((2,SEQ,len(FEATURE_NAMES)),np.float32)
        y=np.full(SEQ,-100,np.int64); q=np.zeros((SEQ,3),np.float32)
        for k,(view,feat) in enumerate([('w','fw'),('v','fv')]):
            raw=d[view][a:b]
            x[k,:n,0]=np.clip((raw-self.norm['ibi_mean'])/self.norm['ibi_std'],-8,8)
            # Preserve absolute IBI in channel 0; channel 1 exposes local variability.
            x[k,:n,1]=np.clip((raw-raw.mean(-1,keepdims=True))/.1,-8,8)
            f[k,:n]=np.clip((d[feat][a:b]-self.mean)/self.std,-8,8)
        y[:n]=TO3[d['y5'][a:b]]; q[:n]=d['q'][a:b]
        return dict(x=torch.from_numpy(x),f=torch.from_numpy(f),y=torch.from_numpy(y),
                    q=torch.from_numpy(q),length=n,index=index)

class ResidualBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(a,b,7,stride=2,padding=3,bias=False),nn.GroupNorm(4,b),nn.GELU(),
            nn.Conv1d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b))
        self.skip=nn.Conv1d(a,b,1,stride=2,bias=False)
    def forward(self,x): return F.gelu(self.net(x)+self.skip(x))
class StudentV6(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(ResidualBlock(2,16),ResidualBlock(16,32),ResidualBlock(32,64))
        self.signal_proj=nn.Linear(128,64)
        self.feature_proj=nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.LayerNorm(32),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(96,96),nn.LayerNorm(96),nn.GELU(),nn.Dropout(.2))
        self.rnn=nn.GRU(96,64,num_layers=2,batch_first=True,bidirectional=True,dropout=.15)
        self.norm=nn.LayerNorm(128); self.dropout=nn.Dropout(.25)
        self.head=nn.Linear(128,3); self.rem_head=nn.Linear(128,1)
    def forward(self,x,features,lengths):
        b,t=x.shape[:2]; lengths=lengths.to('cpu')
        mask=torch.arange(t,device=x.device)[None]<lengths.to(x.device)[:,None]
        z=self.cnn(x[mask]); z=torch.cat([z.mean(-1),z.std(-1,unbiased=False)],-1)
        z=self.signal_proj(z)
        fused=self.fuse(torch.cat([z,self.feature_proj(features[mask])],-1))
        h=fused.new_zeros(b,t,96); h[mask]=fused
        packed=pack_padded_sequence(h,lengths,batch_first=True,enforce_sorted=False)
        encoded,_=self.rnn(packed); encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=t)
        encoded=self.dropout(self.norm(encoded))
        return self.head(encoded),self.rem_head(encoded).squeeze(-1)
```

## Cell 15 — code

```python
def choose_batch():
    if BATCH: return int(BATCH)
    if not AMP: return 2
    free,_=torch.cuda.mem_get_info()
    return 4 if free<10*2**30 else 8 if free<20*2**30 else 16
def make_loader(data,order=None,batch=None):
    workers=NUM_WORKERS if os.name!='nt' else 0
    return DataLoader(data,batch_size=batch or choose_batch(),sampler=order,num_workers=workers,
        pin_memory=AMP,persistent_workers=False,generator=torch.Generator().manual_seed(SEED),
        **({'prefetch_factor':2} if workers else {}))
def move_batch(b):
    # Keep lengths on CPU for packed RNNs; transfer only arrays used by the GPU.
    return {k:(v.to(DEVICE,non_blocking=AMP) if k in ['x','f','y','q'] else v) for k,v in b.items()}
def weighted_ce(z,y):
    return F.cross_entropy(z,y,weight=CLASS_WEIGHTS)
def loss_components(model,b,run,epoch,step):
    mask=b['y']!=-100; y=b['y'][mask]
    # One primary forward pass, primarily on simulated 64 Hz. Original view appears 1/4 steps.
    view=0 if step%4==0 else 1
    logits,rem=model(b['x'][:,view],b['f'][:,view],b['length'])
    z=logits.float()[mask]; r=rem.float()[mask]
    ce=weighted_ce(z,y)
    rem_loss=F.binary_cross_entropy_with_logits(r,(y==2).float(),pos_weight=REM_POS_WEIGHT)
    loss=ce+run['rem_weight']*rem_loss
    kd=z.new_zeros(())
    if run['kd_weight']>0 and epoch>=8:
        q=b['q'][mask]; present=q.sum(-1)>.99
        gate=present&(q.argmax(-1)==y)&(q.max(-1).values>=.50)
        if gate.any():
            temp=2.0; soft=q[gate].clamp_min(1e-7).pow(1/temp); soft/=soft.sum(-1,keepdim=True)
            terms=F.kl_div(F.log_softmax(z[gate]/temp,-1),soft,reduction='none').sum(-1)
            # Same class weighting, not teacher confidence, prevents majority reliability domination.
            w=CLASS_WEIGHTS[y[gate]]; kd=(terms*w).sum()/w.sum()*temp**2
            loss=loss+run['kd_weight']*min(1.,(epoch-7)/5)*kd
    # Low-weight consistency only after supervised learning has started; every fourth step.
    js=z.new_zeros(())
    if epoch>=8 and step%4==0:
        other,_=model(b['x'][:,1-view],b['f'][:,1-view],b['length'])
        p=z.softmax(-1); q=other.float()[mask].softmax(-1); m=((p+q)/2).clamp_min(1e-7)
        js=.5*((p*(p.clamp_min(1e-7).log()-m.log())).sum(-1).mean()+
                 (q*(q.clamp_min(1e-7).log()-m.log())).sum(-1).mean())
        loss=loss+.03*min(1.,(epoch-7)/5)*js
    return loss,dict(ce=ce.detach(),rem_aux=rem_loss.detach(),kd=kd.detach(),js=js.detach())

@torch.inference_mode()
def predict_subjects(model,subjects,norm,batch=None,store=None):
    store=STORE if store is None else store; data=SleepWindows(subjects,norm,False,store)
    sums={s:np.zeros((len(store[s]['e']),3),np.float64) for s in subjects}
    counts={s:np.zeros(len(store[s]['e']),np.int64) for s in subjects}
    model.eval()
    for raw in make_loader(data,batch=batch):
        b=move_batch(raw)
        with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=AMP):
            z,_=model(b['x'][:,1],b['f'][:,1],b['length'])
        z=z.float().cpu().numpy()
        for j,index in enumerate(raw['index'].tolist()):
            sid,a,end=data.windows[index]; n=end-a
            sums[sid][a:end]+=z[j,:n]; counts[sid][a:end]+=1
    tables=[]
    for sid in subjects:
        assert np.all(counts[sid]>0), f'{sid}: prediction coverage gap'
        z=sums[sid]/counts[sid][:,None]; d=store[sid]
        t=pd.DataFrame(z,columns=['z_wake','z_nrem','z_rem'])
        t.insert(0,'epoch',d['e']); t.insert(0,'subject',sid); t['true']=TO3[d['y5']]
        tables.append(t)
    result=pd.concat(tables,ignore_index=True)
    assert not result.duplicated(['subject','epoch']).any()
    return result
def metrics_from_prob(y,p):
    p=np.asarray(p,np.float64); p=np.maximum(p,1e-12); p/=p.sum(1,keepdims=True)
    y=np.asarray(y,int); pred=p.argmax(1); f=f1_score(y,pred,labels=[0,1,2],average=None,zero_division=0)
    result=dict(epochs=len(y),accuracy=accuracy_score(y,pred),macro_f1=float(f.mean()),
        weighted_f1=f1_score(y,pred,labels=[0,1,2],average='weighted',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,pred),kappa=cohen_kappa_score(y,pred,labels=[0,1,2]),
        nll=log_loss(y,p,labels=[0,1,2]),brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))
    for i,name in enumerate(['wake','nrem','rem']):
        result[name+'_f1']=float(f[i]); result['predicted_'+name]=int((pred==i).sum()); result['true_'+name]=int((y==i).sum())
    result['rem_ap']=float(average_precision_score(y==2,p[:,2])) if (y==2).any() else None
    confidence=p.max(1); correct=(pred==y); ece=0.
    for lo in np.arange(0,1,.1):
        m=(confidence>=lo)&(confidence<(lo+.1) if lo<.89 else confidence<=1)
        if m.any(): ece+=m.mean()*abs(correct[m].mean()-confidence[m].mean())
    result['ece']=float(ece)
    return result
def raw_metrics(table):
    return metrics_from_prob(table.true,softmax(table[['z_wake','z_nrem','z_rem']].to_numpy(float),axis=1))
```

## Cell 16 — code

```python
# Full run identity: no model/decoder/prediction reuse solely because a filename exists.
EXPERIMENT=dict(revision=PIPELINE_REV,tag=RUN_TAG,split=SPLIT,prep=PREP,
    data={sid:METADATA[sid]['key'] for sid in sorted(STORE)},seq=SEQ,train_stride=TRAIN_STRIDE,eval_stride=EVAL_STRIDE,
    epochs=MAX_EPOCHS,minimum=MIN_EPOCHS,patience=PATIENCE,lr=LR,weight_decay=WEIGHT_DECAY,
    class_weights=weights,norm=NORM,runs=RUNS,batch_setting=BATCH,
    teacher_targets=digest({str(k):v.tolist() for k,v in TARGETS.items()}) if TARGETS else None)
EXPERIMENT_ID=digest(EXPERIMENT); ROOT=OUT/RUN_TAG
ROOT.mkdir(parents=True,exist_ok=True)
if (ROOT/'experiment.json').exists():
    assert digest(json.loads((ROOT/'experiment.json').read_text()))==EXPERIMENT_ID, 'Settings/data changed: select a new RUN_TAG or OUT. Existing results are preserved.'
else: save_json(EXPERIMENT,ROOT/'experiment.json')
def rng_state():
    return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state_all() if AMP else [])
def restore_rng(x):
    random.setstate(x['py']); np.random.set_state(x['np']); torch.set_rng_state(x['cpu'])
    if AMP and len(x['gpu'])==torch.cuda.device_count(): torch.cuda.set_rng_state_all(x['gpu'])
def train_run(run):
    seed_all(run['seed']); folder=ROOT/'models'/run['name']; folder.mkdir(parents=True,exist_ok=True)
    model=StudentV6().to(DEVICE); optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=MAX_EPOCHS,eta_min=LR*.1)
    scaler=torch.amp.GradScaler('cuda',enabled=AMP and AMP_DTYPE==torch.float16)
    state=dict(epoch=0,step=0,best=-1.,bad=0,done=False,batch=choose_batch(),history=[],loss_sum=0.,loss_n=0)
    if (folder/'last.pt').exists():
        checkpoint=load_own_checkpoint(folder/'last.pt')
        assert checkpoint['experiment_id']==EXPERIMENT_ID
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        state=checkpoint['state']; restore_rng(checkpoint['rng'])
    data=SleepWindows(SPLIT['train'],NORM,True)
    def snapshot():
        save_torch(dict(experiment_id=EXPERIMENT_ID,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),state=state,rng=rng_state(),norm=NORM),folder/'last.pt')
    if not (folder/'last.pt').exists(): snapshot()
    while not state['done'] and state['epoch']<MAX_EPOCHS:
        order=np.random.default_rng(run['seed']+state['epoch']).permutation(len(data)).tolist()
        start=state['step']; model.train(); t0=time.time()
        iterator=make_loader(data,order[start*state['batch']:],state['batch'])
        try:
            for step,raw in enumerate(tqdm(iterator,desc=f"{run['name']} epoch {state['epoch']+1}",leave=False),start=start):
                b=move_batch(raw); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=AMP):
                    loss,parts=loss_components(model,b,run,state['epoch'],step)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; inspect IBI scaling/features.')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
                state['step']=step+1; state['loss_sum']+=float(loss.detach()); state['loss_n']+=1
                if state['step']%SAVE_STEPS==0: snapshot()
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError('GPU memory exhausted. Close other GPU jobs and rerun to resume last checkpoint. If still needed, use a new RUN_TAG and smaller BATCH; partial-batch gradients are never reused.') from None
        table=predict_subjects(model,SPLIT['select'],NORM,state['batch']); m=raw_metrics(table)
        improved=m['macro_f1']>state['best']+1e-6
        state['bad']=0 if improved else state['bad']+1
        if improved:
            state['best']=m['macro_f1']
            save_torch(dict(experiment_id=EXPERIMENT_ID,model=model.state_dict(),norm=NORM,run=run,
                epoch=state['epoch']+1,selection_metrics=m),folder/'best.pt')
            save_csv(table,folder/'selection_logits.csv')
        row=dict(epoch=state['epoch']+1,loss=state['loss_sum']/max(state['loss_n'],1),
            lr=optimizer.param_groups[0]['lr'],seconds=time.time()-t0,**m)
        state['history'].append(row)
        print(run['name'],{k:round(row[k],4) if isinstance(row[k],float) else row[k] for k in
            ['epoch','loss','accuracy','macro_f1','wake_f1','nrem_f1','rem_f1','predicted_rem','rem_ap']})
        if m['predicted_rem']==0: print('REM collapse diagnostic: no REM predictions this epoch; inspect history and validation probabilities.')
        scheduler.step(); state.update(epoch=state['epoch']+1,step=0,loss_sum=0.,loss_n=0)
        state['done']=state['epoch']>=MAX_EPOCHS or (state['epoch']>=MIN_EPOCHS and state['bad']>=PATIENCE)
        snapshot(); save_csv(pd.DataFrame(state['history']),folder/'history.csv')
    # Recover the CSV even if a previous interruption occurred just after checkpoint saving.
    save_csv(pd.DataFrame(state['history']),folder/'history.csv')
    best=load_own_checkpoint(folder/'best.pt'); model.load_state_dict(best['model'])
    del optimizer,scaler,scheduler,data; gc.collect()
    return model,folder,best
```

## Cell 17 — code

```python
# Expensive cell: new student training (or version-checked resume). CPU fallback is supported.
candidate_rows=[]
for run in RUNS:
    model,folder,best=train_run(run)
    candidate_rows.append(dict(name=run['name'],best_epoch=best['epoch'],**best['selection_metrics']))
    del model,best; gc.collect()
    if AMP: torch.cuda.empty_cache()
selection=pd.DataFrame(candidate_rows).sort_values(['macro_f1','rem_f1','accuracy'],ascending=False)
save_csv(selection,ROOT/'candidate_selection.csv'); print(selection.to_string(index=False))
CHOSEN=str(selection.iloc[0]['name']); chosen_folder=ROOT/'models'/CHOSEN
best=load_own_checkpoint(chosen_folder/'best.pt'); student=StudentV6().to(DEVICE); student.load_state_dict(best['model'])
MODEL_SHA=file_sha(chosen_folder/'best.pt')
if selection.iloc[0]['rem_f1']==0: warnings.warn('The selected raw model still has REM collapse. The notebook will report it; a fix is not established merely by training.')
print('Frozen model choice:',CHOSEN,'| model hash:',MODEL_SHA[:12])
```

## Cell 18 — explanation

## Validation-only calibration and class decisions

Model selection is now finished. On the separate 8 calibration people, fit scalar temperature for probability calibration, then search a small predeclared Wake/REM log-probability offset grid. The selected offsets must retain calibration accuracy within **0.5 percentage points** of the temperature-only model and improve macro F1. Zero offsets remain a candidate. This constraint is a development safeguard, **not a promise about test accuracy**. Report both raw decisions and adjusted decisions; offsets can improve recall at a precision cost. No test metric selects a model, offset, epoch, loss or threshold.


## Cell 19 — code

```python
def fit_decoder(table):
    z=table[['z_wake','z_nrem','z_rem']].to_numpy(np.float64); y=table.true.to_numpy(int)
    # Bounded temperature grid is stable, deterministic, and sufficient for this 3-class calibration.
    temperatures=np.unique(np.r_[1.,np.geomspace(.3,5.,80)])
    temperature=float(min(temperatures,key=lambda t:log_loss(y,softmax(z/t,axis=1),labels=[0,1,2])))
    raw_p=softmax(z/temperature,axis=1); base=metrics_from_prob(y,raw_p)
    best_key=(base['macro_f1'],base['accuracy'],0.)
    offsets=np.zeros(3); rows=[]
    for wake in np.arange(-1.,1.01,.25):
        for rem in np.arange(-1.,1.01,.25):
            bias=np.array([wake,0.,rem]); m=metrics_from_prob(y,softmax(z/temperature+bias,axis=1))
            admissible=m['accuracy']>=base['accuracy']-.005
            key=(m['macro_f1'],m['accuracy'],-float(np.square(bias).sum()))
            rows.append(dict(wake_offset=wake,rem_offset=rem,admissible=admissible,**m))
            if admissible and key>best_key: best_key=key; offsets=bias
    return dict(temperature=temperature,offsets=offsets.tolist(),raw_calibration=base,
        adjusted_calibration=metrics_from_prob(y,softmax(z/temperature+offsets,axis=1)),
        accuracy_tolerance=.005),pd.DataFrame(rows)
decoder_key=digest(dict(experiment=EXPERIMENT_ID,model=MODEL_SHA,norm=NORM,
    calibration={sid:METADATA[sid]['key'] for sid in SPLIT['calibrate']},
    method='temperature-grid80-plus1_offsets-minus1-to1-step025_accuracy-tolerance0005'))
decoder_path=ROOT/'decoder.json'
if decoder_path.exists():
    DECODER=json.loads(decoder_path.read_text()); assert DECODER['key']==decoder_key, 'Stale decoder: use new RUN_TAG.'
    calibration_table=pd.read_csv(ROOT/'calibration_logits.csv') if (ROOT/'calibration_logits.csv').exists() else predict_subjects(student,SPLIT['calibrate'],NORM)
else:
    calibration_table=predict_subjects(student,SPLIT['calibrate'],NORM)
    DECODER,grid=fit_decoder(calibration_table); DECODER.update(key=decoder_key,model_sha=MODEL_SHA,chosen=CHOSEN)
    save_json(DECODER,decoder_path); save_csv(grid,ROOT/'calibration_offset_search.csv')
save_csv(calibration_table,ROOT/'calibration_logits.csv')
print(json.dumps(DECODER,indent=2))
save_json(dict(experiment_id=EXPERIMENT_ID,chosen=CHOSEN,model_sha=MODEL_SHA,decoder=DECODER,
               test_subjects=SPLIT['test'],policy='No selection using final test results'),ROOT/'frozen_selection.json')

# Diagnostic plots use development data only.
fig,axes=plt.subplots(1,3,figsize=(15,4))
for run in RUNS:
    h=pd.read_csv(ROOT/'models'/run['name']/'history.csv')
    axes[0].plot(h.epoch,h.macro_f1,label=run['name']); axes[1].plot(h.epoch,h.rem_f1,label=run['name'])
z=calibration_table[['z_wake','z_nrem','z_rem']].to_numpy(float)
p=softmax(z/DECODER['temperature'],axis=1)
for i,name in enumerate(CLASS_NAMES):
    axes[2].hist(p[calibration_table.true.to_numpy()==i,2],bins=25,alpha=.5,density=True,label='True '+name)
axes[0].set_title('Selection macro F1'); axes[1].set_title('Selection REM F1'); axes[2].set_title('Calibration REM probability')
for ax in axes: ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ROOT/'development_diagnostics.png',dpi=150); plt.show()
```

## Cell 20 — code

```python
# Final benchmark is opened only after the model and decoder are frozen.
for sid in tqdm(SPLIT['test'],desc='Prepare fixed historical test'):
    d,meta=prepare_subject(sid,64); d['fw'],d['fv']=ibi_features(d['w']),ibi_features(d['v'])
    d['q']=np.zeros((len(d['e']),3),np.float32); STORE[sid],METADATA[sid]=d,meta
PRED=ROOT/'predictions'; PRED.mkdir(parents=True,exist_ok=True)
for sid in tqdm(SPLIT['test'],desc='Evaluate chosen student'):
    key=digest(dict(model=MODEL_SHA,decoder=DECODER,data=METADATA[sid]['key'],experiment=EXPERIMENT_ID))
    path=PRED/f'{sid}.csv'; mp=PRED/f'{sid}.json'
    if path.exists() and mp.exists() and json.loads(mp.read_text()).get('key')==key:
        t=pd.read_csv(path)
        assert len(t)==len(STORE[sid]['e']) and np.array_equal(t.epoch.to_numpy(),STORE[sid]['e'])
        continue
    t=predict_subjects(student,[sid],NORM)
    z=t[['z_wake','z_nrem','z_rem']].to_numpy(np.float64)/DECODER['temperature']
    raw=softmax(z,axis=1); adjusted=softmax(z+np.array(DECODER['offsets']),axis=1)
    for j,name in enumerate(['wake','nrem','rem']): t['raw_p_'+name]=raw[:,j]; t['p_'+name]=adjusted[:,j]
    t['raw_predicted']=raw.argmax(1); t['predicted']=adjusted.argmax(1); t['epoch_start_seconds']=30*t.epoch
    save_csv(t,path); save_json(dict(key=key,subject=sid),mp)
all_predictions=pd.concat([pd.read_csv(PRED/f'{s}.csv') for s in SPLIT['test']],ignore_index=True)
assert not all_predictions.duplicated(['subject','epoch']).any()
assert len(all_predictions)==sum(len(STORE[s]['e']) for s in SPLIT['test'])
save_csv(all_predictions,ROOT/'predictions.csv')
```

## Cell 21 — code

```python
def report_predictions(table,prefix=''):
    p=table[[prefix+'p_wake',prefix+'p_nrem',prefix+'p_rem']].to_numpy(np.float64)
    p=np.maximum(p,1e-12); p/=p.sum(1,keepdims=True)
    return metrics_from_prob(table.true,p)
summaries=[]
for prefix,name in [('raw_','temperature_only_argmax'),('','validation_adjusted')]:
    m=report_predictions(all_predictions,prefix); summaries.append(dict(method=name,**m))
    pred=all_predictions[[prefix+'p_wake',prefix+'p_nrem',prefix+'p_rem']].to_numpy().argmax(1)
    report=pd.DataFrame(classification_report(all_predictions.true,pred,labels=[0,1,2],
        target_names=CLASS_NAMES,output_dict=True,zero_division=0)).T
    save_csv(report.reset_index(names='class'),ROOT/f'{name}_class_metrics.csv')
    cm=confusion_matrix(all_predictions.true,pred,labels=[0,1,2])
    frame=pd.DataFrame(cm,columns=['pred_'+c for c in CLASS_NAMES]); frame.insert(0,'true_class',CLASS_NAMES)
    save_csv(frame,ROOT/f'{name}_confusion_matrix.csv')
    print(name); print(report.to_string())
summary=pd.DataFrame(summaries); save_csv(summary,ROOT/'metrics.csv'); print(summary.to_string(index=False))
per_subject=pd.DataFrame([dict(subject=s,method=name,**report_predictions(g,prefix))
    for s,g in all_predictions.groupby('subject') for prefix,name in [('raw_','raw'),('','adjusted')]])
save_csv(per_subject,ROOT/'subject_metrics.csv')

qc=[]
for sid,meta in METADATA.items():
    counts=meta['qc']; total=sum(counts.values())
    qc.append(dict(subject=sid,rate=meta['raw']['rate'],source=meta['source'],
        total_epochs=total,retained_fraction=counts.get('kept',0)/max(total,1),**counts))
save_csv(pd.DataFrame(qc).fillna(0),ROOT/'quality_control.csv')
```

## Cell 22 — code

```python
# Subject-cluster bootstrap: resample whole people, not correlated epochs.
def bootstrap_intervals(table,pred_col='predicted',n=1000,seed=SEED):
    rng=np.random.default_rng(seed)
    mats=np.stack([confusion_matrix(g.true,g[pred_col],labels=[0,1,2]) for _,g in table.groupby('subject')])
    values=[]
    for _ in range(n):
        m=mats[rng.integers(0,len(mats),len(mats))].sum(0); den=m.sum(0)+m.sum(1)
        f=np.divide(2*np.diag(m),den,out=np.zeros(3,dtype=float),where=den>0)
        values.append([np.trace(m)/m.sum(),f.mean(),f[2]])
    bounds=np.quantile(values,[.025,.975],axis=0)
    return pd.DataFrame(dict(metric=['accuracy','macro_f1','rem_f1'],lower_95=bounds[0],upper_95=bounds[1],samples=n))
interval=bootstrap_intervals(all_predictions); save_csv(interval,ROOT/'confidence_intervals.csv'); print(interval)

# V5 comparison is strictly post-evaluation; no result here changes model/decoder selection.
V5_REPORTED=dict(accuracy=.720515,macro_f1=.472204,weighted_f1=.664222,rem_f1=0.)
print('V5 reported historical baseline:',V5_REPORTED)
baseline_path=V5/'predictions.csv'
if baseline_path.exists():
    old=pd.read_csv(baseline_path)
    old['epoch']=np.rint(old.epoch_start_seconds/30).astype(np.int64)
    assert not old.duplicated(['subject','epoch']).any()
    pair=all_predictions.merge(old[['subject','epoch','true','predicted']],on=['subject','epoch'],suffixes=('','_v5'),validate='one_to_one')
    assert len(pair)>0 and np.array_equal(pair.true,pair.true_v5), 'V5/V6 label or alignment mismatch'
    rows=[]
    for col,name in [('predicted_v5','V5'),('raw_predicted','V6 raw'),('predicted','V6 adjusted')]:
        y=pair.true.to_numpy(); p=pair[col].to_numpy(); f=f1_score(y,p,labels=[0,1,2],average=None,zero_division=0)
        rows.append(dict(model=name,paired_epochs=len(pair),accuracy=accuracy_score(y,p),macro_f1=f.mean(),rem_f1=f[2]))
    comparison=pd.DataFrame(rows); save_csv(comparison,ROOT/'v5_v6_matched_comparison.csv'); print(comparison.to_string(index=False))
    gain=comparison.iloc[2][['accuracy','macro_f1','rem_f1']].to_numpy(float)-comparison.iloc[0][['accuracy','macro_f1','rem_f1']].to_numpy(float)
    print('All three paired metrics improved:',bool(np.all(gain>0)),'| deltas accuracy / macro F1 / REM F1:',gain)
else:
    print('No V5 prediction file: cannot establish a matched-epoch improvement. Compare reported aggregates with coverage caveats.')
if (all_predictions.predicted==2).sum()==0:
    warnings.warn('REM remains unpredicted: the measured result does not establish a fix. Inspect class supports, teacher audit, REM AP, confusion and quality-control output before further experiments.')
print('Saved results:',ROOT)
```

## Cell 23 — explanation

## Reuse and next run

- Rerun all cells after a Colab reconnect. Validated V6 subject caches are reused. The version-checked `last.pt` resumes training near its last saved minibatch; unsaved minibatches repeat. `best.pt` is selected by macro F1 on selection subjects, with REM F1 and accuracy breaking candidate ties.
- V5 raw-cache reuse requires matching raw-file size/mtime/path, label/column/interval settings, original split and S097 policy. Otherwise preprocessing runs again from raw CSVs. V5 student weights, optimizer, temperatures and predictions never initialize V6. Keep the old output directory unchanged.
- Optional KD reuses only alignment-checked OOF targets with exclusion manifests. Missing/incompatible teachers do not block supervised training. There is no automatic expensive five-teacher rebuild; the default remedy deliberately tests whether the student improves without KD.
- New architecture/loss/data settings need a new `RUN_TAG` or output directory; this avoids silently mixing experiments. Keep raw CSVs and all output files on persistent storage. Processes and GPU RAM do not survive a reconnect.
- For future inference, recreate `StudentV6`, load the chosen own `best.pt`, its train-only normalization and `decoder.json`, then apply the same `make_ibi`, feature extraction and overlapping prediction path. Use the 64 Hz view. These functions require sleep labels only for this evaluation dataset wrapper; deployment should supply inputs without ground-truth labels.

**Read the outputs before claiming success:** `candidate_selection.csv`, both class reports and confusion matrices, `metrics.csv`, `subject_metrics.csv`, `quality_control.csv`, `confidence_intervals.csv` and (when V5 predictions exist) `v5_v6_matched_comparison.csv`. A higher REM recall alone does not prove higher REM F1 or accuracy. If REM average precision stays near prevalence, the IBI representation may not separate REM sufficiently; additional wearable modalities or external training data are a separate experiment, not something a threshold can guarantee to repair.

