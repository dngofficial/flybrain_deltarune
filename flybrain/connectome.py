"""Reproducible induced subgraph from Janelia MaleCNS v1.0; ~1.11 GB source download."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES={"annotations":"body-annotations-male-cns-v1.0-minconf-0.5.feather","nt":"body-neurotransmitters-male-cns-v1.0.feather","weights":"connectome-weights-male-cns-v1.0-minconf-0.5.feather"}
DEFAULT_GRAPH=Path(__file__).parent/"data"/"male_cns_visual_1024.npz"
def checksum(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def download(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    for name in FILES.values():
        p=cache/name
        if not p.exists():
            print("Downloading",name,flush=True)
            temp=p.with_suffix(".partial")
            urllib.request.urlretrieve(BASE+name,temp);temp.replace(p)
    return cache
def batches(path):
    import pyarrow as pa
    with pa.memory_map(str(path),"r") as f:
        reader=pa.ipc.open_file(f)
        for i in range(reader.num_record_batches):
            b=reader.get_batch(i)
            yield {k:b.column(k).to_numpy(zero_copy_only=False) for k in ("body_pre","body_post","weight")}
def build(cache,out,max_neurons=1024,seed_types=("LC4","LPLC2","DNp01","DNg13"),min_weight=2):
    import pyarrow.feather as feather
    cache=Path(cache)
    ann=feather.read_table(cache/FILES["annotations"]).to_pandas()
    ann=ann[ann.superclass.notna()&ann.type.notna()&(ann.status=="Traced")]
    ann=ann.sort_values("bodyId").drop_duplicates("bodyId").set_index("bodyId")
    seeds=ann.index[ann.type.isin(seed_types)].to_numpy(np.int64)
    if not len(seeds):raise ValueError("No matching seed cell types")
    if max_neurons and len(seeds)>max_neurons:raise ValueError("Limit must include all seed neurons")
    valid=ann.index.to_numpy(np.int64)
    if max_neurons:
        scores=np.zeros(len(valid),float)
        print("Selecting neighbours of",len(seeds),"seed cells",flush=True)
        for b in batches(cache/FILES["weights"]):
            pre,post,w=b["body_pre"],b["body_post"],b["weight"]
            for src,dst in ((pre,post),(post,pre)):
                take=np.isin(src,seeds)&(w>=min_weight)
                ids,values=dst[take],w[take]
                idx=np.searchsorted(valid,ids);good=idx<len(valid)
                good[good]&=valid[idx[good]]==ids[good]
                np.add.at(scores,idx[good],values[good])
        chosen=list(map(int,seeds));used=set(chosen)
        for i in np.argsort(-scores,kind="stable"):
            if scores[i]<=0 or len(chosen)>=max_neurons:break
            ident=int(valid[i])
            if ident not in used:chosen.append(ident);used.add(ident)
        ids=np.array(sorted(chosen),np.int64)
    else:ids=valid
    pre_out=[];post_out=[];weights=[]
    print("Extracting induced edges among",len(ids),"cells",flush=True)
    for b in batches(cache/FILES["weights"]):
        p,q,w=b["body_pre"],b["body_post"],b["weight"]
        take=(w>=min_weight)&np.isin(p,ids)&np.isin(q,ids)
        if take.any():
            pre_out.append(np.searchsorted(ids,p[take]).astype(np.int32))
            post_out.append(np.searchsorted(ids,q[take]).astype(np.int32))
            weights.append(w[take].astype(np.float32))
    if not weights:raise ValueError("Graph is empty")
    cells=ann.loc[ids]
    nt=feather.read_table(cache/FILES["nt"],columns=["body","consensus_nt"]).to_pandas().set_index("body").reindex(ids)
    transmitters=nt.consensus_nt.fillna("unknown").to_numpy(str)
    signs=np.where(np.isin(transmitters,["gaba","glutamate"]),-1.,1.).astype(np.float32)
    positions=np.array([p if isinstance(p,(list,np.ndarray)) and len(p)==3 else [np.nan]*3 for p in cells.somaLocation],np.float32)
    info=dict(kind="real_connectome_subset" if max_neurons else "real_connectome",dataset="male-cns:v1.0",neurons=len(ids),edges=sum(map(len,weights)),synapse_count=int(sum(x.sum(dtype=np.float64) for x in weights)),seed_types=list(seed_types),min_weight=min_weight,
        selection="Traced typed seeds plus strongest summed incoming/outgoing one-hop neighbours; induced directed graph; stable body-ID ties",
        weight_model="log1p(count), incoming absolute normalization; GABA/glutamate negative, other/unknown positive (modeling assumption)",
        source="https://male-cns.janelia.org/download/",license="CC-BY-4.0",attribution="FlyEM (HHMI Janelia), University of Cambridge, MRC Laboratory of Molecular Biology, Google Research; MaleCNS v1.0",
        source_files={k:dict(url=BASE+v,sha256=checksum(cache/v)) for k,v in FILES.items()})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,ids=ids,pre=np.concatenate(pre_out),post=np.concatenate(post_out),weight=np.concatenate(weights),sign=signs,types=cells.type.to_numpy(str),classes=cells.superclass.to_numpy(str),positions=positions,transmitters=transmitters,metadata=np.array(json.dumps(info)))
    out.with_suffix(".json").write_text(json.dumps(info,indent=2)+"\n")
    print({k:info[k] for k in ("neurons","edges","synapse_count")},flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",default="connectome_cache");p.add_argument("--out",default=str(DEFAULT_GRAPH))
    p.add_argument("--download",action="store_true");p.add_argument("--max-neurons",type=int,default=1024)
    p.add_argument("--min-weight",type=int,default=2);p.add_argument("--seed-types",nargs="+",default=["LC4","LPLC2","DNp01","DNg13"])
    a=p.parse_args()
    if a.max_neurons<0 or a.min_weight<1:p.error("Use nonnegative neuron limit and positive minimum weight")
    if a.download:download(a.cache)
    build(a.cache,a.out,a.max_neurons,a.seed_types,a.min_weight)
if __name__=="__main__":main()
