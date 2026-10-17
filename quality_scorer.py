import os, json, logging, numpy as np
logger = logging.getLogger(__name__)

def compute_scores(out_dir, mesh=None):
    import trimesh
    mp = os.path.join(out_dir,'model_repaired.glb')
    if not os.path.exists(mp): mp = os.path.join(out_dir,'model.glb')
    if not os.path.exists(mp): return _empty_scores()
    if mesh is None:
        try: mesh = trimesh.load(mp, force='mesh')
        except Exception as e: logger.warning('qs load: %s', e); return _empty_scores()
    nv=len(mesh.vertices); nf=len(mesh.faces)
    try: degen=int((mesh.area_faces<1e-10).sum())
    except: degen=0
    geometry=int(min(nv/50000,1.0)*50+min(nf/100000,1.0)*50)
    wt=bool(mesh.is_watertight)
    try:
        ae=np.sort(np.vstack([mesh.faces[:,[0,1]],mesh.faces[:,[1,2]],mesh.faces[:,[2,0]]]),axis=1)
        _,cnt=np.unique(ae,axis=0,return_counts=True)
        nm=(cnt>2).any(); holes=(cnt==1).any()
    except: nm=False; holes=not wt
    surface=(50 if wt else 0)+(25 if not nm else 0)+(25 if not holes else 0)
    try: nc=len(mesh.split(only_watertight=False))
    except: nc=1
    topology=max(0,100-(nc-1)*10)
    has_uv=os.path.exists(os.path.join(out_dir,'model_uv_textured.glb'))
    has_vc=_has_vc(mesh)
    texture=100 if has_uv else (50 if has_vc else 0)
    comp=_completeness(mesh,out_dir)
    overall=int(np.clip(round(0.20*geometry+0.25*surface+0.20*topology+0.15*texture+0.20*comp),0,100))
    scores={'geometry':geometry,'surface':surface,'texture':texture,'topology':topology,'completeness':comp,'overall':overall}
    qr=os.path.join(out_dir,'quality_report.json')
    try:
        d=json.load(open(qr)) if os.path.exists(qr) else {}
        d['scores']=scores; d.setdefault('watertight',wt)
        with open(qr,'w',encoding='utf-8') as f: json.dump(d,f,indent=2)
    except Exception as e: logger.warning('qs QR: %s',e)
    logger.info('  QualityScores: %s',scores)
    return scores

def _empty_scores():
    return {'geometry':0,'surface':0,'texture':0,'topology':0,'completeness':0,'overall':0}

def _has_vc(mesh):
    try:
        vc=getattr(getattr(mesh,'visual',None),'vertex_colors',None)
        if vc is None: return False
        vc=np.asarray(vc)
        return vc.ndim==2 and vc.shape[0]==len(mesh.vertices)
    except: return False

def _completeness(mesh, out_dir):
    try:
        import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
        from reconstructor import load_from_dir,_project,IMG_SIZE,FG_ALPHA_THRESHOLD
        from PIL import Image as PI
        views=load_from_dir(out_dir)
        if not views: return 50
        verts=np.array(mesh.vertices,dtype=np.float32)
        vis=np.zeros(len(verts),dtype=bool); S=IMG_SIZE
        for v in views:
            if v.get('mask_path') and os.path.exists(v['mask_path']):
                fg=np.array(PI.open(v['mask_path']).convert('L').resize((S,S)))>=FG_ALPHA_THRESHOLD
            elif v.get('alpha_mask') is not None: fg=v['alpha_mask']
            else: continue
            u,vp,d,ok=_project(verts,v['az'],v['el'])
            ui=np.round(u).astype(np.int32); vi=np.round(vp).astype(np.int32)
            ii=ok&(ui>=0)&(ui<S)&(vi>=0)&(vi<S)
            sel=np.where(ii)[0]
            if not len(sel): continue
            vis[sel[fg[vi[sel],ui[sel]]]]=True
        return int(np.clip(round(vis.mean()*100),0,100))
    except Exception as e: logger.debug('completeness: %s',e); return 50