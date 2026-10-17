import os, json, logging, numpy as np
logger = logging.getLogger(__name__)

def compute_confidence(out_dir, mesh, view_images=None):
    try:
        import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
        from reconstructor import load_from_dir,_project,IMG_SIZE,FG_ALPHA_THRESHOLD
        from PIL import Image as PI
        views = load_from_dir(out_dir)
        if not views: return _write_uniform(out_dir, mesh, 0.5)
        verts = np.array(mesh.vertices, dtype=np.float32)
        N = len(verts); conf = np.zeros(N, dtype=np.float32); S=IMG_SIZE
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
            conf[sel[fg[vi[sel],ui[sel]]]]=1.0
        result = {'vertex_confidence': conf.tolist()}
        p=os.path.join(out_dir,'confidence_map.json')
        with open(p,'w',encoding='utf-8') as f: json.dump(result,f)
        logger.info('  Confidence map written: %d/%d visible', int((conf==1.0).sum()), N)
        return result
    except Exception as e:
        logger.warning('confidence_mapper failed: %s', e)
        return _write_uniform(out_dir, mesh, 0.5)

def _write_uniform(out_dir, mesh, val):
    N=len(mesh.vertices) if mesh is not None else 0
    r={'vertex_confidence':[val]*N}
    try:
        with open(os.path.join(out_dir,'confidence_map.json'),'w',encoding='utf-8') as f: json.dump(r,f)
    except: pass
    return r