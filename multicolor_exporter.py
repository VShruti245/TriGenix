import os, logging, numpy as np
logger = logging.getLogger(__name__)

def run_multicolor_export(out_dir, fmt='3mf'):
    import trimesh
    mp=os.path.join(out_dir,'model_repaired.glb')
    if not os.path.exists(mp): mp=os.path.join(out_dir,'model.glb')
    if not os.path.exists(mp): raise FileNotFoundError('No model found')
    mesh=trimesh.load(mp,force='mesh')
    # Get vertex colors
    vc=getattr(getattr(mesh,'visual',None),'vertex_colors',None)
    if vc is None:
        try:
            import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
            from reconstructor import paint_colors
            from reconstructor import load_from_dir
            views=load_from_dir(out_dir)
            tex=os.path.join(out_dir,'model_textured.glb')
            mesh=paint_colors(mesh,views,triposr_glb_path=tex if os.path.exists(tex) else None)
            vc=np.array(mesh.visual.vertex_colors)[:,:3].astype(np.float32)
        except Exception as e:
            raise ValueError(f'No vertex colors and paint_colors failed: {e}')
    else:
        vc=np.array(vc)[:,:3].astype(np.float32)
    # Cluster face colors by majority vertex color
    from sklearn.cluster import MiniBatchKMeans
    k=min(8,max(3,int(len(vc)**0.2)))
    km=MiniBatchKMeans(n_clusters=k,random_state=42,n_init=3)
    labels=km.fit_predict(vc)
    face_labels=np.array([np.bincount(labels[mesh.faces[i]]).argmax() for i in range(len(mesh.faces))])
    files=[]
    if fmt=='stl_per_region':
        out_dir_mc=os.path.join(out_dir,'multicolor'); os.makedirs(out_dir_mc,exist_ok=True)
        for cl in range(k):
            mask=face_labels==cl
            if not mask.any(): continue
            sub=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces[mask],process=False)
            p=os.path.join(out_dir_mc,f'region_{cl+1}.stl')
            sub.export(p); files.append(p)
    elif fmt=='3mf':
        try:
            mesh.export(os.path.join(out_dir,'model_multicolor.3mf'))
            files.append(os.path.join(out_dir,'model_multicolor.3mf'))
        except Exception as e:
            raise ValueError(f'3MF export failed: {e}')
    urls=['/'+os.path.relpath(p).replace(os.sep,'/') for p in files]
    logger.info('Multicolor export: %d files', len(files))
    return {'ok':True,'files':urls}