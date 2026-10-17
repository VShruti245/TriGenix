import os, json, logging, numpy as np
logger = logging.getLogger(__name__)

def classify_vertex(thickness, min_thickness_mm):
    return 'thin' if thickness < min_thickness_mm else 'adequate'

def run_thickness_check(out_dir, min_thickness_mm=1.5):
    import trimesh
    mp=os.path.join(out_dir,'model_repaired.glb')
    if not os.path.exists(mp): mp=os.path.join(out_dir,'model.glb')
    if not os.path.exists(mp): raise FileNotFoundError(f'No model in {out_dir}')
    mesh=trimesh.load(mp, force='mesh')
    verts=np.array(mesh.vertices,dtype=np.float32)
    normals=np.array(mesh.vertex_normals,dtype=np.float32)
    N=len(verts); thicknesses=np.full(N,float('inf'))
    try:
        ray_origins=verts - normals*0.001
        ray_dirs=-normals
        locs,ray_idx,_=mesh.ray.intersects_location(ray_origins=ray_origins,ray_directions=ray_dirs,multiple_hits=False)
        if len(ray_idx):
            dists=np.linalg.norm(locs-ray_origins[ray_idx],axis=1)
            np.minimum.at(thicknesses, ray_idx, dists)
    except Exception as e:
        logger.warning('thickness ray-cast failed: %s', e)
    thin_mask=thicknesses<min_thickness_mm
    thin_count=int(thin_mask.sum())
    total_count=N
    thin_fraction=float(thin_count/max(1,total_count))
    # cap inf values for JSON serialisation
    t_list=[float(min(t,9999.0)) for t in thicknesses]
    report={'min_thickness_mm':min_thickness_mm,'thin_vertex_count':thin_count,
            'total_vertex_count':total_count,'thin_fraction':thin_fraction,'vertex_thickness':t_list}
    rp=os.path.join(out_dir,'thickness_report.json')
    with open(rp,'w',encoding='utf-8') as f: json.dump(report,f)
    logger.info('  Thickness: thin=%d/%d (%.1f%%)', thin_count, total_count, thin_fraction*100)
    return report