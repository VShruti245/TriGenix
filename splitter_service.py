import os, math, logging
logger = logging.getLogger(__name__)

def compute_section_count(bbox_x, bbox_y, bed_w, bed_d):
    return math.ceil(bbox_x/bed_w), math.ceil(bbox_y/bed_d)

def run_split(out_dir, bed_width_mm, bed_depth_mm, connector_type='pin'):
    import trimesh, numpy as np
    mp=os.path.join(out_dir,'model_repaired.glb')
    if not os.path.exists(mp): mp=os.path.join(out_dir,'model.glb')
    if not os.path.exists(mp): raise FileNotFoundError('No model found')
    mesh=trimesh.load(mp,force='mesh')
    bb=mesh.bounding_box.extents
    nx,ny=compute_section_count(bb[0],bb[1],bed_width_mm,bed_depth_mm)
    if nx==1 and ny==1: raise ValueError('Model fits on bed; splitting not needed')
    out_dir_split=os.path.join(out_dir,'split')
    os.makedirs(out_dir_split,exist_ok=True)
    bounds=mesh.bounds; parts=[]; n=0
    x_cuts=np.linspace(bounds[0][0],bounds[1][0],nx+1)
    y_cuts=np.linspace(bounds[0][1],bounds[1][1],ny+1)
    for i in range(nx):
        for j in range(ny):
            n+=1
            sub=mesh.copy()
            for plane_origin,plane_normal in [
                ([x_cuts[i],0,0],[1,0,0]),([x_cuts[i+1],0,0],[-1,0,0]),
                ([0,y_cuts[j],0],[0,1,0]),([0,y_cuts[j+1],0],[0,-1,0])]:
                try: sub=trimesh.intersections.slice_mesh_plane(sub,plane_normal,plane_origin)
                except: pass
            if sub is not None and len(sub.faces)>0:
                ppath=os.path.join(out_dir_split,f'part_{n}.stl')
                sub.export(ppath); parts.append(ppath)
    urls=['/'+os.path.relpath(p).replace(os.sep,'/') for p in parts]
    logger.info('Split into %d parts', len(parts))
    return {'ok':True,'part_count':len(parts),'parts':urls}