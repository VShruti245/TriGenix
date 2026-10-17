import os, logging
logger = logging.getLogger(__name__)

def run_hollow(out_dir, wall_mm=3.0, drain_diameter_mm=4.0):
    import trimesh, numpy as np
    mp=os.path.join(out_dir,'model_repaired.glb')
    if not os.path.exists(mp): mp=os.path.join(out_dir,'model.glb')
    if not os.path.exists(mp): raise FileNotFoundError('No model found')
    mesh=trimesh.load(mp,force='mesh')
    if not mesh.is_watertight: raise ValueError('non-watertight input: cannot hollow safely')
    if wall_mm < 1.0: raise ValueError('wall_mm must be >= 1.0')
    if drain_diameter_mm < 2.0: raise ValueError('drain_diameter_mm must be >= 2.0')
    try:
        # Use trimesh.boolean if available (requires manifold3d or blender)
        inner=mesh.copy()
        scale=1.0-(wall_mm/max(mesh.bounding_box.extents))*2
        if scale <= 0: raise ValueError('wall_mm too large for this model')
        inner.apply_scale(scale)
        hollow=trimesh.boolean.difference([mesh,inner])
        if hollow is None or len(hollow.faces)<100:
            raise ValueError('boolean difference produced invalid result')
        # Add drain holes (cylinder subtraction)
        bb=mesh.bounding_box.bounds
        r=drain_diameter_mm/2.0; h=(bb[1][2]-bb[0][2])*1.1
        cx=[(bb[0][0]+bb[1][0])/2+bb[1][0]*0.25, (bb[0][0]+bb[1][0])/2-bb[1][0]*0.25]
        cy=(bb[0][1]+bb[1][1])/2
        for x in cx:
            cyl=trimesh.creation.cylinder(radius=r,height=h)
            cyl.apply_translation([x,cy,bb[0][2]+h/2])
            hollow=trimesh.boolean.difference([hollow,cyl])
        out_path=os.path.join(out_dir,'model_hollow.stl')
        hollow.export(out_path)
        logger.info('Hollowing complete: %s', out_path)
        stl_url='/'+os.path.relpath(out_path).replace(os.sep,'/')
        return {'ok':True,'stl_url':stl_url}
    except ImportError:
        raise ValueError('boolean operations require manifold3d: pip install manifold3d')