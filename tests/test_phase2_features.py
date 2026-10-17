import os, sys, json, time, tempfile, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

# ── Quality scorer ────────────────────────────────────────────────────────

def test_quality_scorer_valid_structure():
    from quality_scorer import _empty_scores
    e = _empty_scores()
    assert set(e.keys()) == {'geometry','surface','texture','topology','completeness','overall'}
    for v in e.values():
        assert isinstance(v,int) and 0<=v<=100

def test_quality_scorer_missing_model():
    from quality_scorer import compute_scores
    with tempfile.TemporaryDirectory() as td:
        s = compute_scores(td)
    assert s['overall'] == 0

def test_quality_scorer_icosphere():
    import trimesh
    from quality_scorer import compute_scores
    with tempfile.TemporaryDirectory() as td:
        m = trimesh.creation.icosphere(subdivisions=3)
        m.export(os.path.join(td,'model.glb'))
        s = compute_scores(td, mesh=m)
    assert 0 <= s['overall'] <= 100
    assert s['surface'] == 100
    assert s['geometry'] > 0

def test_quality_scorer_writes_report():
    import trimesh
    from quality_scorer import compute_scores
    with tempfile.TemporaryDirectory() as td:
        trimesh.creation.icosphere(subdivisions=2).export(os.path.join(td,'model.glb'))
        compute_scores(td)
        qr = json.load(open(os.path.join(td,'quality_report.json')))
    assert 'scores' in qr and 'overall' in qr['scores']

# ── Cleanup worker ────────────────────────────────────────────────────────

def test_is_stale():
    from cleanup_worker import is_stale
    n = time.time()
    assert     is_stale(n-31*86400, 30, now=n)
    assert not is_stale(n-29*86400, 30, now=n)
    assert not is_stale(n-30*86400, 30, now=n)
    assert not is_stale(n,          30, now=n)

def test_is_failed_run():
    from cleanup_worker import is_failed_run
    with tempfile.TemporaryDirectory() as td:
        assert is_failed_run(td)
        open(os.path.join(td,'masked.png'),'w').write('x')
        assert is_failed_run(td)
        open(os.path.join(td,'model.glb'),'wb').write(b'x')
        assert not is_failed_run(td)

def test_cleanup_deletes_stale():
    from cleanup_worker import run_cleanup_pass
    with tempfile.TemporaryDirectory() as base:
        stale = os.path.join(base,'stale')
        os.makedirs(stale)
        os.utime(stale, (time.time()-35*86400,)*2)
        r = run_cleanup_pass(base, max_age_days=30)
        assert any('stale' in p for p in r['deleted'])
        assert not os.path.exists(stale)

def test_cleanup_keeps_fresh():
    import trimesh
    from cleanup_worker import run_cleanup_pass
    with tempfile.TemporaryDirectory() as base:
        fresh = os.path.join(base,'fresh')
        os.makedirs(fresh)
        trimesh.creation.icosphere().export(os.path.join(fresh,'model.glb'))
        r = run_cleanup_pass(base, max_age_days=30)
        assert not any('fresh' in p for p in r['deleted'])
        assert os.path.exists(fresh)

def test_cleanup_status():
    from cleanup_worker import get_status
    s = get_status()
    assert 'deleted_count' in s

# ── Thickness checker ─────────────────────────────────────────────────────

def test_classify_vertex():
    from thickness_checker import classify_vertex
    assert classify_vertex(1.0, 1.5) == 'thin'
    assert classify_vertex(2.0, 1.5) == 'adequate'
    assert classify_vertex(1.5, 1.5) == 'adequate'

def test_thickness_fraction_invariant():
    import trimesh
    from thickness_checker import run_thickness_check
    with tempfile.TemporaryDirectory() as td:
        trimesh.creation.icosphere(subdivisions=3, radius=50.0).export(
            os.path.join(td,'model.glb'))
        r = run_thickness_check(td, min_thickness_mm=1.5)
    nc,nt = r['thin_vertex_count'], r['total_vertex_count']
    assert abs(r['thin_fraction'] - nc/nt) < 1e-9
    assert 0.0 <= r['thin_fraction'] <= 1.0

def test_thickness_report_written():
    import trimesh
    from thickness_checker import run_thickness_check
    with tempfile.TemporaryDirectory() as td:
        trimesh.creation.box().export(os.path.join(td,'model.glb'))
        r = run_thickness_check(td)
        assert os.path.exists(os.path.join(td,'thickness_report.json'))
    for k in ('min_thickness_mm','thin_vertex_count','total_vertex_count',
              'thin_fraction','vertex_thickness'):
        assert k in r

# ── Auth: share CRUD (each test gets its own DB file to avoid collisions) ──

def _make_auth_db(path):
    import auth as a
    orig = a.DB_PATH
    a.DB_PATH = path
    a.init_db()
    return a, orig

def _restore_auth(a, orig):
    a.DB_PATH = orig

def _seed(a, suffix=''):
    name = 'usr' + str(uuid.uuid4().hex[:8]) + suffix
    result = a.register_user(name, name+'@t.com', 'pass1234')
    assert result['ok'], f"register_user failed: {result}"
    with a._conn() as c:
        uid = c.execute("SELECT id FROM users WHERE username=?", (name,)).fetchone()[0]
        c.execute("INSERT INTO history (user_id,created,out_dir,watertight) VALUES (?,?,?,0)",
                  (uid, int(time.time()), f'generated/{name}'))
        hid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    return uid, hid

def test_share_create_get(tmp_path):
    db = str(tmp_path/'a.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        uid,hid = _seed(a)
        s = a.create_share(uid, hid)
        assert len(s['token'])==64
        assert all(c in '0123456789abcdef' for c in s['token'])
        row = a.get_share(s['token'])
        assert row is not None and row['active']==1
    finally:
        a.DB_PATH=orig

def test_share_revoke(tmp_path):
    db = str(tmp_path/'b.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        uid,hid = _seed(a)
        s = a.create_share(uid,hid)
        assert a.revoke_share(uid, s['token'])
        assert a.get_share(s['token']) is None
    finally:
        a.DB_PATH=orig

def test_share_expired(tmp_path):
    db = str(tmp_path/'c.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        uid,hid = _seed(a)
        s = a.create_share(uid,hid, expires=int(time.time())-100)
        assert a.get_share(s['token']) is None
    finally:
        a.DB_PATH=orig

def test_share_token_uniqueness(tmp_path):
    db = str(tmp_path/'d.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        uid,hid = _seed(a)
        tokens = {a.create_share(uid,hid)['token'] for _ in range(10)}
        assert len(tokens)==10
    finally:
        a.DB_PATH=orig

# ── Auth: job persistence ─────────────────────────────────────────────────

def test_upsert_get_job(tmp_path):
    db = str(tmp_path/'j.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        a.upsert_job('generated/abc', user_id=1, status='queued', stage='queued')
        r = a.get_job('generated/abc')
        assert r is not None and r['status']=='queued'
        a.upsert_job('generated/abc', status='running')
        assert a.get_job('generated/abc')['status']=='running'
    finally:
        a.DB_PATH=orig

def test_mark_interrupted(tmp_path):
    db = str(tmp_path/'k.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        a.upsert_job('generated/r1', status='running')
        a.upsert_job('generated/r2', status='queued')
        a.upsert_job('generated/r3', status='done')
        a.mark_interrupted_jobs()
        assert a.get_job('generated/r1')['status']=='interrupted'
        assert a.get_job('generated/r2')['status']=='interrupted'
        assert a.get_job('generated/r3')['status']=='done'
    finally:
        a.DB_PATH=orig

def test_job_partial_update(tmp_path):
    db = str(tmp_path/'p.db')
    import auth as a; orig=a.DB_PATH; a.DB_PATH=db; a.init_db()
    try:
        a.upsert_job('generated/u1', status='queued', backend='triposr')
        a.upsert_job('generated/u1', status='done')
        r = a.get_job('generated/u1')
        assert r['status']=='done'
        assert r['backend']=='triposr'
    finally:
        a.DB_PATH=orig