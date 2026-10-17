import os, time, threading, logging
logger = logging.getLogger(__name__)
_status = {'last_run': None, 'deleted_count': 0}
_lock = threading.Lock()

def is_stale(mtime, max_age_days, now=None):
    if now is None: now=time.time()
    return (now - mtime) > max_age_days * 86400

def is_failed_run(subdir_path):
    try:
        files=[f for f in os.listdir(subdir_path) if os.path.getsize(os.path.join(subdir_path,f))>0]
        return files==['masked.png'] or files==[]
    except: return False

def run_cleanup_pass(generated_root, max_age_days=30):
    deleted=[]; errors=[]
    if not os.path.isdir(generated_root): return {'deleted':deleted,'errors':errors}
    now=time.time()
    for name in os.listdir(generated_root):
        p=os.path.join(generated_root,name)
        if not os.path.isdir(p): continue
        try:
            mtime=os.path.getmtime(p)
            if is_stale(mtime, max_age_days, now) or is_failed_run(p):
                import shutil; shutil.rmtree(p)
                logger.info('CleanupWorker: deleted %s', p)
                # Remove from DB
                try:
                    out_dir_rel='generated/'+name
                    import sqlite3; c=sqlite3.connect('trigenix.db')
                    rows=c.execute('SELECT id,user_id FROM history WHERE out_dir=?',(out_dir_rel,)).fetchall()
                    for rid,uid in rows:
                        from auth import delete_history_entry
                        delete_history_entry(uid,rid)
                    c.close()
                except: pass
                deleted.append(p)
        except OSError as e:
            logger.error('CleanupWorker: failed to delete %s: %s', p, e)
            errors.append(p)
    with _lock:
        _status['last_run']=int(time.time())
        _status['deleted_count']+=len(deleted)
    return {'deleted':deleted,'errors':errors}

def get_status():
    with _lock: return {'ok':True,'last_run':_status['last_run'],'deleted_count':_status['deleted_count']}

def start_cleanup_daemon(generated_root, max_age_days=30):
    def _loop():
        while True:
            time.sleep(86400)  # 24 hours
            try: run_cleanup_pass(generated_root, max_age_days)
            except Exception as e: logger.error('CleanupWorker pass failed: %s', e)
    t=threading.Thread(target=_loop, daemon=True)
    t.start()
    logger.info('CleanupWorker daemon started (max_age_days=%d)', max_age_days)