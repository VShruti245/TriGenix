'use strict';

/* XSS-safe HTML escape helper */
function esc(v) {
  if (v == null) return '';
  return String(v)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}



/* ── Auth guard ─────────────────────────────────────────────── */
var _token    = localStorage.getItem('tg_token')    || '';
var _username = localStorage.getItem('tg_username') || '';

(function checkAuth() {
  if (!_token) { window.location.href = '/HTML/Login.html'; return; }
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/auth/me', true);
  xhr.setRequestHeader('Authorization', 'Bearer ' + _token);
  xhr.onload = function() {
    try {
      var d = JSON.parse(xhr.responseText);
      if (!d.ok) { window.location.href = '/HTML/Login.html'; }
      else {
        var el = document.getElementById('nav-username');
        if (el) el.textContent = d.username;
      }
    } catch(e) { window.location.href = '/HTML/Login.html'; }
  };
  xhr.onerror = function() {};
  xhr.send();
})();

/* ── Logout ──────────────────────────────────────────────────── */
window.doLogout = function() {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/auth/logout', true);
  xhr.setRequestHeader('Authorization', 'Bearer ' + _token);
  xhr.onload = xhr.onerror = function() {
    localStorage.removeItem('tg_token');
    localStorage.removeItem('tg_username');
    window.location.href = '/HTML/Login.html';
  };
  xhr.send();
};

/* ── State ─────────────────────────────────────────────────────── */
var _file       = null;
var _depthMode  = 'color';
var _lastData   = null;
var _pollTimer  = null;
var _currentOutDir = null;
var _webcamStream  = null;
var _userViews  = {};

function _updateMultiviewStatus() {
  var status = ge('mvStatus');
  if (!status) return;
  var count = Object.keys(_userViews).length;
  if (count === 0) {
    status.className = 'mv-status';
    status.textContent = 'No extra views selected — Zero123++ will synthesize them.';
  } else if (count === 6) {
    status.className = 'mv-status ready';
    status.textContent = 'All 6 user views selected — real photos will drive geometry.';
  } else {
    status.className = 'mv-status partial';
    status.textContent = count + '/6 views selected — add all six or clear extras to use synthesis.';
  }
}

(function initMultiviewInputs() {
  for (var i = 0; i < 6; i++) {
    (function(idx) {
      var input = ge('mv' + idx);
      if (!input) return;
      input.addEventListener('change', function() {
        var field = input.getAttribute('data-view') || ('view_' + idx);
        if (input.files && input.files[0]) {
          _userViews[field] = input.files[0];
        } else {
          delete _userViews[field];
        }
        _updateMultiviewStatus();
      });
    })(i);
  }
})();

function _geometryNoteHtml(data) {
  if (!data) return '';
  var gi = data.geometry_inputs || {};
  var note = gi.note || '';
  var cls = 'geometry-note';
  if (data.method === 'instantmesh' || gi.generated_views) cls += ' ok';
  if (!note && data.method === 'triposr') {
    note = 'Shape came from the front photo only (TripoSR). Generated side/back views were used for color, not geometry.';
  }
  if (!note) return '';
  return '<div class="' + cls + '">' + note + '</div>';
}

function _qualityIssuesHtml(data) {
  var issues = (data.quality && data.quality.issues) || [];
  if (!issues.length) return '';
  return '<div class="geometry-note">' +
    '<strong>Printability checks:</strong><ul style="margin:6px 0 0 16px;padding:0;">' +
    issues.map(function(i) { return '<li>' + esc(i) + '</li>'; }).join('') +
    '</ul></div>';
}

/* ── Helpers ─────────────────────────────────────────────────── */
function ge(id) { return document.getElementById(id); }

function authUrl(url) {
  if (!url) return url;
  var isGenerated = url.indexOf('/generated/') === 0 ||
    url.indexOf(window.location.origin + '/generated/') === 0;
  if (!isGenerated || url.indexOf('token=') >= 0) return url;
  return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'token=' + encodeURIComponent(_token);
}

function absoluteAuthUrl(url) {
  if (!url) return url;
  return authUrl(url.indexOf('http') === 0 ? url : window.location.origin + url);
}

/* ── Stage definitions ──────────────────────────────────────── */
var STAGES = [
  { id: 'stage-0', label: 'Removing background…',    pct: 10  },
  { id: 'stage-1', label: 'Synthesising views…',      pct: 30  },
  { id: 'stage-2', label: 'Estimating depth maps…',   pct: 58  },
  { id: 'stage-3', label: 'Reconstructing 3D mesh…',  pct: 82  },
  { id: 'stage-4', label: 'Packaging .glb file…',     pct: 96  },
];

var _activeStage = -1;
var _smoothTimer = null;
var _currentPct  = 0;
var _targetPct   = 0;

function _setStage(idx) {
  if (idx === _activeStage) return;
  _activeStage = idx;
  STAGES.forEach(function(s, i) {
    var el = ge(s.id);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (i < idx)  el.classList.add('done');
    if (i === idx) el.classList.add('active');
  });
  if (idx >= 0 && idx < STAGES.length)
    _setProgress(STAGES[idx].pct, STAGES[idx].label);
}

function _setProgress(pct, label) {
  _targetPct = pct;
  if (label) ge('progressLabel').textContent = label;
  _smoothFill();
}

function _smoothFill() {
  clearInterval(_smoothTimer);
  _smoothTimer = setInterval(function() {
    if (_currentPct >= _targetPct) { clearInterval(_smoothTimer); return; }
    _currentPct = Math.min(_currentPct + 0.6, _targetPct);
    ge('progressFill').style.width = _currentPct + '%';
    ge('progressPct').textContent  = Math.round(_currentPct) + '%';
  }, 18);
}

function _finishProgress(msg) {
  clearInterval(_smoothTimer);
  _targetPct = 100; _currentPct = 100;
  ge('progressFill').style.width = '100%';
  ge('progressPct').textContent  = '100%';
  ge('progressLabel').textContent = msg || 'Done!';
  var etaEl = ge('eta-display');
  if (etaEl) etaEl.textContent = '';
  STAGES.forEach(function(s) {
    var el = ge(s.id);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  });
  _logEvent(msg || 'Done!');
  _stopETA();
  setTimeout(function() { ge('progressWrap').style.display = 'none'; }, 2200);
}

function _failProgress() {
  clearInterval(_smoothTimer);
  ge('progressWrap').style.display = 'none';
  _currentPct = 0; _targetPct = 0; _activeStage = -1;
  _stopETA();
  var etaEl = ge('eta-display');
  if (etaEl) etaEl.textContent = '';
}

function _stopProgress() {
  ge('cancelBtn').style.display = 'none';
  ge('submitBtn').disabled = false;
  ge('regenBtn').style.display = _file ? 'block' : 'none';
  _stopETA();
}

function _startProgress() {
  _currentPct = 0; _targetPct = 0; _activeStage = -1;
  _logEntries = []; _logStart = null;
  ge('progressFill').style.width = '0%';
  ge('progressPct').textContent  = '0%';
  STAGES.forEach(function(s) {
    var el = ge(s.id);
    if (el) el.classList.remove('active', 'done');
  });
  ge('progressWrap').style.display = 'block';
  ge('cancelBtn').style.display = 'block';
  ge('regenBtn').style.display  = 'none';
  ge('gen-log').style.display = 'block';
  _logEvent('Generation started');
  _startETA();
  _setStage(0);
}

/* ══════════════════════════════════════════════════════════════
   WEBCAM
══════════════════════════════════════════════════════════════ */
window.openWebcam = function() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Webcam not supported in this browser.'); return;
  }
  navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false })
    .then(function(stream) {
      _webcamStream = stream;
      var video = ge('webcamVideo');
      video.srcObject = stream;
      ge('webcamModal').classList.add('open');
    })
    .catch(function(err) {
      alert('Could not access webcam: ' + err.message);
    });
};

window.closeWebcam = function() {
  if (_webcamStream) {
    _webcamStream.getTracks().forEach(function(t) { t.stop(); });
    _webcamStream = null;
  }
  ge('webcamModal').classList.remove('open');
};

window.captureWebcam = function() {
  var video  = ge('webcamVideo');
  var canvas = ge('webcamCanvas');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(function(blob) {
    var file = new File([blob], 'webcam_capture.png', { type: 'image/png' });
    closeWebcam();
    _loadFile(file);
  }, 'image/png');
};

/* ══════════════════════════════════════════════════════════════
   IMAGE QUALITY CHECKER
══════════════════════════════════════════════════════════════ */
function _analyzeImageQuality(file, callback) {
  var img = new Image();
  var url = URL.createObjectURL(file);
  img.onload = function() {
    var w = img.naturalWidth, h = img.naturalHeight;
    var canvas = document.createElement('canvas');
    var scale  = Math.min(1, 200 / Math.max(w, h));
    canvas.width  = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    var data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    URL.revokeObjectURL(url);

    var warnings = [];

    // ── Resolution check ──────────────────────────────────────
    if (w < 256 || h < 256) {
      warnings.push('Image is very small (' + w + '×' + h + 'px). Use at least 512×512 for better results.');
    }

    // ── Brightness check ──────────────────────────────────────
    var totalBright = 0, n = data.length / 4;
    for (var i = 0; i < data.length; i += 4) {
      totalBright += (data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114);
    }
    var avgBright = totalBright / n;
    if (avgBright < 40) {
      warnings.push('Image is too dark (avg brightness: ' + Math.round(avgBright) + '). Try better lighting.');
    } else if (avgBright > 220) {
      warnings.push('Image is overexposed (avg brightness: ' + Math.round(avgBright) + '). Reduce lighting.');
    }

    // ── Contrast / detail check (variance) ───────────────────
    var variance = 0;
    for (var j = 0; j < data.length; j += 4) {
      var lum = data[j]*0.299 + data[j+1]*0.587 + data[j+2]*0.114;
      variance += (lum - avgBright) * (lum - avgBright);
    }
    variance /= n;
    if (variance < 200) {
      warnings.push('Image is very low contrast or blurry. The object may not be detected well.');
    }

    // ── Background complexity check ───────────────────────────
    // Sample corner pixels — if they're all very different, bg is complex
    var corners = [];
    var cw = canvas.width, ch = canvas.height, r = Math.max(2, Math.floor(cw / 10));
    var regions = [
      ctx.getImageData(0, 0, r, r),
      ctx.getImageData(cw - r, 0, r, r),
      ctx.getImageData(0, ch - r, r, r),
      ctx.getImageData(cw - r, ch - r, r, r)
    ];
    regions.forEach(function(rd) {
      var tot = 0;
      for (var k = 0; k < rd.data.length; k += 4)
        tot += rd.data[k]*0.299 + rd.data[k+1]*0.587 + rd.data[k+2]*0.114;
      corners.push(tot / (rd.data.length / 4));
    });
    var cornerRange = Math.max.apply(null, corners) - Math.min.apply(null, corners);
    if (cornerRange > 80) {
      warnings.push('Background appears complex or cluttered. A plain/white background gives better results.');
    }

    // ── Object size in frame check ────────────────────────────
    // Check how much of the image is non-white/near-white
    var nonWhite = 0;
    for (var p = 0; p < data.length; p += 4) {
      var r2 = data[p], g2 = data[p+1], b2 = data[p+2];
      if (!(r2 > 220 && g2 > 220 && b2 > 220)) nonWhite++;
    }
    var objFrac = nonWhite / n;
    if (objFrac < 0.05) {
      warnings.push('Object appears very small in the frame. Try to fill more of the image with the object.');
    }

    callback(warnings);
  };
  img.onerror = function() { URL.revokeObjectURL(url); callback([]); };
  img.src = url;
}

function _showQualityWarnings(warnings) {
  var warn = ge('qualityWarn');
  var list = ge('qualityList');
  if (!warnings || warnings.length === 0) {
    warn.className = 'quality-warn ok show';
    warn.innerHTML = '<span class="quality-icon">✅</span><div>Image looks good — ready to generate!</div>';
    setTimeout(function() { warn.classList.remove('show'); }, 3000);
    return;
  }
  warn.className = 'quality-warn show';
  warn.innerHTML = '<span class="quality-icon">⚠️</span><div><strong>Quality warnings:</strong><ul id="qualityList">' +
    warnings.map(function(w) { return '<li>' + esc(w) + '</li>'; }).join('') + '</ul></div>';
}

/* ══════════════════════════════════════════════════════════════
   CANCEL GENERATION
══════════════════════════════════════════════════════════════ */
window.cancelGeneration = function() {
  if (!_currentOutDir) {
    // Job hasn't started reconstruction yet — just abort the XHR
    if (window._activeXhr) {
      window._activeXhr.abort();
    }
    _failProgress();
    _stopProgress();
    ge('resultArea').innerHTML = '<div class="result-err"><h3>Cancelled</h3><p>Generation was cancelled.</p></div>';
    return;
  }
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/cancel?dir=' + encodeURIComponent(_currentOutDir), true);
  xhr.setRequestHeader('Authorization', 'Bearer ' + _token);
  xhr.onload = function() {
    if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    _failProgress();
    _stopProgress();
    ge('resultArea').innerHTML = '<div class="result-err"><h3>Cancelled</h3><p>Generation was cancelled.</p></div>';
  };
  xhr.send();
};

/* ── Drop zone ───────────────────────────────────────────────── */
ge('dropZone').onclick = function() { ge('fileInput').click(); };

ge('fileInput').onchange = function(e) {
  if (e.target.files[0]) _loadFile(e.target.files[0]);
};

ge('dropZone').addEventListener('dragover', function(e) {
  e.preventDefault(); this.classList.add('over');
});
ge('dropZone').addEventListener('dragleave', function() {
  this.classList.remove('over');
});
ge('dropZone').addEventListener('drop', function(e) {
  e.preventDefault(); this.classList.remove('over');
  var f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) _loadFile(f);
});

document.addEventListener('paste', function(e) {
  for (var i = 0; i < e.clipboardData.items.length; i++) {
    if (e.clipboardData.items[i].type.startsWith('image/')) {
      _loadFile(e.clipboardData.items[i].getAsFile()); break;
    }
  }
});

function _loadFile(f) {
  _file = f;
  var r = new FileReader();
  r.onload = function(e) {
    ge('preview').src = e.target.result;
    ge('previewWrap').classList.add('has-image');
    ge('submitBtn').disabled = false;
    var icon  = ge('dropZone').querySelector('.dz-icon');
    var title = ge('dropZone').querySelector('.dz-title');
    var hint  = ge('dropZone').querySelector('.dz-hint');
    if (icon)  icon.style.display  = 'none';
    if (title) title.style.display = 'none';
    if (hint)  hint.style.display  = 'none';
    // Run quality check
    _analyzeImageQuality(f, _showQualityWarnings);
  };
  r.readAsDataURL(f);
}

ge('removeBtn').onclick = function(e) {
  e.stopPropagation();
  _file = null;
  ge('preview').src = '#';
  ge('previewWrap').classList.remove('has-image');
  ge('fileInput').value = '';
  ge('submitBtn').disabled = true;
  ge('qualityWarn').classList.remove('show');
  var icon  = ge('dropZone').querySelector('.dz-icon');
  var title = ge('dropZone').querySelector('.dz-title');
  var hint  = ge('dropZone').querySelector('.dz-hint');
  if (icon)  icon.style.display  = '';
  if (title) title.style.display = '';
  if (hint)  hint.style.display  = '';
};

/* ── Submit ──────────────────────────────────────────────────── */
ge('submitBtn').onclick = function() {
  if (!_file) return;
  if (!_token) { window.location.href = '/HTML/Login.html'; return; }
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  ge('resultArea').innerHTML = '';
  ge('submitBtn').disabled = true;
  _currentOutDir = null;
  _startProgress();

  var fd = new FormData();
  fd.append('image', _file);

  // Attach notification email if provided
  var notifyEmail = (ge('notifyEmail').value || '').trim();
  if (notifyEmail && notifyEmail.indexOf('@') > 0) {
    fd.append('notify_email', notifyEmail);
  }

  var targetWidth = parseFloat((ge('targetWidth') && ge('targetWidth').value) || '');
  var targetHeight = parseFloat((ge('targetHeight') && ge('targetHeight').value) || '');
  var targetDepth = parseFloat((ge('targetDepth') && ge('targetDepth').value) || '');
  if (targetWidth > 0) fd.append('target_width', String(targetWidth));
  if (targetHeight > 0) fd.append('target_height', String(targetHeight));
  if (targetDepth > 0) fd.append('target_depth', String(targetDepth));

  var mvKeys = Object.keys(_userViews);
  if (mvKeys.length > 0) {
    if (mvKeys.length !== 6) {
      _failProgress();
      _showErr('Multi-view mode requires all 6 side/back photos (view_0 … view_5).', '');
      ge('submitBtn').disabled = false;
      return;
    }
    mvKeys.sort().forEach(function(field) {
      fd.append(field, _userViews[field], _userViews[field].name || field + '.jpg');
    });
  }

  var xhr = new XMLHttpRequest();
  window._activeXhr = xhr;
  xhr.open('POST', '/upload', true);
  xhr.timeout = 25 * 60 * 1000;
  xhr.setRequestHeader('Authorization', 'Bearer ' + _token);

  var _stageTimer = null;
  var _stageSeq   = [
    { delay: 800,   stage: 0, log: 'Removing background…'   },
    { delay: 4000,  stage: 1, log: 'Synthesising views…'     },
    { delay: 18000, stage: 2, log: 'Estimating depth maps…'  },
  ];
  var _sqIdx = 0;

  function _advanceStageSeq() {
    if (_sqIdx >= _stageSeq.length) return;
    var entry = _stageSeq[_sqIdx++];
    _stageTimer = setTimeout(function() {
      _setStage(entry.stage);
      _logEvent(entry.log);
      _advanceStageSeq();
    }, entry.delay);
  }
  _advanceStageSeq();

  xhr.onload = function() {
    clearTimeout(_stageTimer);
    _stopProgress();
    var data;
    try { data = JSON.parse(xhr.responseText); } catch(e) {
      _failProgress(); _showErr('Non-JSON response', xhr.responseText); return;
    }
    if (xhr.status === 401) {
      _failProgress();
      localStorage.removeItem('tg_token');
      localStorage.removeItem('tg_username');
      window.location.href = '/HTML/Login.html';
      return;
    }
    if (xhr.status === 200 && data.status === 'success') {
      _setStage(3);
      _logEvent('Views & depth maps complete in ' + (data.time || '?') + 's');
      _logEvent('Starting 3D mesh reconstruction…');
      // Save stage times for future ETA estimates
      if (data.stage_times) _saveAvgTimes(data.stage_times);
      _lastData = data;
      _currentOutDir = data.recon_dir || null;
      _renderResult(data);
      if (data.recon_status === 'running' && data.recon_dir) {
        _startReconPoll(data.recon_dir);
      } else {
        _finishProgress('Views & depth ready!');
      }
    } else {
      _logEvent('ERROR: ' + (data.message || 'HTTP ' + xhr.status));
      _failProgress();
      _showErr(data.message || 'HTTP ' + xhr.status, xhr.responseText);
    }
  };
  xhr.onerror   = function() { clearTimeout(_stageTimer); _failProgress(); _stopProgress(); _showErr('Cannot reach server. Make sure server.py is running.', ''); };
  xhr.ontimeout = function() { clearTimeout(_stageTimer); _failProgress(); _stopProgress(); _showErr('Timed out after 25 min. A GPU is recommended.', ''); };
  xhr.onabort   = function() { clearTimeout(_stageTimer); };
  xhr.send(fd);
};

/* ── Reconstruction polling ─────────────────────────────────── */
function _startReconPoll(outDir) {
  _updateReconCard('running', null);
  _pollTimer = setInterval(function() {
    var xhr2 = new XMLHttpRequest();
    xhr2.open('GET', '/recon_status?dir=' + encodeURIComponent(outDir), true);
    xhr2.timeout = 10000;
    xhr2.setRequestHeader('Authorization', 'Bearer ' + _token);
    xhr2.onload = function() {
      var d;
      try { d = JSON.parse(xhr2.responseText); } catch(e) { return; }
      if (xhr2.status === 401) {
        clearInterval(_pollTimer); _pollTimer = null;
        _failProgress();
        localStorage.removeItem('tg_token');
        localStorage.removeItem('tg_username');
        window.location.href = '/HTML/Login.html';
        return;
      }
      if (d.status === 'done' || d.status === 'success') {
        clearInterval(_pollTimer); _pollTimer = null;
        _setStage(4);
        _logEvent('Mesh reconstruction complete — ' + (d.vertices||0).toLocaleString() + ' verts, ' + (d.faces||0).toLocaleString() + ' faces');
        _logEvent('Backend: ' + (d.method || 'unknown') +           (d.geometry_source ? ' (' + d.geometry_source + ')' : ''));
        if (d.geometry_inputs && d.geometry_inputs.note) _logEvent(d.geometry_inputs.note);
        if (d.method === 'triposr') _logEvent('Tip: install nvdiffrast (pip install nvdiffrast) to enable InstantMesh for better multi-view geometry.');
        _logEvent('.glb file ready: ' + (d.final_glb || ''));
        if (d.recon_time) _saveAvgTimes({ reconstruction: parseFloat(d.recon_time) });
        // Save processor stage times if returned alongside recon result
        if (d.stage_times) _saveAvgTimes(d.stage_times);
        _updateReconCard('done', d);
        _finishProgress('.glb ready!');
        _updateCompareViewer(d.final_glb);
      } else if (d.status === 'running' || d.status === 'queued') {
        // Update ETA using real elapsed_seconds from server
        if (typeof d.elapsed_seconds === 'number') {
          _updateETAFromServer(d.elapsed_seconds);
        }
        var qpos = d.queue_position || 0;
        if (qpos > 1) {
          _logEvent('GPU queue position: ' + qpos);
          var card = ge('recon-card');
          if (card) {
            var sub = card.querySelector('.recon-sub');
            if (sub) sub.textContent = 'Waiting in GPU queue (position ' + qpos + ')…';
          }
        }
      } else if (d.status === 'error') {
        clearInterval(_pollTimer); _pollTimer = null;
        _logEvent('ERROR: Reconstruction failed — ' + (d.message || ''));
        _updateReconCard('error', d);
        _finishProgress('Reconstruction failed');
      } else if (d.status === 'cancelled') {
        clearInterval(_pollTimer); _pollTimer = null;
        _logEvent('Generation cancelled by user.');
        _updateReconCard('error', { message: 'Generation was cancelled.' });
        _failProgress();
      }
    };
    xhr2.onerror = function() {};
    xhr2.send();
  }, 5000);
}

window.toggleLog = function() {
  var body = ge('gen-log-body');
  var icon = ge('log-toggle-icon');
  if (!body) return;
  var hidden = body.style.display === 'none';
  body.style.display = hidden ? '' : 'none';
  if (icon) icon.textContent = hidden ? '▲' : '▼';
};

function _updateReconCard(status, data) {
  var card = ge('recon-card');
  if (!card) return;
  card.className = 'recon-card ' + status;

  if (status === 'running') {
    card.innerHTML =
      '<div class="recon-row">' +
        '<div class="recon-icon">🧊</div>' +
        '<div class="recon-info">' +
          '<div class="recon-title"><span class="pulse-dot"></span>Rebuilding 3D mesh…</div>' +
          '<div class="recon-sub">Running mesh reconstruction in the background. This takes a few minutes on CPU.</div>' +
        '</div>' +
      '</div>';
  } else if (status === 'done') {
    var glb  = data.final_glb || '';
    // Prefer UV-textured GLB for viewing if available (better color fidelity)
    var _uvp = (data.asset_profiles || {}).uv_textured;
    var _viewGlb = (_uvp && _uvp.url) ? _uvp.url : glb;
    var vUrl = '/HTML/Viewer3d.html?model=' + encodeURIComponent(absoluteAuthUrl(_viewGlb)) + '&name=model.glb';
    var dlUrl = authUrl(glb);
    var profiles = data.asset_profiles || {};
    var dims = data.dimensions || {};
    var dimText = dims.width
      ? (Number(dims.width).toFixed(2) + ' x ' + Number(dims.height).toFixed(2) + ' x ' + Number(dims.depth).toFixed(2))
      : '?';
    function profileButton(key, text, viewer) {
      var p = profiles[key];
      if (!p || !p.url) return '';
      var href = viewer
        ? '/HTML/Viewer3d.html?model=' + encodeURIComponent(absoluteAuthUrl(p.url)) + '&name=' + encodeURIComponent(text)
        : authUrl(p.url);
      return '<a class="btn-3d" href="' + href + '"' + (viewer ? ' target="_blank"' : ' download') + ' style="background:transparent;border:1px solid var(--border2);color:var(--text);">' + text + '</a>';
    }
    var profileButtons =
      profileButton('print_stl', 'Print STL', false) +
      profileButton('print_glb', 'Print GLB', true) +
      profileButton('ar_vr', 'AR/VR', true) +
      profileButton('game', 'Game LOD', true) +
      profileButton('simulation', 'Simulation', true) +
      profileButton('robotics_collision', 'Collision OBJ', false) +
      profileButton('texture_preview', 'Texture Preview', true) +
      profileButton('quality_report', 'Quality JSON', false);

    // Mesh info panel
    var meshInfo =
      '<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">' +
        _infoBadge('Vertices',   Number(data.vertices||0).toLocaleString()) +
        _infoBadge('Faces',      Number(data.faces||0).toLocaleString()) +
        _infoBadge('W x H x D',   dimText) +
        _infoBadge('Method',     data.method || '?') +
        _infoBadge('Time',       (data.recon_time||'?') + 's') +
        (data.silhouette_iou != null ? _infoBadge('Shape IoU', Number(data.silhouette_iou).toFixed(2), data.silhouette_iou >= 0.35 ? 'green' : 'red') : '') +
        (data.print_ready || data.watertight ? _infoBadge('Print-ready','✓','green') : _infoBadge('Print-ready','✗','red')) +
        (data.topology && data.topology.watertight === false && data.visual_watertight === false ? _infoBadge('Visual','has openings','orange') : '') +
      '</div>';

    card.innerHTML =
      '<div class="recon-row">' +
        '<div class="recon-icon">✅</div>' +
        '<div class="recon-info" style="flex:1;">' +
          '<div class="recon-title">.glb ready</div>' +
          '<div class="recon-sub">' + meshInfo + _geometryNoteHtml(data) + _qualityIssuesHtml(data) + '</div>' +
        '</div>' +
        '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-start;">' +
          '<a class="btn-3d" href="' + vUrl + '" target="_blank">Open 3D Viewer →</a>' +
          '<a class="btn-3d" href="' + dlUrl + '" download="model.glb" style="background:transparent;border:1px solid var(--border2);color:var(--text);">⬇ Download .glb</a>' +
          profileButtons +
        '</div>' +
      '</div>';
  } else {
    card.innerHTML =
      '<div class="recon-row">' +
        '<div class="recon-icon">❌</div>' +
        '<div class="recon-info">' +
          '<div class="recon-title" style="color:var(--red)">Reconstruction failed</div>' +
          '<div class="recon-sub">' + esc(data && data.message || 'Unknown error') + '</div>' +
        '</div>' +
      '</div>';
  }
}

/* ── Side-by-side comparison viewer updater ─────────────────── */
function _updateCompareViewer(glbPath) {
  var placeholder = ge('compare-3d-placeholder');
  if (!placeholder || !glbPath) return;
  var vUrl = '/HTML/Viewer3d.html?model=' + encodeURIComponent(absoluteAuthUrl(glbPath)) + '&name=model.glb';
  placeholder.outerHTML =
    '<a href="' + vUrl + '" target="_blank" style="text-decoration:none;display:block;">' +
      '<div style="width:100%;max-height:200px;border-radius:7px;border:1px solid rgba(139,127,248,.4);background:rgba(139,127,248,.06);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;padding:24px;cursor:pointer;">' +
        '<span style="font-size:2rem;">🧊</span>' +
        '<span style="font-family:var(--mono);font-size:.68rem;color:var(--acc);">Click to open 3D Viewer →</span>' +
      '</div>' +
    '</a>';
}

function _infoBadge(label, value, color) {
  var col = color === 'green' ? 'rgba(74,222,128,.8)' : color === 'red' ? 'var(--red)' : 'var(--text)';
  return '<div style="background:var(--surface);border:1px solid var(--border2);border-radius:6px;padding:5px 10px;font-family:var(--mono);font-size:.62rem;">' +
    '<span style="color:var(--muted);">' + label + '</span><br>' +
    '<span style="color:' + col + ';font-weight:700;">' + value + '</span>' +
  '</div>';
}

function _depthAnalysis(depths) {
  // Returns HTML for a simple depth analysis summary
  if (!depths || depths.length === 0) return '';
  return (
    '<div style="margin-top:10px;padding:10px 14px;background:var(--bg);border:1px solid var(--border2);border-radius:8px;">' +
      '<div style="font-family:var(--mono);font-size:.62rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">Depth Map Analysis <span style="opacity:.5;font-size:.85em;">(preview only)</span></div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;">' +
        _infoBadge('Total views', depths.length + '') +
        _infoBadge('Front depth', 'view 0') +
        _infoBadge('Coverage', depths.length + '/7 views') +
      '</div>' +
      '<p style="font-size:.72rem;color:var(--muted);margin-top:8px;line-height:1.5;">Depth maps estimate per-pixel distance. Warmer colours (red/yellow) = closer to camera. Cooler colours (blue/green) = farther away.</p>' +
    '</div>'
  );
}

/* ── Render result ──────────────────────────────────────────── */
function _renderResult(data) {
  var views  = data.views  || [];
  var dColor = data.depths_color || [];

  var frontHtml =
    '<div class="section-label">Input</div>' +
    '<div class="front-row">' +
      _tile(data.masked,  'Subject') +
      _tile(views[0] || '', 'View 0°') +
    '</div>';

  var viewLabels = ['30°', '90°', '150°', '210°', '270°', '330°'];
  var viewsTitle = data.input_mode === 'user_multiview' ? 'Uploaded Views' : 'Synthesised Views';
  var viewHtml = '<div class="section-label">' + viewsTitle + '</div><div class="img-grid">';
  for (var i = 0; i < views.length; i++)
    viewHtml += _tile(views[i], 'az ' + (viewLabels[i] || i));
  viewHtml += '</div>';

  var depthCtrl =
    '<div class="section-label">Depth Maps</div>' +
    '<div class="depth-toggle">' +
      '<button class="dtgl-btn active" id="dtgl-color" onclick="setDepthMode(\'color\')">Colour</button>' +
      '<button class="dtgl-btn"        id="dtgl-grey"  onclick="setDepthMode(\'grey\')">Greyscale</button>' +
    '</div>' +
    '<div class="img-grid" id="depthGrid"></div>' +
    _depthAnalysis(data.depths_color);

  // Side-by-side comparison
  var compareHtml =
    '<div class="compare-wrap">' +
      '<div class="compare-header">' +
        '<span>Side-by-side comparison</span>' +
      '</div>' +
      '<div class="compare-grid">' +
        '<div class="compare-side">' +
          '<div class="compare-label">Input photo</div>' +
          (data.masked ? '<img src="' + authUrl(data.masked) + '" alt="input">' : '<div style="color:var(--muted);font-size:.8rem;">—</div>') +
        '</div>' +
        '<div class="compare-side">' +
          '<div class="compare-label">Generated 3D</div>' +
          '<div class="compare-3d-placeholder" id="compare-3d-placeholder">' +
            '<span style="font-size:1.5rem;opacity:.3;">🧊</span>' +
            '<span style="font-family:var(--mono);font-size:.6rem;color:var(--muted);">Waiting for mesh…</span>' +
          '</div>' +
        '</div>' +
      '</div>' +
    '</div>';

  var reconCard =
    '<div class="recon-card running" id="recon-card">' +
      '<div class="recon-row">' +
        '<div class="recon-icon">⏳</div>' +
        '<div class="recon-info">' +
          '<div class="recon-title">Starting mesh reconstruction…</div>' +
          '<div class="recon-sub">Will update automatically when the .glb is ready.</div>' +
        '</div>' +
      '</div>' +
    '</div>';

  var histLink =
    '<div style="margin-top:14px;text-align:right;">' +
      '<a href="History.html" style="font-family:var(--mono);font-size:.72rem;color:var(--acc2);">View all generations →</a>' +
    '</div>';

  ge('resultArea').innerHTML =
    '<div class="result-ok">' +
      '<div class="meta-bar">' +
        '<h3>✓ Processing complete</h3>' +
        '<div class="meta-badge">time: <span>' + (data.time || '?') + 's</span></div>' +
        '<div class="meta-badge">views: <span>' + (views.length + 1) + '</span></div>' +
      '</div>' +
      frontHtml + viewHtml + depthCtrl + compareHtml + reconCard + histLink +
    '</div>';

  _renderDepthGrid(data);
  _setupLightbox();
}

window.setDepthMode = function(mode) {
  _depthMode = mode;
  ge('dtgl-grey').classList.toggle('active',  mode === 'grey');
  ge('dtgl-color').classList.toggle('active', mode === 'color');
  if (_lastData) _renderDepthGrid(_lastData);
};

function _renderDepthGrid(data) {
  var grid = ge('depthGrid');
  if (!grid) return;
  var arr    = _depthMode === 'color' ? (data.depths_color || []) : (data.depths_grey || []);
  var label  = _depthMode === 'color' ? 'colour' : 'grey';
  var labels = ['front', '30°', '90°', '150°', '210°', '270°', '330°'];
  var html   = '';
  for (var i = 0; i < arr.length; i++)
    html += _tile(arr[i], label + ' · ' + (labels[i] || i));
  grid.innerHTML = html;
  _setupLightbox();
}

function _tile(src, label) {
  if (!src) return '';
  var safeSrc = authUrl(src);
  return '<div class="img-tile" data-src="' + safeSrc + '">' +
    '<img src="' + safeSrc + '" loading="lazy" alt="' + label + '">' +
    '<div class="tile-label">' + label + '</div>' +
  '</div>';
}

function _setupLightbox() {
  document.querySelectorAll('.img-tile').forEach(function(el) {
    el.onclick = function() {
      ge('lbImg').src = this.dataset.src;
      ge('lb').classList.add('open');
    };
  });
}

ge('lb').onclick = function() {
  this.classList.remove('open');
  ge('lbImg').src = '';
};

/* ── Error display ──────────────────────────────────────────── */
function _showErr(msg, detail) {
  ge('submitBtn').disabled = false;
  ge('cancelBtn').style.display = 'none';
  ge('resultArea').innerHTML =
    '<div class="result-err"><h3>Error</h3><p>' + esc(msg) + '</p>' +
    (detail ? '<pre>' + esc(String(detail).slice(0, 500)) + '</pre>' : '') +
    '</div>';
}

/* ══════════════════════════════════════════════════════════════
   ESTIMATED TIME REMAINING
   Uses real elapsed_seconds from server + historical stage averages
   from localStorage to show accurate live countdown.
══════════════════════════════════════════════════════════════ */
var _genStartTime  = null;
var _etaInterval   = null;
// Default stage estimates (seconds) — updated from localStorage after each run
var STAGE_DEFAULTS = {
  bg_removal:       8,
  view_synthesis:   900,   // ~15 min on RTX 3050
  depth_estimation: 60,
  reconstruction:   300,   // ~5 min TripoSR
};

var STAGE_LABELS = [
  { key: 'bg_removal',       label: 'Background removal' },
  { key: 'view_synthesis',   label: 'View synthesis'     },
  { key: 'depth_estimation', label: 'Depth estimation'   },
  { key: 'reconstruction',   label: 'Reconstruction'     },
];

function _getAvgTimes() {
  try {
    return JSON.parse(localStorage.getItem('tg_avg_times') || '{}');
  } catch(e) { return {}; }
}

function _saveAvgTimes(stageTimes) {
  try {
    var avg = _getAvgTimes();
    Object.keys(stageTimes).forEach(function(k) {
      var v = parseFloat(stageTimes[k]);
      if (!isFinite(v) || v <= 0) return;
      avg[k] = avg[k] ? Math.round(avg[k] * 0.6 + v * 0.4) : Math.round(v);
    });
    localStorage.setItem('tg_avg_times', JSON.stringify(avg));
  } catch(e) {}
}

function _getTotalEstimate() {
  var avg = _getAvgTimes();
  var keys = ['bg_removal','view_synthesis','depth_estimation','reconstruction'];
  var total = 0;
  keys.forEach(function(k) {
    total += avg[k] || STAGE_DEFAULTS[k] || 0;
  });
  return total;  // seconds
}

function _fmtRemaining(secs) {
  secs = Math.max(0, Math.round(secs));
  if (secs < 60) return '~' + secs + 's remaining';
  var m = Math.floor(secs / 60), s = secs % 60;
  return '~' + m + 'm ' + (s < 10 ? '0' : '') + s + 's remaining';
}

function _updateETADisplay(remainingSecs) {
  var etaEl = ge('eta-display');
  if (!etaEl) return;
  if (remainingSecs <= 0) { etaEl.textContent = 'almost done…'; return; }
  etaEl.textContent = _fmtRemaining(remainingSecs);
}

// Called every poll tick with real elapsed_seconds from server
function _updateETAFromServer(elapsedSecs) {
  var total = _getTotalEstimate();
  if (!total) return;
  var remaining = total - elapsedSecs;
  _updateETADisplay(remaining);
}

function _startETA() {
  _genStartTime = Date.now();
  var total = _getTotalEstimate();
  // Show estimate immediately (client-side elapsed before first poll)
  _etaInterval = setInterval(function() {
    var clientElapsed = (Date.now() - _genStartTime) / 1000;
    var remaining = total - clientElapsed;
    _updateETADisplay(remaining);
  }, 1000);
  _updateETADisplay(total);
}

function _stopETA() {
  if (_etaInterval) { clearInterval(_etaInterval); _etaInterval = null; }
}

/* ══════════════════════════════════════════════════════════════
   GENERATION LOG
══════════════════════════════════════════════════════════════ */
var _logEntries = [];
var _logStart   = null;

function _logEvent(msg) {
  if (!_logStart) _logStart = Date.now();
  var elapsed = ((Date.now() - _logStart) / 1000).toFixed(1);
  _logEntries.push({ t: elapsed, msg: msg });
  _renderLog();
}

function _renderLog() {
  var el = ge('gen-log-body');
  if (!el) return;
  el.innerHTML = _logEntries.map(function(e) {
    return '<div class="log-line"><span class="log-t">[' + esc(e.t) + 's]</span><span class="log-msg">' + esc(e.msg) + '</span></div>';
  }).join('');
  el.scrollTop = el.scrollHeight;
}

/* ── Re-generate ─────────────────────────────────────────────── */
window.reGenerate = function() {
  if (!_file) return;
  ge('resultArea').innerHTML = '';
  ge('submitBtn').click();
};
