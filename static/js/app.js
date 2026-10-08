'use strict';

/* ── Utilities ───────────────────────────────────────────────────────────── */
function showError(msg) { const el = document.getElementById('errorMsg'); if (el) el.textContent = msg; }
function clearError()   { const el = document.getElementById('errorMsg'); if (el) el.textContent = ''; }

function setLoading(loading) {
  const spinner    = document.getElementById('spinnerWrap');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const getLookBtn = document.getElementById('getLookBtn');
  const submitBtn  = document.getElementById('submitBtn');
  if (spinner)    spinner.hidden    = !loading;
  if (analyzeBtn) { analyzeBtn.disabled = loading; analyzeBtn.textContent = loading ? 'Analyzing…' : 'Analyze my style'; }
  if (getLookBtn) { getLookBtn.disabled = loading; getLookBtn.textContent = loading ? 'Curating…'  : 'Reveal my look →'; }
  if (submitBtn)  { submitBtn.disabled  = loading; submitBtn.textContent  = loading ? 'Finding your look…' : 'Get my outfit →'; }
}

/* ── Age group mapping (mirrors Python AGE_BUCKETS) ─────────────────────── */
function ageToGroup(age) {
  const buckets = [[0,4,'0-4'],[5,9,'5-9'],[10,14,'10-14'],[15,19,'15-19'],
    [20,24,'20-24'],[25,29,'25-29'],[30,34,'30-34'],[35,39,'35-39'],
    [40,44,'40-44'],[45,49,'45-49'],[50,54,'50-54'],[55,99,'55-59']];
  for (const [lo, hi, label] of buckets) if (age >= lo && age <= hi) return label;
  return '55-59';
}

/* ── Theme toggle ─────────────────────────────────────────────────────────── */
(function initTheme() {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  if (localStorage.getItem('sg-theme') === 'light') document.body.classList.add('light');
  btn.addEventListener('click', () => {
    document.body.classList.toggle('light');
    localStorage.setItem('sg-theme', document.body.classList.contains('light') ? 'light' : 'dark');
  });
})();

/* ══════════════════════════════════════════════════════════════════════════
   CAPTURE PAGE — UPLOAD
   ══════════════════════════════════════════════════════════════════════════ */
(function initUpload() {
  const captureArea = document.getElementById('captureArea');
  if (!captureArea || captureArea.dataset.mode !== 'upload') return;

  const fileInput   = document.getElementById('fileInput');
  const dropzone    = document.getElementById('dropzone');
  const previewWrap = document.getElementById('previewWrap');
  const preview     = document.getElementById('preview');
  const retakeBtn   = document.getElementById('retakeBtn');
  const analyzeBtn  = document.getElementById('analyzeBtn');
  let selectedFile  = null;

  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  dropzone.addEventListener('dragover',  (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', ()  => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop',      (e) => { e.preventDefault(); dropzone.classList.remove('drag-over'); if (e.dataTransfer?.files[0]) handleFile(e.dataTransfer.files[0]); });
  dropzone.addEventListener('keydown',   (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });

  retakeBtn.addEventListener('click', () => {
    selectedFile = null; preview.src = ''; fileInput.value = '';
    previewWrap.hidden = true; analyzeBtn.hidden = true; captureArea.hidden = false; clearError();
  });

  analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    const form = new FormData(); form.append('image', selectedFile);
    sendAnalysis(form, true);
  });

  function handleFile(file) {
    if (!['image/jpeg','image/jpg','image/png','image/webp'].includes(file.type)) return showError('Only JPG, PNG, or WEBP images accepted.');
    if (file.size > 10 * 1024 * 1024) return showError('Image must be smaller than 10 MB.');
    clearError(); selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => { preview.src = e.target.result; previewWrap.hidden = false; analyzeBtn.hidden = false; captureArea.hidden = true; };
    reader.readAsDataURL(file);
  }
})();

/* ══════════════════════════════════════════════════════════════════════════
   CAPTURE PAGE — WEBCAM
   ══════════════════════════════════════════════════════════════════════════ */
(function initWebcam() {
  const captureArea = document.getElementById('captureArea');
  if (!captureArea || captureArea.dataset.mode !== 'webcam') return;

  const video       = document.getElementById('camera');
  const canvas      = document.getElementById('canvas');
  const snapBtn     = document.getElementById('snapBtn');
  const previewWrap = document.getElementById('previewWrap');
  const preview     = document.getElementById('preview');
  const retakeBtn   = document.getElementById('retakeBtn');
  const analyzeBtn  = document.getElementById('analyzeBtn');
  let capturedURL   = null;
  let stream        = null;

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream;
    } catch {
      showError('Camera access denied. Allow camera permission or use the Upload option.');
      snapBtn.disabled = true;
    }
  }
  startCamera();

  snapBtn.addEventListener('click', () => {
    if (!stream) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    capturedURL = canvas.toDataURL('image/jpeg', 0.92);
    stream.getTracks().forEach(t => t.stop()); stream = null;
    preview.src = capturedURL; previewWrap.hidden = false; analyzeBtn.hidden = false;
    captureArea.hidden = true; clearError();
  });

  retakeBtn.addEventListener('click', () => {
    capturedURL = null; previewWrap.hidden = true; analyzeBtn.hidden = true;
    captureArea.hidden = false; clearError(); startCamera();
  });

  analyzeBtn.addEventListener('click', () => { if (capturedURL) sendAnalysis(capturedURL, false); });
})();

/* ── Shared analysis sender ───────────────────────────────────────────────── */
async function sendAnalysis(data, isFile) {
  clearError(); setLoading(true);
  try {
    const opts = isFile
      ? { method: 'POST', body: data }
      : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: data }) };
    const res    = await fetch('/api/analyze', opts);
    const result = await res.json();
    if (!result.ok) throw new Error(result.error || 'Analysis failed.');
    window.location.href = '/prediction';
  } catch (err) {
    showError(err.message || 'Something went wrong. Please try again.');
    setLoading(false);
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   MANUAL ENTRY PAGE
   ══════════════════════════════════════════════════════════════════════════ */
(function initManual() {
  const form       = document.getElementById('manualForm');
  if (!form) return;

  const ageInput       = document.getElementById('ageInput');
  const agePreview     = document.getElementById('ageGroupPreview');
  const occasionGrid   = document.getElementById('occasionGrid');
  const occasionHidden = document.getElementById('occasionHidden');
  const submitBtn      = document.getElementById('submitBtn');

  function validate() {
    const gender    = form.querySelector('input[name="gender"]:checked');
    const skinTone  = form.querySelector('input[name="skin_tone"]:checked');
    const age       = parseInt(ageInput.value, 10);
    const occasion  = occasionHidden.value;
    submitBtn.disabled = !(gender && skinTone && age >= 1 && age <= 99 && occasion);
  }

  // Age preview
  ageInput.addEventListener('input', () => {
    const age = parseInt(ageInput.value, 10);
    agePreview.textContent = (!isNaN(age) && age >= 1 && age <= 99) ? `→ ${ageToGroup(age)}` : '';
    validate();
  });

  // Gender + tone change
  form.querySelectorAll('input[name="gender"], input[name="skin_tone"]').forEach(el => el.addEventListener('change', validate));

  // Occasion chips
  if (occasionGrid) {
    occasionGrid.addEventListener('click', (e) => {
      const chip = e.target.closest('.occasion-chip');
      if (!chip) return;
      occasionGrid.querySelectorAll('.occasion-chip').forEach(c => { c.classList.remove('selected'); c.setAttribute('aria-pressed', 'false'); });
      chip.classList.add('selected'); chip.setAttribute('aria-pressed', 'true');
      occasionHidden.value = chip.dataset.occasion;
      validate();
    });
  }

  // Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearError(); setLoading(true);
    const payload = {
      gender:    form.querySelector('input[name="gender"]:checked')?.value,
      age:       parseInt(ageInput.value, 10),
      skin_tone: form.querySelector('input[name="skin_tone"]:checked')?.value,
      occasion:  occasionHidden.value,
    };
    try {
      const res    = await fetch('/api/manual', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const result = await res.json();
      if (!result.ok) throw new Error(result.error || 'Something went wrong.');
      window.location.href = '/recommendation';
    } catch (err) {
      showError(err.message); setLoading(false);
    }
  });
})();

/* ══════════════════════════════════════════════════════════════════════════
   PREDICTION PAGE — Occasion chips + inline edit
   ══════════════════════════════════════════════════════════════════════════ */
(function initPrediction() {
  const grid       = document.getElementById('occasionGrid');
  const hiddenSel  = document.getElementById('occasionSelect');
  const getLookBtn = document.getElementById('getLookBtn');
  if (!grid || !getLookBtn) return;

  let selectedOccasion = '';
  let needsInput = document.getElementById('predictionFlow')?.dataset.needsInput === 'true';

  // Occasion chips
  grid.addEventListener('click', (e) => {
    const chip = e.target.closest('.occasion-chip');
    if (!chip) return;
    grid.querySelectorAll('.occasion-chip').forEach(c => { c.classList.remove('selected'); c.setAttribute('aria-pressed', 'false'); });
    chip.classList.add('selected'); chip.setAttribute('aria-pressed', 'true');
    selectedOccasion = chip.dataset.occasion;
    if (hiddenSel) hiddenSel.value = selectedOccasion;
    getLookBtn.disabled = needsInput; clearError();
  });

  // Reveal outfit
  getLookBtn.addEventListener('click', async () => {
    if (!selectedOccasion) { showError('Please select an occasion first.'); return; }
    clearError(); setLoading(true);
    try {
      const res    = await fetch('/api/recommend', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ occasion: selectedOccasion }) });
      const result = await res.json();
      if (!result.ok) throw new Error(result.error || 'Failed to build recommendation.');
      window.location.href = '/recommendation';
    } catch (err) {
      showError(err.message); setLoading(false);
    }
  });

  // ── Inline attribute editors ──────────────────────────────────────────────
  document.querySelectorAll('.edit-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const field    = btn.dataset.field;
      const editorId = `editor${field.charAt(0).toUpperCase() + field.slice(1)}`;
      // Capitalise correctly for skin_tone → editorSkin_tone
      const editor   = document.getElementById(`editor${field.charAt(0).toUpperCase()}${field.slice(1)}`);
      if (editor) editor.hidden = !editor.hidden;
    });
  });

  document.querySelectorAll('.save-edit-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const field = btn.dataset.field;
      let value;

      if (field === 'age') {
        value = document.getElementById('editAgeInput')?.value;
      } else if (field === 'gender') {
        value = document.querySelector('input[name="editGender"]:checked')?.value;
        if (!value) { showError('Please select Male or Female.'); return; }
      } else if (field === 'skin_tone') {
        value = document.getElementById('editSkinSelect')?.value;
      }

      if (!value) { showError('Please enter a value.'); return; }
      clearError();

      try {
        const res    = await fetch('/api/edit-analysis', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ field, value }) });
        const result = await res.json();
        if (!result.ok) throw new Error(result.error);

        // Update the displayed value
        const displayEl = document.getElementById(`display${field.charAt(0).toUpperCase()}${field.slice(1)}`);
        if (displayEl) {
          if (field === 'age') displayEl.textContent = result.analysis.age_group;
          else if (field === 'gender') displayEl.textContent = result.analysis.gender;
          else if (field === 'skin_tone') displayEl.textContent = result.analysis.skin_tone;
        }

        if (field === 'age' || field === 'gender') {
          needsInput = result.analysis.needs_input;
          document.getElementById('predictionFlow').dataset.needsInput = String(needsInput);
          document.getElementById('modelUnavailableNote')?.toggleAttribute('hidden', !needsInput);
          const badgeText = document.getElementById('detectedBadgeText');
          if (badgeText) badgeText.textContent = needsInput
            ? 'Face detected — age and gender need your input'
            : 'Face detected — review and correct if needed';
          if (!needsInput) {
            if (field === 'age') document.getElementById('ageConfidence').childNodes[0].textContent = 'Entered by you · ';
            if (field === 'gender') document.getElementById('genderConfidence').textContent = 'Entered by you';
          }
          if (selectedOccasion) getLookBtn.disabled = needsInput;
        }

        // Hide editor
        const editor = document.getElementById(`editor${field.charAt(0).toUpperCase()}${field.slice(1)}`);
        if (editor) editor.hidden = true;

      } catch (err) {
        showError(err.message || 'Could not save change.');
      }
    });
  });
})();
