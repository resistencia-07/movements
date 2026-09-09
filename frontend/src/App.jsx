import { useEffect, useEffectEvent, useState } from 'react'
import './motion-lab.css'
import './light-theme.css'

const views = ['overview', 'camera-lab', 'analysis', 'compare', 'reports', 'settings']

function App() {
  const [view, setView] = useState('overview')
  const [session, setSession] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function request(path, options) {
    const isFormData = options?.body instanceof FormData
    const response = await fetch(`/api${path}`, {
      headers: isFormData ? {} : { 'Content-Type': 'application/json' },
      ...options,
    })
    const body = await response.text()
    let payload
    try {
      payload = body ? JSON.parse(body) : null
    } catch {
      throw new Error(`Backend returned an invalid response (${response.status})`)
    }
    if (!payload) {
      if (response.status === 502 || response.status === 503) {
        throw new Error('Backend is offline. Start app.py on port 5000.')
      }
      throw new Error(`Backend returned an empty response (${response.status})`)
    }
    if (!response.ok || payload.ok === false) throw new Error(payload.error || 'Backend request failed')
    return payload
  }

  async function refreshSession() {
    try {
      const payload = await request('/session')
      setSession(payload)
      setError('')
    } catch (requestError) {
      setError(requestError.message)
    }
  }

  async function control(action, camera) {
    setBusy(true)
    try {
      const body = { action }
      if (camera) body.camera = camera
      const payload = await request('/control', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      setSession(payload.session)
      setError('')
      if (action === 'start_practice') setView('practice')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  async function uploadSource(formData) {
    setBusy(true)
    try {
      const payload = await request('/source', { method: 'POST', body: formData })
      setError(payload.message || 'Reference ready. Restarting the dance engine.')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setBusy(false)
    }
  }

  const refreshSessionEvent = useEffectEvent(refreshSession)

  useEffect(() => {
    const initialLoad = window.setTimeout(refreshSessionEvent, 0)
    const timer = window.setInterval(refreshSessionEvent, 1500)
    return () => {
      window.clearTimeout(initialLoad)
      window.clearInterval(timer)
    }
  }, [])

  const cameraReady = Boolean(session?.camera_available)
  const practiceActive = Boolean(session?.practice_active)

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <a className="brand" href="#overview" onClick={() => setView('overview')}>
          <span className="brand-mark"><span /><span /><span /></span>
          <span><strong>MOTION</strong><small>LAB / REACT</small></span>
        </a>
        <div className="sidebar-section">
          <p className="eyebrow">Workspace</p>
          <nav className="nav-list">
            {views.map((item) => (
              <button className={`nav-link ${view === item ? 'active' : ''}`} key={item} onClick={() => setView(item)}>
                <span className="nav-icon">{item === 'camera-lab' ? '[]' : item === 'analysis' ? '/' : item === 'compare' ? '<>' : item === 'reports' ? '=' : '+'}</span>
                {item.replace('-', ' ')}
              </button>
            ))}
          </nav>
        </div>
        <div className="sidebar-bottom">
          <div className="system-status"><span className={`status-dot ${session ? '' : 'offline'}`} /><span><strong>{session ? 'Flask engine' : 'Backend offline'}</strong><small>{session ? 'Session connected' : 'Start app.py'}</small></span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="mobile-brand"><span className="brand-mark"><span /><span /><span /></span><strong>MOTION LAB</strong></div><span className="profile-copy"><strong>AI Dance Coach</strong><small>Flask + React session</small></span></header>
        <div className="content-wrap">
          {error && <div className="panel error-banner" role="alert">{error}</div>}
          {(view === 'camera-lab' || view === 'practice') && <ReferenceSource busy={busy} onSubmit={uploadSource} />}
          {(view === 'camera-lab' || view === 'practice') && <CameraSelector session={session} busy={busy} onSelect={(camera) => control('open_camera', camera)} />}
          {view === 'overview' && <Overview session={session} onOpen={() => setView('camera-lab')} />}
          {view === 'camera-lab' && <CameraLab session={session} busy={busy} onControl={control} onPractice={() => setView('practice')} />}
          {view === 'practice' && <Practice session={session} busy={busy} onControl={control} />}
          {view === 'analysis' && <Placeholder title="Analysis Studio" text="Analysis state is supplied by the existing pose engine." />}
          {view === 'compare' && <Placeholder title="Session Compare" text="Compare captured sessions after practice." />}
          {view === 'reports' && <Placeholder title="Movement Reports" text="Reports remain available from the existing workspace." />}
          {view === 'settings' && <Placeholder title="System Settings" text="Local processing settings." />}
          {!cameraReady && view === 'practice' && <p className="lede">Camera is not connected. Return to Camera Lab and open the camera.</p>}
          {practiceActive && view !== 'practice' && <button className="primary-button floating-practice" onClick={() => setView('practice')}>Return to practice</button>}
        </div>
      </main>
    </div>
  )
}

function Overview({ session, onOpen }) {
  return <section className="view active"><div className="page-heading"><div><p className="eyebrow">AI Dance Coach / React</p><h1>Make movement <em>measurable.</em></h1><p className="lede">Connect your camera, choose a dance reference, and practice against the existing MediaPipe pose engine.</p></div><button className="primary-button" onClick={onOpen}>Open Camera Lab <span>→</span></button></div><div className="metric-grid"><article className="metric-card accent-teal"><div className="metric-label">Backend</div><strong>{session ? 'Ready' : 'Offline'}</strong><span className="metric-foot">Flask API connection</span></article><article className="metric-card accent-blue"><div className="metric-label">Reference video</div><strong>{session?.source_video || 'dance.mp4'}</strong><span className="metric-foot">{session?.detected_frames || 0} detected frames</span></article><article className="metric-card accent-lime"><div className="metric-label">Practice step</div><strong>{session?.current_step || 1}<small> / {session?.total_steps || 0}</small></strong><span className="metric-foot">Pose sequence</span></article><article className="metric-card accent-amber"><div className="metric-label">Camera</div><strong>{session?.camera_available ? 'Live' : 'Closed'}</strong><span className="metric-foot">{session?.camera_source || 'Not opened'}</span></article></div><section className="panel"><div className="panel-heading"><div><p className="eyebrow">Pipeline</p><h2>Browser to pose engine</h2></div></div><p className="lede">React calls Flask through relative `/api` requests. Flask forwards controls to the existing OpenCV and MediaPipe runtime.</p></section></section>
}

function CameraLab({ session, busy, onControl, onPractice }) {
  const ready = Boolean(session?.camera_available)
  return <section className="view active"><div className="page-heading"><div><p className="eyebrow">Capture workspace</p><h1>Camera <em>Lab</em></h1><p className="lede">Open the server camera, then start the dance practice session.</p></div><div className="heading-actions"><span className="capture-ready"><i className="live-pulse" /> {ready ? 'Camera ready' : 'Camera closed'}</span><button className="primary-button" disabled={busy} onClick={() => onControl(ready ? 'start_practice' : 'open_camera')}>{busy ? 'Connecting...' : ready ? 'Start practice' : 'Open camera'}</button></div></div><div className="camera-layout"><div className="camera-grid"><article className="camera-card live-camera"><div className="camera-preview"><div className="scan-lines" /><div className="preview-skeleton"><span className="head" /><span className="torso" /><span className="limb l1" /><span className="limb l2" /><span className="limb l3" /><span className="limb l4" /></div><span className="preview-label">{ready ? 'LIVE / FRONT' : 'WAITING FOR CAMERA'}</span><span className="confidence">{ready ? session.camera_source : 'No device selected'}</span></div><div className="camera-card-foot"><span><strong>Camera 1</strong><small>Front view / Primary</small></span><span className={ready ? 'connected' : 'disconnected'}>{ready ? 'Connected' : 'Not connected'}</span></div></article></div><aside className="capture-sidebar"><section className="panel setup-panel"><div className="panel-heading"><div><p className="eyebrow">Session setup</p><h2>{ready ? 'Ready when you are' : 'Open camera first'}</h2></div></div><div className="calibration-list"><div><span className="check-icon">✓</span><span><strong>Pose model</strong><small>MediaPipe landmark engine</small></span><em>Ready</em></div><div><span className="check-icon">✓</span><span><strong>Reference</strong><small>{session?.source_video || 'dance.mp4'}</small></span><em>Ready</em></div></div>{ready && <button className="secondary-button wide-button" onClick={() => { onControl('start_practice'); onPractice() }}>Practice with camera <span>→</span></button>}</section></aside></div></section>
}

function Practice({ session, busy, onControl }) {
  return <section className="view active"><div className="page-heading"><div><p className="eyebrow">Live practice / Step {session?.current_step || 1}</p><h1>Dance <em>Practice</em></h1><p className="lede">Your backend camera stays inside this page while the existing engine scores your movement.</p></div><button className="secondary-button" disabled={busy} onClick={() => onControl('stop_practice')}>Stop practice</button></div><section className="panel practice-stage"><div className="browser-camera"><img src="/api/camera-stream" alt="Live dance practice camera" /><span className="camera-feed-label">LIVE / BACKEND CAMERA</span></div><div className="practice-reference"><div className="canvas-grid" /><div className="analysis-figure"><span className="figure-head" /><span className="figure-body" /><span className="figure-limb a" /><span className="figure-limb b" /><span className="figure-limb c" /><span className="figure-limb d" /></div><span className="canvas-tag">{session?.body_detected ? 'BODY DETECTED' : 'SHOW FULL BODY'}</span><span className="frame-time">REFERENCE / STEP {session?.current_step || 1}</span></div></section><div className="metric-grid"><article className="metric-card accent-teal"><div className="metric-label">Pose score</div><strong>{Math.round(session?.score || 0)}%</strong><span className="metric-foot">Live backend score</span></article><article className="metric-card accent-blue"><div className="metric-label">Language</div><strong>{session?.language || 'ENGLISH'}</strong><span className="metric-foot">Voice coach</span></article></div></section>
}

function Placeholder({ title, text }) {
  return <section className="view active"><div className="page-heading"><div><p className="eyebrow">Motion Lab workspace</p><h1>{title}</h1><p className="lede">{text}</p></div></div><section className="panel"><h2>Connected to Flask</h2><p className="lede">This view keeps the existing workspace route available while the live capture pipeline remains focused on Camera Lab and Practice.</p></section></section>
}

function CameraSelector({ session, busy, onSelect }) {
  const options = session?.camera_options ?? []
  const [selectedCamera, setSelectedCamera] = useState('')
  const displayedCamera = selectedCamera || session?.camera_source || options[0] || ''

  return <div className="camera-selector panel"><div><p className="eyebrow">Input device</p><strong>Choose a camera</strong><small>{options.length ? `${options.length} camera${options.length === 1 ? '' : 's'} detected` : 'No compatible camera detected'}</small></div><select aria-label="Choose camera device" value={displayedCamera} disabled={busy || !options.length} onChange={(event) => { setSelectedCamera(event.target.value); onSelect(event.target.value) }}><option value="">No camera detected</option>{options.map((option, index) => <option key={option} value={option}>Camera {index + 1}</option>)}</select></div>
}

function ReferenceSource({ busy, onSubmit }) {
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [fileName, setFileName] = useState('')

  function submitFile(event) {
    const file = event.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    const formData = new FormData()
    formData.append('video', file)
    onSubmit(formData)
  }

  function submitYoutube(event) {
    event.preventDefault()
    if (!youtubeUrl.trim()) return
    const formData = new FormData()
    formData.append('youtube_url', youtubeUrl.trim())
    onSubmit(formData)
  }

  return <section className="reference-source panel"><div><p className="eyebrow">Dance reference</p><strong>Choose what to practice</strong><small>Upload a video or prepare a YouTube reference before starting.</small></div><div className="reference-source-actions"><label className="secondary-button file-button">{fileName || 'Upload video'}<input type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm" disabled={busy} onChange={submitFile} /></label><form className="youtube-form" onSubmit={submitYoutube}><input type="url" aria-label="YouTube URL" placeholder="Paste YouTube link" value={youtubeUrl} disabled={busy} onChange={(event) => setYoutubeUrl(event.target.value)} /><button className="secondary-button" type="submit" disabled={busy || !youtubeUrl.trim()}>Use link</button></form></div></section>
}

export default App
