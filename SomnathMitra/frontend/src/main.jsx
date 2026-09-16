import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ArrowUp, BookOpen, ChevronDown, CircleHelp, Clock3, Compass, Copy, ExternalLink, Menu, RotateCcw, Sparkles, X } from 'lucide-react'
import './styles.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.100.182:7777'
const examples = [
  { icon: '✦', text: 'What are the darshan timings at Somnath temple?' },
  { icon: '⌁', text: 'How can I get to Somnath from Mumbai?' },
  { icon: '◒', text: 'Find good vegetarian restaurants nearby' },
]

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeDetails, setActiveDetails] = useState(null)
  const [mobileMenu, setMobileMenu] = useState(false)
  const endRef = useRef(null)

  // Effects must return either nothing or a cleanup function. Keep the
  // scroll action inside a block so the effect never returns the DOM method's
  // value (some browser implementations do return a value here).
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function ask(question) {
    const q = question.trim()
    if (!q || loading) return
    setInput(''); setError(''); setLoading(true)
    setMessages(prev => [...prev, { role: 'user', text: q }])
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q, history: [], stream: true, user_lat: 0, user_lon: 0 }),
      })

      if (!response.ok) {
        throw new Error(`The guide returned an error (${response.status}).`)
      }

      // Add an empty assistant message that we'll fill in as tokens arrive
      setMessages(prev => [...prev, { role: 'assistant', text: '', data: null }])

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let firstChunk = true

      while (true) {
        const { done, value } = await reader.read()

        if (done) {
          // Final pass: extract metadata after the \x00 delimiter
          const nullIdx = buffer.indexOf('\x00')
          if (nullIdx >= 0) {
            const text = buffer.slice(0, nullIdx)
            let meta = null
            try { meta = JSON.parse(buffer.slice(nullIdx + 1)) } catch { /* no-op */ }
            setMessages(prev => {
              const msgs = [...prev]
              msgs[msgs.length - 1] = { role: 'assistant', text, data: meta }
              return msgs
            })
          }
          break
        }

        if (firstChunk) { setLoading(false); firstChunk = false }

        buffer += decoder.decode(value, { stream: true })
        // Show only the text portion while streaming (before the metadata delimiter)
        const nullIdx = buffer.indexOf('\x00')
        const displayText = nullIdx >= 0 ? buffer.slice(0, nullIdx) : buffer
        setMessages(prev => {
          const msgs = [...prev]
          msgs[msgs.length - 1] = { role: 'assistant', text: displayText, data: null }
          return msgs
        })
      }
    } catch (err) {
      setError(
        `${err instanceof Error ? err.message : 'Unknown error'} Check that the GPU backend is running and reachable.`
      )
    } finally {
      setLoading(false)
    }
  }

  function reset() { setMessages([]); setError(''); setActiveDetails(null) }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileMenu ? 'mobile-open' : ''}`}>
      <div className="brand"><div className="brand-mark">ॐ</div><div><div className="brand-name">Somnath <span>Mitra</span></div><div className="brand-subtitle">Your pilgrimage companion</div></div><button className="mobile-close" onClick={() => setMobileMenu(false)}><X size={18} /></button></div>
      <div className="sidebar-content">
        <button className="new-chat" onClick={reset}><Sparkles size={17} /> New conversation <span>⌘ K</span></button>
        <div className="sidebar-label">Explore</div>
        <div className="nav-item active"><Compass size={17} /> Ask Somnath Mitra</div>
        <div className="nav-item"><BookOpen size={17} /> Temple guide</div>
        <div className="nav-item"><CircleHelp size={17} /> Help &amp; information</div>
      </div>
      <div className="sidebar-footer"><div className="status-dot" /><div><strong>Connected to Mitra</strong><small>GPU backend · online</small></div><div className="online-pulse" /></div>
    </aside>
    <main className="main-panel">
      <header className="topbar"><button className="menu-button" onClick={() => setMobileMenu(true)}><Menu size={20} /></button><div className="crumb">Somnath Mitra <span>/</span> <b>New conversation</b></div><div className="top-actions"><button title="Start new conversation" onClick={reset}><RotateCcw size={17} /></button><span className="avatar">SM</span></div></header>
      <div className="chat-area">
        {messages.length === 0 ? <Welcome onAsk={ask} /> : <div className="conversation"><div className="conversation-date">TODAY</div>{messages.map((message, i) => <Message key={i} message={message} onDetails={setActiveDetails} />)}<div className="blessing-footer">🙏 May you be blessed by Pratham Jotirling, Shri Somnath Dada</div></div>}
        {loading && <div className="typing-row"><div className="mitra-mini">ॐ</div><div className="typing"><i /><i /><i /></div><span>Mitra is thinking…</span></div>}
        {error && <div className="error-banner"><CircleHelp size={17} /><span>{error}</span><button onClick={() => setError('')}><X size={15} /></button></div>}
        <div ref={endRef} />
      </div>
      <div className="composer-wrap"><form className="composer" onSubmit={e => { e.preventDefault(); ask(input) }}><textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(input) } }} placeholder="Ask anything about Somnath…" rows="1" /><button className="send-button" disabled={!input.trim() || loading} aria-label="Send"><ArrowUp size={18} /></button></form><div className="composer-hint"><span><span className="key">↵</span> to send <span className="key">⇧ ↵</span> for new line</span><span>Mitra can make mistakes. Verify important details.</span></div></div>
    </main>
    {activeDetails && <Details data={activeDetails} onClose={() => setActiveDetails(null)} />}</div>
}

function Welcome({ onAsk }) { return <div className="welcome"><div className="welcome-symbol">ॐ</div><div className="eyebrow">WELCOME TO SOMNATH</div><h1>How can I help<br /><em>your journey?</em></h1><p className="welcome-copy">I’m Mitra — your thoughtful guide to the temple,<br className="desktop-only" /> the town, and the road to Somnath.</p><div className="example-grid">{examples.map(example => <button key={example.text} onClick={() => onAsk(example.text)}><span className="example-icon">{example.icon}</span><span>{example.text}</span><ArrowUp size={15} /></button>)}</div></div> }

function ProductCards({ products }) {
  return (
    <div className="product-strip">
      {products.map((p, i) => (
        <a key={i} href={p.product_url} target="_blank" rel="noreferrer" className="product-card">
          <div className="product-img-wrap">
            <img src={p.image_url} alt={p.name} loading="lazy" />
          </div>
          <div className="product-info">
            <div className="product-name">{p.name}</div>
            <div className="product-price">₹{p.price}</div>
          </div>
        </a>
      ))}
    </div>
  )
}

function Message({ message, onDetails }) {
  const isUser = message.role === 'user'
  const hasProducts = !isUser && message.data?.products?.length > 0
  return (
    <div className={`message-row ${isUser ? 'user-row' : ''}`}>
      <div className="message-avatar">{isUser ? 'SM' : 'ॐ'}</div>
      <div className="message-body">
        <div className="message-meta">{isUser ? 'You' : 'Mitra'} <span>·</span> {isUser ? 'Just now' : 'Somnath guide'}</div>
        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          {isUser ? message.text : <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>}
        </div>
        {hasProducts && <ProductCards products={message.data.products} />}
        {!isUser && message.data && (
          <div className="message-tools">
            <button onClick={() => navigator.clipboard?.writeText(message.text)}><Copy size={13} /> Copy</button>
            <button onClick={() => onDetails(message.data)}><ChevronDown size={14} /> Details</button>
            {message.data.sources?.length > 0 && <span className="source-count"><BookOpen size={13} /> {message.data.sources.length} source{message.data.sources.length > 1 ? 's' : ''}</span>}
          </div>
        )}
      </div>
    </div>
  )
}

function Details({ data, onClose }) { return <div className="details-overlay" onClick={onClose}><section className="details-panel" onClick={e => e.stopPropagation()}><div className="details-heading"><div><div className="eyebrow">RESPONSE DETAILS</div><h2>How Mitra found this</h2></div><button onClick={onClose}><X size={18} /></button></div><div className="metric-row"><div><Clock3 size={16} /><span>Response time</span><strong>{data.elapsed_ms} ms</strong></div><div><Compass size={16} /><span>Intent</span><strong>{data.intent?.intent || 'General guide'}</strong></div></div><div className="details-section"><h3>Sources</h3>{data.sources?.length ? data.sources.map(source => <a className="source" key={source.chunk_id} href={source.page_url || '#'} target="_blank" rel="noreferrer"><BookOpen size={15} /><span>{source.heading || 'Somnath knowledge base'}<small>Relevance score {source.score}</small></span><ExternalLink size={14} /></a>) : <p className="muted">This answer was generated from Mitra’s general knowledge.</p>}</div></section></div> }

createRoot(document.getElementById('root')).render(<App />)
