import React, { useState, useRef } from 'react';
import { useNER } from './hooks/useNER';
import TokenDisplay from './components/TokenDisplay';
import EntityLegend from './components/EntityLegend';
import StatsBar     from './components/StatsBar';

const EXAMPLES = [
  'ဘူးသီးတောင်မြို့နယ် မှ ဒေသခံ မှိုင်းဝေ က PDF အဖွဲ့ ကို ဝေဖန်ခဲ့သည်',
  'မန္တလေးမြို့ ရှိ ကုလသမဂ္ဂ ရုံး မှ ၂၀၂၅ ခုနှစ် ဇန်နဝါရီ ၁ ရက် တွင် ထုတ်ပြန်ခဲ့သည်',
  'စစ်ကိုင်းတိုင်းဒေသကြီး မြောင်မြို့နယ် ရွာသာကြီးရွာ မှ ပြည်သူများ ထွက်ပြေးခဲ့ကြသည်',
];

export default function App() {
  const [text, setText]     = useState('');
  const { result, loading, error, predict, reset } = useNER();
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e?.preventDefault();
    predict(text);
  };

  const handleExample = (ex) => {
    setText(ex);
    reset();
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit();
  };

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>

      {/* ── Noise grain overlay */}
      <div style={{
        position:   'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
        opacity:    0.025,
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
      }} />

      {/* ── Gradient blobs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: '-20%', left: '-10%',
          width: '600px', height: '600px', borderRadius: '50%',
          background: 'radial-gradient(circle, #38bdf815 0%, transparent 70%)',
        }} />
        <div style={{
          position: 'absolute', bottom: '-20%', right: '-10%',
          width: '500px', height: '500px', borderRadius: '50%',
          background: 'radial-gradient(circle, #818cf810 0%, transparent 70%)',
        }} />
      </div>

      <div style={{ position: 'relative', zIndex: 1, flex: 1, display: 'flex', flexDirection: 'column' }}>

        {/* ── Header */}
        <header style={{
          borderBottom: '1px solid var(--border)',
          padding:      '1rem 2rem',
          display:      'flex',
          alignItems:   'center',
          gap:          '1rem',
          backdropFilter: 'blur(12px)',
          background:   'rgba(6,13,26,.7)',
          position:     'sticky', top: 0, zIndex: 10,
        }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem', flexShrink: 0,
          }}>
            ✦
          </div>
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: '.8rem',
              color: 'var(--muted)', letterSpacing: '.12em', textTransform: 'uppercase',
            }}>
              Myanmar NER
            </div>
            <div style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem', color: 'var(--text)', lineHeight: 1 }}>
              Named Entity Recognition
            </div>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '.5rem', alignItems: 'center' }}>
            <StatusDot />
          </div>
        </header>

        {/* ── Main */}
        <main style={{ flex: 1, maxWidth: '860px', width: '100%', margin: '0 auto', padding: '2rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

          {/* Title block */}
          <div style={{ textAlign: 'center', padding: '1rem 0 .5rem', animation: 'fadeUp 400ms ease both' }}>
            <h1 style={{
              fontFamily: 'var(--font-serif)', fontSize: 'clamp(1.8rem, 4vw, 2.8rem)',
              fontWeight: 400, color: 'var(--text)', letterSpacing: '-.01em', lineHeight: 1.2,
            }}>
              Annotate <em style={{ color: 'var(--accent)' }}>Myanmar</em> text
            </h1>
            <p style={{ color: 'var(--muted)', fontSize: '.9rem', marginTop: '.5rem' }}>
              Paste or type Burmese text to identify persons, organisations, locations, dates, and more.
            </p>
          </div>

          {/* Input form */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '.75rem', animation: 'fadeUp 450ms ease both' }}>
            <div style={{ position: 'relative' }}>
              <textarea
                ref={textareaRef}
                value={text}
                onChange={e => { setText(e.target.value); if (result) reset(); }}
                onKeyDown={handleKeyDown}
                placeholder="ဤနေရာတွင် မြန်မာစာ ရိုက်ထည့်ပါ…"
                rows={4}
                style={{
                  width:        '100%',
                  background:   'var(--surface)',
                  border:       '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  padding:      '1rem',
                  color:        'var(--text)',
                  fontSize:     '1rem',
                  fontFamily:   'var(--font-body)',
                  resize:       'vertical',
                  outline:      'none',
                  lineHeight:   1.8,
                  transition:   'border-color var(--transition)',
                }}
                onFocus={e  => e.target.style.borderColor = 'var(--accent)'}
                onBlur={e   => e.target.style.borderColor = 'var(--border)'}
              />
              <div style={{
                position:   'absolute', bottom: '10px', right: '12px',
                fontSize:   '.65rem', color: 'var(--muted)', fontFamily: 'var(--font-mono)',
                pointerEvents: 'none',
              }}>
                {text.trim().split(/\s+/).filter(Boolean).length} words · Ctrl+Enter to run
              </div>
            </div>

            <div style={{ display: 'flex', gap: '.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <button
                type="submit"
                disabled={loading || !text.trim()}
                style={{
                  padding:      '.6rem 1.5rem',
                  background:   loading ? 'var(--surface2)' : 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  border:       'none',
                  borderRadius: 'var(--radius)',
                  color:        loading ? 'var(--muted)' : '#060d1a',
                  fontFamily:   'var(--font-mono)',
                  fontSize:     '.8rem',
                  fontWeight:   700,
                  cursor:       loading || !text.trim() ? 'not-allowed' : 'pointer',
                  letterSpacing: '.05em',
                  transition:   'opacity var(--transition)',
                  display:      'flex', alignItems: 'center', gap: '.5rem',
                }}
              >
                {loading && <Spinner />}
                {loading ? 'Analysing…' : 'Analyse →'}
              </button>

              {text && (
                <button
                  type="button"
                  onClick={() => { setText(''); reset(); }}
                  style={{
                    padding:    '.6rem 1rem',
                    background: 'transparent',
                    border:     '1px solid var(--border)',
                    borderRadius: 'var(--radius)',
                    color:      'var(--muted)',
                    fontFamily: 'var(--font-mono)',
                    fontSize:   '.8rem',
                    cursor:     'pointer',
                  }}
                >
                  Clear
                </button>
              )}

              <span style={{ color: 'var(--muted)', fontSize: '.75rem', fontFamily: 'var(--font-mono)' }}>
                Examples:
              </span>
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => handleExample(ex)}
                  style={{
                    padding:    '.35rem .75rem',
                    background: 'var(--surface2)',
                    border:     '1px solid var(--border)',
                    borderRadius: '99px',
                    color:      'var(--muted)',
                    fontFamily: 'var(--font-mono)',
                    fontSize:   '.72rem',
                    cursor:     'pointer',
                    transition: 'color var(--transition)',
                  }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--accent)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--muted)'}
                >
                  #{i + 1}
                </button>
              ))}
            </div>
          </form>

          {/* Error */}
          {error && (
            <div style={{
              padding:      '.85rem 1rem',
              background:   '#f8717118',
              border:       '1px solid #f8717150',
              borderRadius: 'var(--radius)',
              color:        '#f87171',
              fontFamily:   'var(--font-mono)',
              fontSize:     '.85rem',
              animation:    'fadeUp 200ms ease both',
            }}>
              ✗ {error}
            </div>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div style={{
              height:       '80px',
              background:   `linear-gradient(90deg, var(--surface) 25%, var(--surface2) 50%, var(--surface) 75%)`,
              backgroundSize: '400px 100%',
              animation:    'shimmer 1.2s ease infinite',
              borderRadius: 'var(--radius)',
              border:       '1px solid var(--border)',
            }} />
          )}

          {/* Results */}
          {result && !loading && (
            <>
              <StatsBar result={result} />
              <TokenDisplay tokens={result.tokens} />
              <EntityLegend activeCounts={result.entity_counts} />
            </>
          )}

          {/* Initial legend */}
          {!result && !loading && (
            <EntityLegend activeCounts={null} />
          )}

        </main>

        {/* ── Footer */}
        <footer style={{
          borderTop:  '1px solid var(--border)',
          padding:    '1rem 2rem',
          textAlign:  'center',
          color:      'var(--muted)',
          fontSize:   '.72rem',
          fontFamily: 'var(--font-mono)',
          letterSpacing: '.05em',
        }}>
          Myanmar NER · BiLSTM-CRF + CharCNN · BIOES tagging ·{' '}
          <a href="https://github.com/your-username/myanmar-ner" target="_blank" rel="noreferrer"
            style={{ color: 'var(--accent)', textDecoration: 'none' }}>GitHub</a>
        </footer>

      </div>
    </div>
  );
}

function StatusDot() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
      <div style={{ position: 'relative', width: '8px', height: '8px' }}>
        <div style={{
          position: 'absolute', inset: 0, borderRadius: '50%',
          background: '#34d399',
          animation: 'pulse-ring 1.8s ease-out infinite',
        }} />
        <div style={{ position: 'absolute', inset: '1px', borderRadius: '50%', background: '#34d399' }} />
      </div>
      <span style={{ fontSize: '.7rem', color: '#34d399', fontFamily: 'var(--font-mono)' }}>API live</span>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{
      width: '14px', height: '14px',
      border: '2px solid transparent',
      borderTop: '2px solid currentColor',
      borderRadius: '50%',
      animation: 'spin 600ms linear infinite',
    }} />
  );
}
