import React from 'react';

function Stat({ label, value, accent }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize:   '1.4rem',
        fontFamily: 'var(--font-mono)',
        fontWeight: 700,
        color:      accent || 'var(--accent)',
        lineHeight:  1,
      }}>
        {value}
      </div>
      <div style={{
        fontSize:      '.68rem',
        color:         'var(--muted)',
        textTransform: 'uppercase',
        letterSpacing: '.1em',
        marginTop:     '4px',
      }}>
        {label}
      </div>
    </div>
  );
}

export default function StatsBar({ result }) {
  if (!result) return null;
  return (
    <div style={{
      display:        'flex',
      gap:            '2rem',
      justifyContent: 'center',
      padding:        '.85rem 1.5rem',
      background:     'var(--surface)',
      border:         '1px solid var(--border)',
      borderRadius:   'var(--radius)',
      animation:      'fadeUp 250ms ease both',
    }}>
      <Stat label="Tokens"   value={result.num_tokens}              accent="var(--accent)" />
      <Stat label="Entities" value={result.num_entities}            accent="#34d399" />
      <Stat label="Latency"  value={`${result.latency_ms} ms`}     accent="#fbbf24" />
      <Stat label="Model"    value={result.model_name.replace(/_/g, ' ')} accent="var(--accent2)" />
    </div>
  );
}
