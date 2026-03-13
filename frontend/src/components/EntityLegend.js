import React from 'react';

const ALL_ENTITIES = [
  { type: 'PER',          label: 'Person',    color: '#f87171' },
  { type: 'ORG',          label: 'Org',       color: '#34d399' },
  { type: 'DATE',         label: 'Date',      color: '#fbbf24' },
  { type: 'TIME',         label: 'Time',      color: '#fb923c' },
  { type: 'NUM',          label: 'Number',    color: '#c084fc' },
  { type: 'LOC',          label: 'Location',  color: '#60a5fa' },
  { type: 'LOC-COUNTRY',  label: 'Country',   color: '#3b82f6' },
  { type: 'LOC-STATE',    label: 'State',     color: '#6ee7b7' },
  { type: 'LOC-DISTRICT', label: 'District',  color: '#5eead4' },
  { type: 'LOC-TOWNSHIP', label: 'Township',  color: '#38bdf8' },
  { type: 'LOC-CITY',     label: 'City',      color: '#7dd3fc' },
  { type: 'LOC-VILLAGE',  label: 'Village',   color: '#86efac' },
  { type: 'LOC-WARD',     label: 'Ward',      color: '#93c5fd' },
];

export default function EntityLegend({ activeCounts }) {
  const hasResult = activeCounts && Object.keys(activeCounts).length > 0;

  return (
    <div style={{
      background:   'var(--surface)',
      border:       '1px solid var(--border)',
      borderRadius: 'var(--radius)',
      padding:      '1rem 1.25rem',
    }}>
      <div style={{
        fontSize:      '.7rem',
        color:         'var(--muted)',
        textTransform: 'uppercase',
        letterSpacing: '.1em',
        marginBottom:  '.75rem',
        fontFamily:    'var(--font-mono)',
      }}>
        Entity Types
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem' }}>
        {ALL_ENTITIES.map(({ type, label, color }) => {
          const count = activeCounts?.[type];
          const isActive = hasResult ? !!count : true;
          return (
            <div
              key={type}
              style={{
                display:      'flex',
                alignItems:   'center',
                gap:          '5px',
                padding:      '3px 8px',
                borderRadius: '99px',
                background:   isActive ? `${color}18` : 'transparent',
                border:       `1px solid ${isActive ? `${color}50` : 'var(--border)'}`,
                opacity:      hasResult && !isActive ? 0.35 : 1,
                transition:   'all 250ms ease',
              }}
            >
              <span style={{
                width:       '7px',
                height:      '7px',
                borderRadius:'50%',
                background:  color,
                flexShrink:   0,
              }} />
              <span style={{
                fontSize:   '.72rem',
                color:      isActive ? color : 'var(--muted)',
                fontFamily: 'var(--font-mono)',
                whiteSpace: 'nowrap',
              }}>
                {label}
                {count ? <span style={{ marginLeft: '4px', opacity: .7 }}>×{count}</span> : null}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
