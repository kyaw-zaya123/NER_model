import React, { useState } from 'react';

const BIOES_LABEL = { 'B': 'begin', 'I': 'inside', 'E': 'end', 'S': 'single', 'O': 'other' };

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

function Token({ token, isActive, onClick }) {
  const { word, tag, entity_type, color, is_entity } = token;
  const prefix = tag.split('-')[0];

  const style = is_entity ? {
    background:   `rgba(${hexToRgb(color)}, 0.15)`,
    borderColor:  `rgba(${hexToRgb(color)}, 0.5)`,
    color:        color,
    borderWidth:  prefix === 'S' ? '1px' : undefined,
    borderLeftWidth:  (prefix === 'B' || prefix === 'S') ? '3px' : '0',
    borderRightWidth: (prefix === 'E' || prefix === 'S') ? '1px' : '0',
    borderTopWidth:   '1px',
    borderBottomWidth:'1px',
    borderRadius:
      prefix === 'S' ? '6px' :
      prefix === 'B' ? '6px 0 0 6px' :
      prefix === 'E' ? '0 6px 6px 0' : '0',
    marginLeft:  prefix === 'B' || prefix === 'S' ? '2px' : '0',
    marginRight: prefix === 'E' || prefix === 'S' ? '2px' : '0',
    boxShadow:   isActive ? `0 0 0 2px ${color}60` : 'none',
  } : {};

  return (
    <span
      onClick={() => is_entity && onClick(token)}
      style={{
        display:     'inline-block',
        padding:     is_entity ? '2px 6px' : '2px 3px',
        cursor:      is_entity ? 'pointer' : 'default',
        fontSize:    '1.05rem',
        lineHeight:  '1.9',
        fontFamily:  'var(--font-body)',
        borderStyle: 'solid',
        borderColor: 'transparent',
        transition:  'all 140ms ease',
        userSelect:  'none',
        ...style,
      }}
      title={is_entity ? `${entity_type} (${BIOES_LABEL[prefix] || prefix})` : undefined}
    >
      {word}
    </span>
  );
}

export default function TokenDisplay({ tokens }) {
  const [active, setActive] = useState(null);

  const handleClick = (token) => {
    setActive(prev =>
      prev && prev.word === token.word && prev.tag === token.tag ? null : token
    );
  };

  return (
    <div style={{ animation: 'fadeUp 300ms ease both' }}>
      {/* Annotated text */}
      <div style={{
        background:   'var(--surface2)',
        border:       '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding:      '1.5rem',
        lineHeight:   '2.2',
        letterSpacing: '.02em',
        minHeight:    '80px',
      }}>
        {tokens.map((tok, i) => (
          <Token
            key={i}
            token={tok}
            isActive={active && active === tok}
            onClick={handleClick}
          />
        ))}
      </div>

      {/* Tooltip for active entity */}
      {active && (
        <div style={{
          marginTop:    '1rem',
          padding:      '0.75rem 1rem',
          background:   `rgba(${hexToRgb(active.color)}, 0.1)`,
          border:       `1px solid rgba(${hexToRgb(active.color)}, 0.4)`,
          borderLeft:   `3px solid ${active.color}`,
          borderRadius: 'var(--radius)',
          display:      'flex',
          gap:          '1.5rem',
          flexWrap:     'wrap',
          animation:    'fadeUp 180ms ease both',
        }}>
          <Detail label="Word"  value={active.word}        color={active.color} />
          <Detail label="Tag"   value={active.tag}         color={active.color} />
          <Detail label="Type"  value={active.entity_type} color={active.color} />
          <Detail label="BIOES" value={BIOES_LABEL[active.tag.split('-')[0]] || '—'} color={active.color} />
        </div>
      )}
    </div>
  );
}

function Detail({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: '.7rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: '2px' }}>
        {label}
      </div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '.85rem', color }}>
        {value || '—'}
      </div>
    </div>
  );
}
