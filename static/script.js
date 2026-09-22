/**
 * News Analyzer — Premium Frontend JavaScript
 * Handles loading state, counter animations, JSON toggle, and micro-interactions.
 */

document.addEventListener('DOMContentLoaded', function () {

    // ── Loading state on form submit ──────────────────────────────
    const form    = document.getElementById('search-form');
    const btn     = document.getElementById('analyze-btn');
    const overlay = document.getElementById('loading-overlay');
    const results = document.getElementById('results-area');

    if (form) {
        form.addEventListener('submit', function () {
            const input = document.getElementById('topic-input');
            if (input && input.value.trim() === '') return;

            if (overlay) overlay.classList.add('active');
            if (results) results.style.opacity = '0';
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span style="display:inline-flex;align-items:center;gap:8px"><span class="btn-spinner"></span> Analyzing...</span>';
            }
        });
    }

    // ── Animated stat counters ────────────────────────────────────
    function animateCounter(el) {
        const target = parseInt(el.textContent.trim(), 10);
        if (isNaN(target) || target === 0) return;

        const duration = 800;
        const start    = performance.now();

        function tick(now) {
            const elapsed  = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out quart
            const eased = 1 - Math.pow(1 - progress, 4);
            el.textContent = Math.round(eased * target);
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // Run counters for stat values (skip status text like SUCCESS/PARTIAL)
    document.querySelectorAll('.stat-value').forEach(el => {
        const text = el.textContent.trim();
        if (/^\d+$/.test(text)) animateCounter(el);
    });

    // ── JSON toggle ───────────────────────────────────────────────
    const jsonToggle = document.getElementById('json-toggle');
    const jsonOutput = document.getElementById('json-output');

    if (jsonToggle && jsonOutput) {
        jsonToggle.addEventListener('click', function () {
            const visible = jsonOutput.classList.toggle('visible');
            jsonToggle.innerHTML = visible
                ? '📄 Hide Raw JSON'
                : '📄 View Raw JSON';

            if (visible) {
                jsonOutput.style.maxHeight = '0';
                jsonOutput.style.overflow  = 'hidden';
                jsonOutput.style.transition = 'max-height 0.4s ease';
                requestAnimationFrame(() => {
                    jsonOutput.style.maxHeight = '2000px';
                });
            }
        });
    }

    // ── Search input — live character feedback ────────────────────
    const input = document.getElementById('topic-input');
    if (input && btn) {
        input.addEventListener('input', function () {
            btn.disabled = input.value.trim().length === 0;
        });
        // Set initial state
        btn.disabled = input.value.trim().length === 0;
    }

    // ── Scroll result cards into view smoothly ────────────────────
    const firstCard = document.querySelector('.result-card');
    if (firstCard) {
        setTimeout(() => {
            firstCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);
    }

    // ── Add ripple effect to buttons ──────────────────────────────
    document.querySelectorAll('.analyze-btn, .new-analysis-btn, .read-article-btn').forEach(btn => {
        btn.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            const rect   = btn.getBoundingClientRect();
            const size   = Math.max(rect.width, rect.height);

            ripple.style.cssText = `
                position: absolute;
                width: ${size}px; height: ${size}px;
                left: ${e.clientX - rect.left - size/2}px;
                top:  ${e.clientY - rect.top  - size/2}px;
                background: rgba(255,255,255,0.15);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple-expand 0.5s ease-out forwards;
                pointer-events: none;
            `;

            if (!btn.style.position || btn.style.position === 'static') {
                btn.style.position = 'relative';
            }
            btn.style.overflow = 'hidden';
            btn.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    });

});

// Inject ripple keyframe dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple-expand {
        to { transform: scale(2); opacity: 0; }
    }
    .btn-spinner {
        width: 14px; height: 14px;
        border: 2px solid rgba(255,255,255,0.3);
        border-top-color: #fff;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
        display: inline-block;
    }
`;
document.head.appendChild(style);
