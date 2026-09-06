// UI-feedback harness: extracts the REAL pure helpers introduced by the
// UI redesign from a generated vispath network HTML and runs them against
// headless Cytoscape — the toast queue, the dialog controller promise
// state machine, and the node-search matcher.
// Usage: node ui_feedback_harness.js <node-modules-dir> <path-to-network.html>
const cytoscape = require(process.argv[2] + '/node_modules/cytoscape');
const fs = require('fs');

const htmlPath = process.argv[3] || '/tmp/vispath-test/network_test.html';
const html = fs.readFileSync(htmlPath, 'utf8');

// Extract a top-level function declaration with balanced braces.
function extractFunction(name, source) {
    const marker = 'function ' + name + '(';
    const start = source.indexOf(marker);
    if (start === -1) throw new Error('function not found: ' + name);
    const open = source.indexOf('{', start);
    let depth = 0;
    for (let i = open; i < source.length; i++) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}') {
            depth--;
            if (depth === 0) return source.slice(start, i + 1);
        }
    }
    throw new Error('unbalanced braces: ' + name);
}

const FUNCTIONS = [
    'createToastQueue', 'createDialogController', 'matchNodes',
    'isVisibleElement', 'cycleSearchMatch',
];

// NOTE: sources come from the project's own generated HTML (trusted,
// locally-produced artifact); new Function only executes that code against
// the headless core + stubs.
function buildScope(cy) {
    const fnSources = FUNCTIONS.map(f => extractFunction(f, html)).join('\n');
    const src = fnSources + `
        return { createToastQueue, createDialogController, matchNodes };
    `;
    return new Function('cy', src)(cy);
}

// Scope for the search stepper: cycleSearchMatch runs against stubbed
// state so the test records WHICH match index gets applied.
function buildSearchScope() {
    const fnSource = extractFunction('cycleSearchMatch', html);
    const src = `
        let searchMatches = [];
        let searchIndex = -1;
        const applied = [];
        const counts = [];
        function updateSearchCount() {
            counts.push(searchMatches.length === 0 ? '' : (searchIndex + 1) + '/' + searchMatches.length);
        }
        function applySearchMatch() { applied.push(searchIndex); }
        ${fnSource}
        return {
            setMatches: (m) => { searchMatches = m; searchIndex = m.length > 0 ? 0 : -1; },
            cycle: (d) => cycleSearchMatch(d),
            getIndex: () => searchIndex,
            getApplied: () => applied,
            getCounts: () => counts,
        };
    `;
    return new Function('cy', src)();
}

// A(0,0), B(100,0), C(0,50) with labels; B additionally matches "bee".
function buildGraph() {
    const cy = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'N_A', node_type: 'source', label: 'Apple' } },
            { data: { id: 'N_B', node_type: 'intermediate', label: 'bee' } },
            { data: { id: 'N_C', node_type: 'target', label: 'Cherry' } },
            { group: 'edges', data: { id: 'e1', source: 'N_A', target: 'N_B', weight: 5 } },
        ],
    });
    return cy;
}

let failures = 0;
function check(name, got, expected) {
    const g = JSON.stringify(got);
    const e = JSON.stringify(expected);
    const pass = g === e;
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | got=' + g + (pass ? '' : ' expected=' + e));
    if (!pass) failures++;
}

// Promise-resolution assertions below need an async context.
(async function main() {

// ===== Test A: toast queue — ordering, cap, expiry =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    const q = api.createToastQueue({ maxVisible: 3, ttl: 3000, errorTtl: 6000 });
    let now = 0;
    q.push('one', 'info', null, now);
    q.push('two', 'success', null, now);
    q.push('three', 'warn', null, now);
    check('three toasts visible', q.active().length, 3);
    q.push('four', 'info', null, now);
    check('cap keeps newest 3', q.active().map(i => i.message), ['two', 'three', 'four']);
    q.push('error', 'error', null, now);
    // at t=3500 the plain toasts expired, the error (6000) survives
    const after = q.expire(3500);
    check('error outlives plain toasts', after.map(i => i.message), ['error']);
    check('expire before ttl keeps all', q.expire(4000).length, 1);
    check('expire after ttl empties', q.expire(9500).length, 0);
}

// ===== Test B: dialog controller — open/set/confirm and cancel =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    const ctl = api.createDialogController();
    const spec = {
        title: 'Edit node',
        fields: [
            { key: 'label', label: 'Label', type: 'text', value: 'A' },
            { key: 'type', label: 'Type', type: 'select', value: 'intermediate', options: ['source', 'intermediate', 'target'] },
        ],
    };
    let resolved = null, rejected = null;
    const promise = ctl.open(spec)
        .then(v => { resolved = v; })
        .catch(e => { rejected = e; });
    check('dialog is active', ctl.isActive(), true);
    check('second open rejected', ctl.open(spec), null);
    ctl.set('label', 'A2');
    ctl.set('type', 'target');
    const values = ctl.confirm();
    check('confirm returns values', [values.label, values.type], ['A2', 'target']);
    await promise;  // microtask: the .then must run before we assert it
    check('promise resolved with values', resolved, { label: 'A2', type: 'target' });
    check('dialog inactive after confirm', ctl.isActive(), false);
    check('confirm without dialog is null', ctl.confirm(), null);
}

{
    const cy = buildGraph();
    const api = buildScope(cy);
    const ctl = api.createDialogController();
    let rejected = null;
    const promise = ctl.open({ title: 'x', fields: [] }).catch(e => { rejected = e; });
    ctl.cancel();
    await promise;
    check('cancel rejects with cancelled', rejected, 'cancelled');
    check('cancel without dialog is null', ctl.cancel(), null);
    check('dialog inactive after cancel', ctl.isActive(), false);
}

// ===== Test C: node search matcher (min 2 chars, id + label, CI) =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    check('short query matches nothing', api.matchNodes('a').length, 0);
    check('empty query matches nothing', api.matchNodes('').length, 0);
    check('id match is case-insensitive', api.matchNodes('n_b').length, 1);
    check('label match works', api.matchNodes('cherry').length, 1);
    check('no match returns empty', api.matchNodes('zebra').length, 0);
    check('two-char prefix matches label', api.matchNodes('ap').length, 1);
}

// ===== Test D: search stepping APPLIES each match (select + center) =====
// (the reported bug: navigation only bumped the 1/N counter — the shared
// cycleSearchMatch helper now drives the ▲/▼ buttons AND the ↑/↓ keys and
// applies every step)
{
    const api = buildSearchScope();
    api.setMatches(['n1', 'n2', 'n3']);
    api.cycle(1);   // 1/3 -> 2/3
    check('next applies match 2', api.getApplied(), [1]);
    api.cycle(1);   // -> 3/3
    api.cycle(1);   // wraps -> 1/3
    check('wrap-around cycle order', api.getApplied(), [1, 2, 0]);
    api.cycle(-1);  // wraps back -> 3/3
    check('previous wraps to the last match', api.getApplied(), [1, 2, 0, 2]);
    check('counter tracked every step', api.getCounts(), ['2/3', '3/3', '1/3', '3/3']);
    // no matches: a no-op, never applies
    const before = api.getApplied().slice();
    api.setMatches([]);
    api.cycle(1);
    api.cycle(-1);
    check('empty matches never apply', api.getApplied(), before);
}

console.log(failures === 0 ? 'ALL UI-FEEDBACK TESTS PASSED' : failures + ' UI-FEEDBACK TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);

})();
