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
    'isVisibleElement',
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

console.log(failures === 0 ? 'ALL UI-FEEDBACK TESTS PASSED' : failures + ' UI-FEEDBACK TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);

})();
