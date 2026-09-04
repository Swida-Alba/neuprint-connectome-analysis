// Layout-transform harness: extracts the REAL gap/rotation transform
// functions from a generated vispath network HTML and runs them against
// headless Cytoscape. The gap is an ABSOLUTE center-to-center distance
// (median neighbor distance along one axis) and both transforms anchor on
// the visible centroid, which neither moves — so rotation conserves every
// pairwise distance and the two transforms compose without drift.
// Usage: node layout_transform_harness.js <node-modules-dir> <path-to-network.html>
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
    'isVisibleElement', 'visibleNodeCentroid', 'measureAxisGap',
    'applyNodeGap', 'applyRotationDelta', 'syncRotateDisplay',
    'syncTransformInputs',
];

// NOTE: sources come from the project's own generated HTML (trusted,
// locally-produced artifact); new Function only executes that code against
// the headless core + stubs.
function buildScope(cy) {
    const fnSources = FUNCTIONS.map(f => extractFunction(f, html)).join('\n');

    const prelude = `
        // Transform trackers (the real page declares these at script top
        // level, next to the other layout state).
        let lastGapX = null;
        let lastGapY = null;
        let baselineGapX = null;
        let baselineGapY = null;
        let lastRotationDeg = 0;
        let pendingTransformState = null;
        function updateHoverInfo(msg) {}
        function pushHistory(label) {}
        function captureState() { return {}; }
        const els = {};
        function makeEl(id) {
            const el = { id: id, value: '', textContent: '', style: {} };
            els[id] = el;
            return el;
        }
        const document = {
            getElementById: function (id) { return els[id] || makeEl(id); }
        };
    `;

    const src = prelude + fnSources + `
        return {
            applyNodeGap, applyRotationDelta, measureAxisGap, visibleNodeCentroid,
            syncTransformInputs,
            getLastGapX: () => lastGapX,
            getLastGapY: () => lastGapY,
            getLastRotationDeg: () => lastRotationDeg,
        };
    `;

    const api = new Function('cy', src)(cy);
    return api;
}

// An L-shaped 2-column/2-row graph:
//   A(0,0)  B(100,0)
//   C(0,50)
// x-coords {0,100,200}: diffs 100,100 -> median x gap 100
// y-coords {0,50}:      diff 50       -> median y gap 50
function buildGraph() {
    // NOTE: the headless core ignores `position` inside constructor element
    // definitions, so coordinates are applied via .position() after creation.
    const cy = cytoscape({
        headless: true,
        styleEnabled: true,
        elements: [
            { data: { id: 'A', node_type: 'source', label: 'A' } },
            { data: { id: 'B', node_type: 'intermediate', label: 'B' } },
            { data: { id: 'C', node_type: 'target', label: 'C' } },
            { group: 'edges', data: { id: 'e1', source: 'A', target: 'B', weight: 5 } },
            { group: 'edges', data: { id: 'e2', source: 'A', target: 'C', weight: 5 } },
        ],
    });
    const coords = { A: [0, 0], B: [100, 0], C: [0, 50] };
    for (const [id, [x, y]] of Object.entries(coords)) {
        cy.getElementById(id).position({ x: x, y: y });
    }
    return cy;
}

function pos(cy, id) {
    const p = cy.getElementById(id).position();
    return [p.x, p.y];
}

let failures = 0;
function check(name, got, expected) {
    const g = JSON.stringify(got);
    const e = JSON.stringify(expected);
    const pass = g === e;
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | got=' + g + (pass ? '' : ' expected=' + e));
    if (!pass) failures++;
}

const EPS = 1e-6;
function close(got, expected) {
    if (got.length !== expected.length) return false;
    return got.every((v, i) => Math.abs(v - expected[i]) < EPS);
}
function checkClose(name, got, expected) {
    const pass = close(got, expected);
    console.log((pass ? 'PASS' : 'FAIL') + ' | ' + name + ' | got=' + JSON.stringify(got)
        + (pass ? '' : ' expected=' + JSON.stringify(expected)));
    if (!pass) failures++;
}

// ===== Test A: measureAxisGap = median neighbor distance =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    checkClose('x gap measured', [api.measureAxisGap('x')], [100]);
    checkClose('y gap measured', [api.measureAxisGap('y')], [50]);
}

// ===== Test B: setting the gap targets the ABSOLUTE distance =====
// (anchored on the centroid: the two columns move symmetrically around it)
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.applyNodeGap('x', 200);
    // centroid x = 33.33; A/C share a column at -33.33, B lands at 166.67
    checkClose('B column at 166.67', pos(cy, 'B'), [166.66666666666666, 0]);
    checkClose('C shares the left column', [pos(cy, 'C')[0]], [pos(cy, 'A')[0]]);
    checkClose('column distance is exactly 200',
        [pos(cy, 'B')[0] - pos(cy, 'A')[0]], [200]);
    checkClose('measured x gap is 200', [api.measureAxisGap('x')], [200]);
    check('tracker holds absolute gap', api.getLastGapX(), 200);
    // y untouched by the x change
    checkClose('y gap conserved', [api.measureAxisGap('y')], [50]);
}

// ===== Test C: rotation conserves every pairwise distance =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.applyNodeGap('x', 200);  // widen first so composition is non-trivial
    const ids = ['A', 'B', 'C'];
    const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
    const before = ids.map(id => pos(cy, id));
    const totalBefore =
        dist(before[0], before[1]) + dist(before[0], before[2]) + dist(before[1], before[2]);
    api.applyRotationDelta(37);
    const after = ids.map(id => pos(cy, id));
    const totalAfter =
        dist(after[0], after[1]) + dist(after[0], after[2]) + dist(after[1], after[2]);
    checkClose('pairwise distances conserved by rotation', [totalAfter], [totalBefore]);
    const cenB = api.visibleNodeCentroid();
    api.applyRotationDelta(0);  // rotate back by the negative delta
    checkClose('rotate back restores geometry', pos(cy, 'A'), before[0]);
    checkClose('centroid stable through round trip', [cenB.x, cenB.y],
        [api.visibleNodeCentroid().x, api.visibleNodeCentroid().y]);
}

// ===== Test D: gap survives rotation (axes swap at 90 degrees) =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.applyNodeGap('x', 300);
    api.applyRotationDelta(90);
    // The former x gaps are now y gaps: measured y gap = 300
    checkClose('x gap conserved as y gap after 90deg', [api.measureAxisGap('y')], [300]);
    check('rotation tracker at 90', api.getLastRotationDeg(), 90);
    // And the spinner sync reflects the measured swap
    api.syncTransformInputs();
    checkClose('synced tracker y = 300', [api.getLastGapY()], [300]);
}

// ===== Test E: gap change AFTER rotation works around the centroid =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.applyNodeGap('x', 200);
    api.applyRotationDelta(90);
    // After a 90deg rotation the former x gaps live on the y axis; setting
    // the y gap must scale them without touching the (now vertical) chain
    const yBefore = api.measureAxisGap('y');
    api.applyNodeGap('y', 100);
    checkClose('y gap retargeted after rotation', [api.measureAxisGap('y')], [100]);
    check('y tracker updated', api.getLastGapY(), 100);
    check('x measurement unchanged by y edit', api.getLastGapX() !== null, true);
    void yBefore;
}

// ===== Test F: hidden nodes excluded from measure and transform =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    cy.getElementById('B').addClass('hidden');
    // visible = A(0,0), C(0,50): x gap collapses to 0 (single column)
    checkClose('hidden excluded from measurement', [api.measureAxisGap('x')], [0]);
    api.applyNodeGap('y', 100);
    // centroid y = 25; scaling by 2 sends A to -25 and C to +75
    checkClose('y scales around centroid', pos(cy, 'A'), [0, -25]);
    checkClose('visible C scaled', pos(cy, 'C'), [0, 75]);
    checkClose('hidden B keeps its position', pos(cy, 'B'), [100, 0]);
}

// ===== Test G: guards — no target, single row/column, identity =====
{
    const cy = buildGraph();
    const api = buildScope(cy);
    api.applyNodeGap('x', 0);    // rejected
    api.applyNodeGap('x', -5);   // rejected
    api.applyNodeGap('x', NaN);  // rejected
    checkClose('invalid targets are no-ops', pos(cy, 'B'), [100, 0]);
    // collapse to a single column: x gap can no longer be derived
    cy.getElementById('B').position({ x: 0, y: 0 });
    checkClose('single column measures 0', [api.measureAxisGap('x')], [0]);
    api.applyNodeGap('x', 250);
    checkClose('single-column set is a no-op', pos(cy, 'C'), [0, 50]);
    check('tracker unchanged by no-op', api.getLastGapX(), null);
    api.applyRotationDelta(0);
    checkClose('identity rotation is a no-op', pos(cy, 'B'), [0, 0]);
}

console.log(failures === 0 ? 'ALL LAYOUT-TRANSFORM TESTS PASSED' : failures + ' LAYOUT-TRANSFORM TEST(S) FAILED');
// process.exitCode alone would hang: styleEnabled cytoscape cores keep
// background timers alive, so exit explicitly with the proper code.
process.exit(failures === 0 ? 0 : 1);
