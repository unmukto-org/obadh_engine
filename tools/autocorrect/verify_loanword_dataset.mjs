// Verify the reviewed data through the shipped native and browser runtimes.
// Prerequisites: cargo build --features cli; ./build.sh wasm.
// Add --verify-sources to fetch and check pinned localization snapshots.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {createReadStream, mkdtempSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createInterface} from 'node:readline';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const dataset = join(root, 'data/autocorrect');
const read = (path) => readFileSync(join(root, path));
const normalize = (text) => text.normalize('NFC');
const manifest = JSON.parse(read('data/autocorrect/lexicons/loanwords/verified_expansion_2026-10-09.json'));
assert(process.argv.slice(2).every((arg) => arg === '--verify-sources'), 'Usage: node tools/autocorrect/verify_loanword_dataset.mjs [--verify-sources]');
const pairs = (file, columns) => {
  const lines = readFileSync(file, 'utf8').trimEnd().split('\n');
  assert.equal(lines.shift(), columns.join('\t'));
  const rows = lines.map((line) => line.split('\t'));
  const keys = new Set();
  for (const row of rows) {
    assert.equal(row.length, columns.length, `Malformed row in ${file}`);
    const key = `${row[1]}\t${normalize(row[0])}`;
    assert(!keys.has(key), `Duplicate normalized pair: ${key}`);
    keys.add(key);
  }
  return keys;
};
const runtimePairs = pairs(join(dataset, 'lexicons/loanwords/en_bn_loanwords.tsv'), ['bangla', 'english']);
const categorizedPairs = pairs(join(dataset, 'lexicons/loanwords/en_bn_loanwords_categorized.tsv'), ['bangla', 'english', 'category']);
const categories = new Map(read('data/autocorrect/lexicons/loanwords/en_bn_loanwords_categorized.tsv').toString().trimEnd().split('\n').slice(1).map((line) => {
  const [bangla, english, category] = line.split('\t');
  return [`${english}\t${normalize(bangla)}`, category];
}));
const reviewedPairs = new Set();
for (const entry of manifest.entries) {
  const key = `${entry.english}\t${normalize(entry.bangla)}`;
  assert(!reviewedPairs.has(key), `Duplicate review: ${key}`);
  reviewedPairs.add(key);
  assert(runtimePairs.has(key), `Reviewed pair missing from runtime TSV: ${key}`);
  assert(categorizedPairs.has(key), `Reviewed pair missing from categorized TSV: ${key}`);
  assert.equal(categories.get(key), entry.category, `Audit category differs from metadata: ${key}`);
  assert(entry.corpus_frequency > 0, `Missing corroborating corpus usage: ${key}`);
  assert(entry.sources.length > 0, `Missing published evidence: ${key}`);
  for (const source of entry.sources) assert(manifest.sources[source]?.url.startsWith('https://'));
}
for (const entry of manifest.excluded_pairs || []) {
  const key = `${entry.english}\t${normalize(entry.bangla)}`;
  assert(!runtimePairs.has(key) && !categorizedPairs.has(key), `Excluded spelling reintroduced: ${key}`);
}

if (process.argv.includes('--verify-sources')) {
  const sourceIds = [...new Set(manifest.entries.filter((entry) => entry.localization_message_id).flatMap((entry) => entry.sources))];
  const snapshots = await Promise.all(sourceIds.map(async (id) => {
    const source = manifest.sources[id];
    assert(source.snapshot_sha256, `Missing snapshot hash: ${id}`);
    const response = await fetch(source.url + '?format=TEXT', {signal: AbortSignal.timeout(30000)});
    assert(response.ok, `${id}: HTTP ${response.status}`);
    const bytes = Buffer.from(await response.text(), 'base64');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), source.snapshot_sha256, `Published localization changed: ${id}`);
    const messages = new Map([...bytes.toString('utf8').matchAll(/<translation id="(\d+)">([\s\S]*?)<\/translation>/g)].map((match) => [match[1], normalize(match[2].replace(/<[^>]+>/g, ''))]));
    return [id, messages];
  }));
  const messages = new Map(snapshots);
  for (const entry of manifest.entries.filter((entry) => entry.localization_message_id)) {
    assert(entry.sources.some((id) => messages.get(id)?.get(entry.localization_message_id)?.includes(normalize(entry.bangla))), `Localization message does not attest ${entry.english} → ${entry.bangla}`);
  }
  console.log(`Verified ${sourceIds.length} pinned localization snapshots and their audited message IDs.`);
}

const corpusCounts = new Map(manifest.entries.map((entry) => [normalize(entry.bangla), 0]));
const corpusHash = createHash('sha256');
const corpusStream = createReadStream(join(dataset, manifest.corpus.path));
corpusStream.on('data', (chunk) => corpusHash.update(chunk));
for await (const line of createInterface({input: corpusStream, crlfDelay: Infinity})) {
  const [word, frequency] = line.split('\t');
  const key = normalize(word);
  if (corpusCounts.has(key)) corpusCounts.set(key, Number(frequency));
}
assert.equal(corpusHash.digest('hex'), manifest.corpus.sha256, 'Corpus evidence snapshot changed; review its counts');
for (const entry of manifest.entries) assert.equal(corpusCounts.get(normalize(entry.bangla)), entry.corpus_frequency, `Incorrect corpus evidence for ${entry.bangla}`);

const cli = join(root, 'target/debug/obadh-autocorrect');
const nativeFst = join(dataset, 'models/bn.fst');
const loanwords = join(dataset, 'models/en_bn_loanwords.fst');
const temp = mkdtempSync(join(tmpdir(), 'obadh-verified-loanwords-'));
let model;
try {
  const rebuilt = join(temp, 'loanwords.fst');
  const report = JSON.parse(execFileSync(cli, ['build-loanword-lexicon', '--input', join(dataset, 'lexicons/loanwords/en_bn_loanwords.tsv'), '--output', rebuilt], {encoding: 'utf8'}));
  assert.equal(report.entries, runtimePairs.size);
  for (const field of ['duplicate_rows', 'malformed_rows', 'empty_rows', 'non_bangla_rows', 'invalid_english_rows']) assert.equal(report[field], 0, field);
  assert.deepEqual(readFileSync(rebuilt), readFileSync(loanwords), 'Dataset model does not reproduce from the runtime TSV');
  assert.deepEqual(readFileSync(loanwords), read('www/assets/autocorrect/en_bn_loanwords.fst'), 'Playground uses a different loanword model');
  const moduleSource = read('www/js/obadh_engine.js').toString();
  const wasm = await import('data:text/javascript;base64,' + Buffer.from(moduleSource).toString('base64'));
  wasm.initSync({module: read('www/js/obadh_engine_bg.wasm')});
  model = wasm.ObadhAutocorrectWasm.fromFstLexiconWithLoanwords(read('www/assets/autocorrect/bn.fst'), readFileSync(loanwords));
  const nativeSuggest = (input) => JSON.parse(execFileSync(cli, ['suggest-fst', '--lexicon', nativeFst, '--loanwords', loanwords, '--input', input, '--response-candidates', '8'], {encoding: 'utf8'}));
  const verifySuggestion = (result, expected, input, surface, source) => {
    const expectedText = normalize(expected);
    if (normalize(result.obadh_output) === expectedText && !source) return;
    const candidate = result.candidates.find((row) => normalize(row.text) === expectedText);
    assert(candidate, `${surface}: missing ${input} → ${expected}`);
    if (source) assert.equal(candidate.source, source, `${surface}: ${input}`);
    const alternatives = result.candidates.filter((row) => normalize(row.text) !== normalize(result.obadh_output)).slice(0, 4);
    assert(alternatives.some((row) => normalize(row.text) === expectedText), `${surface}: ${input} → ${expected} falls outside the five-cell ribbon`);
  };
  for (const entry of manifest.entries) {
    verifySuggestion(nativeSuggest(entry.english), entry.bangla, entry.english, 'CLI');
    verifySuggestion(model.suggest(entry.english), entry.bangla, entry.english, 'WASM');
    verifySuggestion(model.suggest(entry.english.toUpperCase()), entry.bangla, entry.english.toUpperCase(), 'WASM uppercase');
  }
  for (const probe of manifest.fuzzy_probes) {
    assert(!Array.from(runtimePairs).some((pair) => pair.startsWith(probe.input + '\t')), 'Fuzzy typo must not become an exact dictionary key');
    verifySuggestion(nativeSuggest(probe.input), probe.bangla, probe.input, 'CLI fuzzy', 'fst_english_loanword_fuzzy');
    verifySuggestion(model.suggest(probe.input), probe.bangla, probe.input, 'WASM fuzzy', 'fst_english_loanword_fuzzy');
  }
  console.log(`Verified ${manifest.entries.length} reviewed pairs through native CLI and shipped WASM, uppercase inputs, fuzzy probes, artifact reproducibility, and playground model parity (${report.entries} pairs; ${report.unique_english_keys} keys).`);
} finally {
  model?.free();
  rmSync(temp, {recursive: true, force: true});
}
