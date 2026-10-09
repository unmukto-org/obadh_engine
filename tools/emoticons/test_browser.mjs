// Run after cargo build --features cli and ./build.sh wasm.
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, mkdtempSync, rmSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';

const root = fileURLToPath(new URL('../../', import.meta.url));
const read = (name) => readFileSync(join(root, name));
const moduleSource = read('www/js/obadh_engine.js').toString();
const wasm = await import('data:text/javascript;base64,' + Buffer.from(moduleSource).toString('base64'));
wasm.initSync({module: read('www/js/obadh_engine_bg.wasm')});
const engine = new wasm.ObadhaWasm();
const directory = mkdtempSync(join(tmpdir(), 'obadh-emoticon-browser-'));
const backends = [];
try {
  const source = join(directory, 'lexicon.tsv');
  const artifact = join(directory, 'lexicon.fst');
  writeFileSync(source, 'বাংলা\t100\n');
  execFileSync(join(root, 'target/debug/obadh-autocorrect'), [
    'build-fst-lexicon', '--input', source, '--output', artifact,
  ]);
  backends.push(wasm.ObadhAutocorrectWasm.fromTsv('বাংলা\t100\n'));
  backends.push(wasm.ObadhAutocorrectWasm.fromFstLexicon(readFileSync(artifact)));
  const groups = JSON.parse(read('data/emoticons/emoticons.json'));
  let aliases = 0;
  for (const group of groups) {
    for (const alias of group.emoticons) {
      assert.equal(engine.emoticonEmoji(alias), group.emoji);
      for (const backend of backends) {
        const result = backend.suggest(alias);
        assert.equal(result.obadh_output, alias);
        assert.equal(result.replacement, undefined);
        assert.deepEqual(result.candidates.map((candidate) => candidate.text), [alias, group.emoji]);
        assert.equal(result.candidates[1].source, 'emoticon_exact');
        assert.equal(result.candidates[1].frequency, 0);
      }
      aliases++;
    }
  }
  let copiedText;
  let createApp;
  const sandbox = {
    window: {obadhaWasm: engine},
    document: {addEventListener: (_name, callback) => callback()},
    Alpine: {data: (_name, factory) => { createApp = factory; }},
    navigator: {clipboard: {writeText: async (text) => { copiedText = text; }}},
    console, setTimeout, clearTimeout,
  };
  const script = read('www/index.html').toString().match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInNewContext(script, sandbox);
  const app = createApp();
  app.activeInputElement = () => null;
  app.wasmLoaded = true;
  app.$nextTick = () => {};
  app.doTransliterate = () => {};
  app.learnAutosuggestCommit = () => {};
  for (const [alias, emoji] of [[':)', '😃'], [':-)', '😃'], [':D', '😄'], [':-D', '😄']]) {
    app.inputText = 'আমি ' + alias;
    app.ribbonSelectedText = null;
    assert.equal(app.activeProbeRange().text, alias);
    assert.equal(app.computeDraftOutput(), alias);
    app.autocorrectResult = backends[0].suggest(alias);
    assert.equal(app.correctionFeedRows()[0].text, emoji);
    assert.equal(app.applyContextPriorsToAutocorrect(app.autocorrectResult), app.autocorrectResult);
    app.commitActiveDraft(null, true);
    assert.equal(app.inputText, 'আমি ' + alias + ' ');
    app.inputText = 'আমি ' + alias;
    app.commitActiveDraft({text: emoji, source: 'emoticon_exact'}, true);
    assert.equal(app.inputText, 'আমি ' + emoji + ' ');
    app.inputText = 'আমি ' + alias;
    app.ribbonSelectedText = null;
    assert.equal(app.ribbonRows()[0].text, alias);
    app.moveRibbonSelection(-1);
    assert.equal(app.ribbonRows()[app.ribbonSelectedIndex()].text, emoji);
    app.moveRibbonSelection(1);
    assert.equal(app.ribbonSelectedIndex(), 0);
    let prevented = false;
    app.handleComposerKeydown({key: 'Tab', preventDefault: () => { prevented = true; }});
    assert.equal(prevented, true);
    assert.equal(app.ribbonSelectedIndex(), 1);
    app.handleComposerKeydown({key: ' ', preventDefault: () => {}});
    assert.equal(app.inputText, 'আমি ' + emoji + ' ');
  }
  assert.equal(app.transliterateComposerText('ami :) ami 😄 ami'), 'আমি :) আমি 😄 আমি');
  assert.equal(app.transliterateComposerText('১২.৩৪  ami\n:-D'), '১২.৩৪  আমি\n:-D');
  assert.equal(engine.transliterate(':D'), 'ঃড');
  app.inputText = 'বাংলা 😄';
  await app.copyComposer();
  assert.equal(copiedText, app.inputText);
  assert.equal(app.copyNotice, 'Copied');
  console.log(`Passed ${aliases} aliases across both WASM backends, compose recognition, literal defaults, emoji preservation, and ribbon keyboard selection.`);
} finally {
  for (const backend of backends) backend.free();
  engine.free();
  rmSync(directory, {recursive: true, force: true});
}
