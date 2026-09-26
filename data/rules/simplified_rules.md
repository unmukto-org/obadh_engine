# Simplified Transliteration Rules

This document outlines the core rules and special cases for Bengali transliteration in the Obadh Engine.

## Core Approach

This implementation takes a structured approach to Bengali transliteration, focusing on:

### 1. Vowel Handling

Vowels are handled in different modes based on their position and context:

- **Independent Vowels**: When vowels appear on their own
  - Signal: `a` → `আ`, `A` → `আ`, `aY` → `অ্যা`, `AY` → `অ্যা`, `i` → `ই`, `I` → `ঈ`, `u` → `উ`, `U` → `ঊ`, `e` → `এ`, `E` → `এ`, `O` → `ও`
  - Repeated lowercase signal: `aa` → `আআ`, `ee` → `এএ`, `ii` → `ইই`, `oo` → `অঅ`, `uu` → `উউ`

- **Vowel Modifiers**: When vowels follow consonants
  - Signal: `ka` → `কা`, `kaY` → `ক্যা`, `ki` → `কি`, `kI` → `কী`, `ku` → `কু`, `kU` → `কূ`, `ke` → `কে`
  - Signal: `kaby` → `কাব্য`, `kAby` → `কাব্য`
  - Repeated lowercase signal after consonants: `kaa` → `কাআ`, `kee` → `কেএ`, `kii` → `কিই`, `koo` → `কঅ`, `kuu` → `কুউ`
  - Signal: `tiyw` → `তীয়`, `jatiywta` → `জাতীয়তা`

- **Inherent Vowel**: The implied 'অ' sound after consonants
  - Signal: `k` → `ক` (inherently includes the 'অ' sound)

### 2. Consonant Handling

Consonants are handled in different modes:

- **Independent**: Individual consonants with inherent vowel
  - Signal: `k` → `ক`, `kh` → `খ`
  - Aspirated aliases accept documented titlecase/all-caps forms: `Kh`/`KH` → `খ`, `Chh`/`CHH` → `ছ`
  - Missing one-letter alphabetic case variants fall back internally to the exact opposite-case rule signal when that opposite-case signal exists and the typed case is unclaimed. Today this admits `B`, `G`, `K`, `P`, `F`, `V`, `L`, and `H`.
  - Exact signals are never overridden by fallback: `T`, `D`, `N`, `S`, `I`, `U`, `O`, `Y`, `M`, and narrow `Z` keep their documented meanings

- **With Vowel Modifiers**: Consonants followed by explicit vowels
  - Signal: `ka` → `কা`, `ki` → `কি`, `KHi` → `খি`, `CHHi` → `ছি`
  - Case fallback composes before vowel attachment: `Biggan` → `বিজ্ঞান`, `Khela` → `খেলা`, `Ga` → `গা`

- **Conjuncts**: Multiple consonants combined with hasant
  - Signal: `kk` → `ক্ক`, `kt` → `ক্ত`
  - Created by adding hasant (্) between consonants
  - Aspirated aliases compose through canonical conjunct rules: `KHy` → `খ্য`, `acCHHa` → `আচ্ছা`
  - জ্ঞ can be typed by component signal `jNG`, shorthand `jn`, or pronounced shorthand `gg`: `jnan`/`ggan` → `জ্ঞান`

### 3. Special Reph Form

- Double 'r' creates the special reph form
  - Signal: `rrm` → `র্ম`
  - This is different from a normal sequence: `rm` → `রম`
  - Reph form is created by adding hasant (্) between র and the following consonant (e.g. `rrm` → `র্ম`, `rrk` → `র্ক`)
  - A following explicit hasant is redundant before a reph target: `rr,,ka` → `র্কা`
  - Reph can compose over a valid following conjunct cluster without a separate word rule: `rrkSh` → `র্ক্ষ`, `rrsk` → `র্স্ক`
  - The same cluster can be typed with an explicit hasant boundary: `rrk,,Sh` → `র্ক্ষ`
  - <code>rrt</code> is reserved for `র্ত`; use composable <code>rrt``</code> for `র্ৎ`

### 4. Numeric Characters

Bengali has its own numerals that are mapped directly from Latin numerals:

| Latin | Bengali |
|-------|---------|
| 0     | ০       |
| 1     | ১       |
| 2     | ২       |
| 3     | ৩       |
| 4     | ৪       |
| 5     | ৫       |
| 6     | ৬       |
| 7     | ৭       |
| 8     | ৮       |
| 9     | ৯       |

- Numerals are transliterated directly to their Bengali equivalents
- They do not participate in conjunct formation or vowel modification
- Signal: `123` → `১২৩`, `k2` → `ক২`

## Special Rules

### Hasant Handling

- Represented in Obadh input as `,,`
  - Signal: `k,,` → `ক্`
- Between consonants, `,,` is an explicit conjunct command:
  - Signal: `k,,k` → `ক্ক`
- A trailing explicit hasant after a formed conjunct remains visible:
  - Signal: `k,,k,,` → `ক্ক্`
  - Signal: `n,,d,,r,,` → `ন্দ্র্`
- As a standalone marker, `,,` renders the hasant directly:
  - Signal: `,,` → `্`
- A vowel typed after an explicit dead consonant is independent, not a dependent kar on that dead consonant:
  - Signal: `k,,a` → `ক্আ`
  - Signal: `k,,i` → `ক্ই`
- A hasant suppresses the **inherent** vowel, so it needs a consonant to sit on. The inherent
  vowel is still a valid target, but once an explicit kar has been written there is nothing left
  to suppress and the signal is dropped:
  - Signal: `ko,,` → `ক্`
  - Signal: `ka,,` → `কা`
  - Signal: `kOI,,` → `কৈ`
  - Signal: `ka,,k` → `কাক`
- The same holds after any other sign that cannot carry a hasant — chandrabindu, anusvar,
  bisarga, khanda ta, a numeral, or a hasant already in place:
  - Signal: `k^,,` → `কঁ`
  - Signal: `kng,,` → `কং`
  - Signal: `k:,,` → `কঃ`
  - Signal: <code>kt``,,</code> → `কৎ`
  - Signal: `k1,,` → `ক১`
  - Signal: `rr,,` → `র্`
- Khanda-ta composes with reph without a separate word rule:
  - Signal: <code>t``</code> → `ৎ`
  - Signal: <code>rrt``</code> → `র্ৎ`
  - Uppercase <code>T``</code> remains an accepted alias.
  - A following explicit hasant is redundant before another consonant because <code>t``</code> is already the dead খণ্ড ত form: <code>t``,,sa</code> → `ৎসা`

### Diacritic Marks

- Chandrabindu is represented by `^`
  - Signal: `kA^` → `কাঁ`
- Bisarga is represented by `:`
  - Signal: `ku:` → `কুঃ`
- Trailing diacritic marks are explicit ordered marks. The engine preserves the order typed by the user:
  - Signal: `kA^:` → `কাঁঃ`
  - Signal: `kA:^` → `কাঃঁ`
- A vowel typed after an already rendered mark starts independently:
  - Signal: `k^a` → `কঁআ`
  - Signal: `k:a` → `কঃআ`

### Nasal Signals

Nasal input is deterministic and must preserve the user's intended spelling:

- `ng` is anusvar before a consonant or at a word end:
  - Signal: `bangla` → `বাংলা`
  - Signal: `songket` → `সংকেত`
- `ng` before a vowel folds to the velar nasal ঙ and takes the vowel as a kar,
  since an anusvar cannot carry a vowel, so it matches `Ng` + vowel:
  - Signal: `bhanga` → `ভাঙা`
  - Signal: `bangali` → `বাঙালি`
  - Signal: `rongin` → `রঙিন`
- `M` is the explicit anusvar escape:
  - Signal: `sMgo` → `সংগ`
  - Use it before `g`/`gh` when you want literal ংগ/ংঘ rather than the `ngg`/`nggh` shorthand.
- `Ng` is the velar nasal consonant:
  - Signal: `oNgko` → `অঙ্ক`
  - Signal: `oNggo` → `অঙ্গ`
- `ngg` and `nggh` are shorthand for deliberate velar nasal conjuncts:
  - Signal: `bonggo` → `বঙ্গ`
  - Signal: `ngghAt` → `ঙ্ঘাত`
- `NGj` is the source signal for the palatal nasal-ja conjunct, and `nj`/`nJ`
  are narrow ergonomic aliases for the same ঞ + জ cluster:
  - Signal: `jinjira` → `জিঞ্জিরা`
  - Signal: `noj` → `নজ` when the intended spelling is ন + জ with an inherent vowel
  - Signal: `nz` remains `নয`; correction-layer `nz` repairs do not change the
    deterministic core rule

### 'o' as a Blocker

The 'o' character serves special functions:
- Acts as conjunct blocker
  - Signal: `kok` → `কক` (prevents 'k' from forming a conjunct with the following 'k')
- Acts as the deliberate inherent-vowel signal before a following cluster
  - Signal: `bhokt` → `ভক্ত`, `shokti` → `শক্তি`
- Acts as vowel-modifier blocker

### য-ফলা and ব-ফলা Handling

There's special handling for য-ফলা and ব-ফলা:

- **Regular consonants** use 'z' and 'b': 
  - `z` → `য` (ya)
  - `b` → `ব` (ba)

- **Phola forms** (conjunct versions) use 'y' and 'w':
  - `ky` → `ক্য` (k + hasant + ya)
  - `kw` → `ক্ব` (k + hasant + ba)
  - `zy` / `zY` → `য্য` (regular `z` base + ya-phola marker)
  - `bw` → `ব্ব` (regular `b` base + ba-phola marker)
  - Valid phola clusters may also be typed with an explicit boundary: `k,,y` → `ক্য`, `k,,w` → `ক্ব`, `m,,w,,r` → `ম্ব্র`
  - `mw` → `ম্ব`, `mwr` → `ম্ব্র`
  - `y`/`Y` (য-ফলা) compose onto any consonant or conjunct base, so loanword clusters form too: `ply` → `প্ল্য`, `plYan` → `প্ল্যান`, `blYak` → `ব্ল্যাক`, `TrYak` → `ট্র্যাক`. A standalone `r`/`R`/`Rh`/`Ng` refuses ya-phola (`rya` → `রয়া`), but an r-phola tail inside a conjunct is not a standalone `r` and does take it (`krYak` → `ক্র্যাক`). A base already ending in a phola marker takes no further phola (`Swy` → `শ্বয়`)
  - `w` is the ব-ফলা marker only inside a valid conjunct cluster; standalone it is the ওয় glide, e.g. `waTar` → `ওয়াটার` (see Foreign-Sound Letters). Invalid explicit clusters remain decomposed
  - After a short-i-bearing consonant or conjunct, `iyw` is a deliberate long-ঈয় signal rather than a ব-ফলা command: `tiyw` → `তীয়`, `ktiYwta` → `ক্তীয়তা`
  - Lowercase `o` after that signal stays an inherent-vowel terminator (`kiywo` → `কীয়`); uppercase `O` gives visible ও-কার (`kiywO` → `কীয়ো`)
  - `aY`/`AY` are the deliberate অ্যা vowel signals: `aYp` / `AYp` → `অ্যাপ`; lowercase `ayp` remains `আয়প`

- **Regular `z` and `b` do not become phola markers by themselves**:
  - `kz` → `কয` (k + ya, no conjunct)
  - `kb` → `কব` (k + ba, no conjunct)
  - `zoy` → `যয়` (the `o` terminator blocks `zy`)

### Foreign-Sound Letters

Letters with no native Bengali phoneme map to their settled convention instead of leaking ASCII:

- `q` → `ক` (qaf → ka): `iraq` → `ইরাক`, `qatar` → `কাতার`
- `qq` → `ঁ` (চন্দ্রবিন্দু); matched ahead of `q` by longest prefix, so `baqq` → `বাঁ`
- `x` → `ক্স`: `box` → `বক্স`, `fix` → `ফিক্স`
- `w` standalone → `ওয়` glide (`waTar` → `ওয়াটার`); after a consonant base it is the ব-ফলা marker (`kw` → `ক্ব`)

### Reph-ya `র‍্য` Marker

The reph-ya loanword form `র‍্য` (distinct from the reph conjunct `র্য`) is marked by `Y` or `Z` after a leading `র`. Lowercase `y` stays the `য়` glide and composes after the marker:

- `rrYa` → `র্যা` (reph conjunct `র্য` plus vowel)
- `rY` / `rZ` → `র‍্য` (`র` + U+200D + hasant + `য`)
- `rYa` / `rZa` → `র‍্যা`; `rYab` / `rZab` → `র‍্যাব` (the RAB word)
- `rZy` / `rYy` → `র‍্যয়` (marker `র‍্য` plus the `য়` glide)

`Z` is intentionally narrow here. It is not a general compatibility ya-phola alias and does not rewrite unrelated sequences such as `Zya` or `kZya`.
